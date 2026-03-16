from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as transformsF
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader

from data.datasets import AbstractSRDataset
from metrics.metrics import SRMetricSuite
from model.sr_resnet_model import SRResNet
from model.zssr_model import ZSSRConvNet
from data.datasets import AbstractSRDataset
from data.utils import augment, zssr_collate_fn

class AbstractRunner(ABC):
    
    @abstractmethod
    def train(self, dataset: AbstractSRDataset, **kwargs) -> None:
        """
        Trains the model on the provided dataset.
        """
        pass

    @abstractmethod
    def evaluate(self, dataset: AbstractSRDataset, **kwargs) -> dict:
        """
        Predict HR images and return a dict of metrics.
        """
        pass

class SRResNetRunner(AbstractRunner):
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.L1Loss()
        self.learning_rate = 1e-4
        self.model = None
        
        self.metrics = SRMetricSuite(self.device)

    def train(self, dataset, total_iterations, batch_size):
        
        def init_weights(m):
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

        dataloader = DataLoader(dataset, batch_size=batch_size ,shuffle=True)
        self.model = SRResNet().to(self.device) 
        self.model.apply(init_weights)
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, betas=(0.9, 0.999))

        print(f"--- Training SRResNet for {total_iterations} iterations ---")

        self.model.train()
        current_iteration = 0
        while current_iteration < total_iterations:
            if current_iteration >= total_iterations:
                break

            if current_iteration % 5000 == 0:
                self._save_checkpoint(current_iteration, self.model, optimizer, "srresnet_checkpoint_epoch_" + str(current_iteration) + ".pth")

            for lr_im, hr_im in dataloader:                
                lr_patch, hr_true = lr_im.to(self.device), hr_im.to(self.device)
                
                optimizer.zero_grad()
                hr_pred = self.model(lr_patch)
                loss = self.criterion(hr_pred, hr_true)
                loss.backward()
                optimizer.step()

                current_iteration += 1  

            if current_iteration % 100 == 0:
                    print(f"Iteration {current_iteration}/{total_iterations}, Loss: {loss.item():.6f}")
                
            print(f"End of iteration {current_iteration-1}, Loss: {loss.item():.6f}")

    def _save_checkpoint(self, iteration, model, optimizer, path):
        checkpoint = {
            'iteration': iteration,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }
        torch.save(checkpoint, path)

    def _load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        model = SRResNet().to(self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate, betas=(0.9, 0.999))
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        iteration = checkpoint['iteration']
        return model, optimizer, iteration
    
    def _standardize_output(self, hr_pred):
        """
        Convert model output from [-1, 1] to [0, 1] for metric calculation.
        """
        return (hr_pred + 1.0) / 2.0


    def evaluate(self, dataset):
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        
        # TODO: load model from checkpoint if available, else raise error if model is None
        if self.model is None:
            raise ValueError("Model has not been trained yet. Please train the model before evaluation.")
        
        self.model.eval()
        self.metrics.reset() # Clear previous scores
        
        print("--- Evaluating Model ---")
        
        with torch.no_grad():
            for lr_im, hr_im in dataloader:
                lr_patch, hr_true = lr_im.to(self.device), hr_im.to(self.device)
                
                # forward pass
                hr_pred = self.model(lr_patch)
                
                # Standardize both to [0, 1] before metric collection
                # hr_pred is [-1, 1], hr_true is [-1, 1]
                sr_01 = self._standardize_output(hr_pred)
                hr_01 = self._standardize_output(hr_true)
                
                # update metrics
                self.metrics.update(sr_01, hr_01)
        
        # Calculate the final average scores
        results = self.metrics.compute()
        
        print(f"Results -> PSNR: {results['psnr']:.2f} dB, SSIM: {results['ssim']:.4f}")
        return results

    def predict(self, dataset):
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        
        # TODO: load model from checkpoint if available, else raise error if model is None
        if self.model is None:
            raise ValueError("Model has not been trained yet. Please train the model before evaluation.")
        
        self.model.eval()
        self.metrics.reset() # Clear previous scores
        
        print("--- Evaluating Model ---")
        
        with torch.no_grad():
            for lr_im, hr_im in dataloader:
                lr_patch, hr_true = lr_im.to(self.device), hr_im.to(self.device)
                
                # forward pass
                hr_pred = self.model(lr_patch)
                
                # Standardize both to [0, 1] before metric collection
                # hr_pred is [-1, 1], hr_true is [-1, 1]
                sr_01 = self._standardize_output(hr_pred)
                hr_01 = self._standardize_output(hr_true)
                
                # update metrics
                self.metrics.update(sr_01, hr_01)
        
        # Calculate the final average scores
        results = self.metrics.compute()
        
        print(f"Results -> PSNR: {results['psnr']:.2f} dB, SSIM: {results['ssim']:.4f}")
        return results


class ZSSRRunner(AbstractRunner):
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.L1Loss()
        self.learning_rate = 1e-3
        self.test_img: torch.Tensor = None
        self.model: ZSSRConvNet = None
        self.out_size: torch.Size = None

        self.history = {
            'loss': [],
            'grad_mag': []
        }
        self.metrics = SRMetricSuite(self.device)

    def train(self, dataset: AbstractSRDataset, out_size: torch.Size, n_epochs=10, n_scale_factors=6) -> None:
        """
        Trains the model on the internal patches of the test image.
        """

        # Keep test image for intermediate HR fathers and final super-resolution
        self.test_img = dataset.strategy.base_img.unsqueeze(0).to(self.device)
        self.out_size = out_size

        dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=zssr_collate_fn)
        self.model = ZSSRConvNet().to(self.device) 
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = LinearFitLossLR(optimizer)

        print(f"--- Training ZSSR for {n_epochs} epochs ---")

        # Intermediate scale factors to ease learning
        scale_factors = np.linspace(1.0, dataset.scale_factor, n_scale_factors+1)[1:]

        for s_i in scale_factors:
            # Set initial learning rate and current scale factor
            self._reset_lr(optimizer)
            dataset.curr_s_i = s_i
            self.model.train()
            print(f"--- Training ZSSR with s_i={s_i} ---")

            for epoch in range(n_epochs):
                epoch_loss = 0.0
                epoch_grad = 0.0

                for i, (lr_patch, hr_patch) in enumerate(dataloader):   
                    # Training step             
                    lr_patch, hr_true = lr_patch.to(self.device), hr_patch.to(self.device)
                    hr_spatial_dims = hr_true.shape[-2:]

                    lr_patch += self._compute_noise(lr_patch.shape)
                    
                    optimizer.zero_grad()
                    hr_pred = self.model(lr_patch, hr_spatial_dims)
                    loss = self.criterion(hr_pred, hr_true)
                    loss.backward()
                    optimizer.step()
                    scheduler.step(loss.item())

                    epoch_grad += self._compute_grad_mag(self.model)
                    epoch_loss += loss.item()
            
                self.history['grad_mag'].append(epoch_grad / len(dataloader))
                self.history['loss'].append(epoch_loss / len(dataloader))
                    
                print(f"End of epoch {epoch}, Loss: {loss.item():.6f}")

            # Generation of intermediate HR for next scale factor s_{i+1}
            self.model.eval()
            with torch.no_grad():
                intermediate_hr = self._generate_intermediate_hr(self.model, self.test_img, s_i).detach().cpu()
                dataset.add_image(intermediate_hr)

import torch

def evaluate(self, hr_true: torch.Tensor, save_hr: bool = True) -> dict | tuple[dict, torch.Tensor]:
    self.model.eval()
    with torch.no_grad():
        hr_pred = self._predict()
        self.metrics.update(hr_pred, hr_true)
    
    results = self.metrics.compute()
    if save_hr:
        return results, hr_pred
    return results

    def _reset_lr(self, optimizer: optim.Optimizer):
        """
        Reset optimizer's learning rate to initial learning rate.
        """
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.learning_rate

    def _compute_noise(self, lr_size: torch.Size) -> torch.Tensor:
        """
        Compute additional noise to add in current LR patch.
        """
        noise_std = 5.0 / 255.0
        return torch.randn(lr_size, device=self.device) * noise_std

    def _compute_grad_mag(self, model: ZSSRConvNet) -> float:
        """
        Compute gradient's magnitude for given model to return the 'intensity' of the weights' update.
        """
        grad_mag = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                grad_mag += param_norm.item() ** 2
        return grad_mag ** 0.5

    def _generate_intermediate_hr(self, model: torch.nn.Module, test_img: torch.Tensor, s_i: float) -> torch.Tensor:
        """
        Generate intermediate HR_i given the current intermediate scale factor s_i. 
        """        
        h, w = test_img.shape[-2:]
        new_hr_size = (int(h * s_i), int(w * s_i))
        return model(test_img, new_hr_size).squeeze(0)

    def _predict(self) -> torch.Tensor:
        """
        Final prediction that computes the prediction of the 8 augmented LR images and returns the median value,
        as reported in the original ZSSR paper.
        """
        augmented_lr_images = augment(self.test_img) 
        outputs = []
        
        for idx, aug_img in enumerate(augmented_lr_images):  
            # Compute rotation and horizontal flip          
            k = idx // 2
            is_flipped = (idx % 2 == 1)
            
            # Change output size for 90 and 270 degree rotation
            if k % 2 != 0:
                curr_out_size = (self.out_size[1], self.out_size[0])
            else:
                curr_out_size = self.out_size
                
            aug_out = self.model(aug_img, curr_out_size)
            
            if is_flipped:
                aug_out = transformsF.hflip(aug_out)
                
            # Rotate back by -k * 90 to obtain wanted HR image
            aug_out_reversed = torch.rot90(aug_out, -k, dims=[-2, -1]) 
            
            outputs.append(aug_out_reversed)
            
        # Compute median per pixel
        stacked_outputs = torch.stack(outputs)
        final_prediction, _ = torch.median(stacked_outputs, dim=0)
        
        return final_prediction

class LinearFitLossLR(lr_scheduler.LRScheduler):
    """
    Custom Learning Rate Scheduler that periodically fits a linear regression to the recent reconstruction errors (losses).
    If the standard deviation of the errors is greater than the slope by a certain factor, it divides the learning rate by 10.
    """
    def __init__(self, optimizer: optim.Adam, window_size=512, slope_factor=20.0, min_lr=1e-6):
        super().__init__(optimizer)
        self.window_size = window_size
        self.slope_factor = slope_factor
        self.min_lr = min_lr
        self.losses = []

    def step(self, loss: float = None):
        if loss is None:
            return
        
        self.losses.append(loss)
        
        # Update lr only if enough losses
        if len(self.losses) >= self.window_size:
            y = np.array(self.losses)
            x = np.arange(len(y))

            slope, _ = np.polyfit(x, y, deg=1)
            std_dev = np.std(y)

            # Update lr if std_dev is greater than loss' slop of a certain factor
            if std_dev > abs(slope) * self.slope_factor:
                for param_group in self.optimizer.param_groups:
                    new_lr = max(param_group['lr'] / 10.0, self.min_lr)
                    param_group['lr'] = new_lr
                    print(f'new_lr = {new_lr}')
                self.losses = []
            else:
                self.losses.pop(0)
    
    def get_lr(self):
        return [param_group['lr'] for param_group in self.optimizer.param_groups]