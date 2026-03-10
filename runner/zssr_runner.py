import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as transformsF
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from model.zssr_model import ZSSRConvNet
from data.datasets import AbstractSRDataset
from data.utils import augment, zssr_collate_fn

class ZSSRRunner:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.L1Loss()
        self.learning_rate = 1e-3

        self.history = {
            'loss': [],
            'grad_mag': []
        }

    def run(self, dataset: AbstractSRDataset, out_size: torch.Size, n_epochs=10, n_scale_factors=6):
        """
        Trains the model on the internal patches of the test image.
        """
        test_img = dataset.strategy.base_img.unsqueeze(0).to(self.device)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)#, collate_fn=zssr_collate_fn)
        model = ZSSRConvNet().to(self.device) 
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        scheduler = LinearFitLossLR(optimizer)
        print(f"--- Training ZSSR for {n_epochs} epochs ---")

        scale_factors = np.linspace(1.0, dataset.scale_factor, n_scale_factors+1)[1:]

        for s_i in scale_factors:
            self._reset_lr(optimizer)
            dataset.curr_s_i = s_i
            model.train()
            print(f"--- Training ZSSR with s_i={s_i} ---")
            for epoch in range(n_epochs):
                epoch_loss = 0.0
                epoch_grad = 0.0
                for i, (lr_patch, hr_patch) in enumerate(dataloader):                
                    lr_patch, hr_true = lr_patch.to(self.device), hr_patch.to(self.device)
                    hr_spatial_dims = hr_true.shape[-2:]

                    lr_patch += self._compute_noise(lr_patch.shape)
                    
                    optimizer.zero_grad()
                    hr_pred = model(lr_patch, hr_spatial_dims)
                    loss = self.criterion(hr_pred, hr_true)
                    loss.backward()
                    optimizer.step()
                    scheduler.step(loss.item())

                    epoch_grad += self._compute_grad_mag(model)
                    epoch_loss += loss.item()
            
                self.history['grad_mag'].append(epoch_grad / len(dataloader))
                self.history['loss'].append(epoch_loss / len(dataloader))
                    
                print(f"End of epoch {epoch}, Loss: {loss.item():.6f}")

            model.eval()
            with torch.no_grad():
                dataset.add_image(self._generate_intermediate_hr(model, test_img, s_i))
        
        model.eval()
        return self._predict(model, test_img, out_size).squeeze(0)

    def _reset_lr(self, optimizer: optim.Optimizer):
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.learning_rate

    def _compute_noise(self, lr_size: torch.Size) -> torch.Tensor:
        noise_std = 5.0 / 255.0
        return torch.randn(lr_size, device=self.device) * noise_std

    def _compute_grad_mag(self, model: ZSSRConvNet) -> float:
        grad_mag = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                grad_mag += param_norm.item() ** 2
        return grad_mag ** 0.5

    def _generate_intermediate_hr(self, model: torch.nn.Module, test_img: torch.Tensor, s_i: float) -> torch.Tensor:
        h, w = test_img.shape[-2:]
        new_hr_size = (int(h * s_i), int(w * s_i))
        return model(test_img, new_hr_size).squeeze(0)

    def _predict(self, model: ZSSRConvNet, lr_img: torch.Tensor, out_size: torch.Size) -> torch.Tensor:
        augmented_lr_images = augment(lr_img) 
        outputs = []
        
        for idx, aug_img in enumerate(augmented_lr_images):            
            k = idx // 2
            is_flipped = (idx % 2 == 1)
            
            if k % 2 != 0:
                curr_out_size = (out_size[1], out_size[0])
            else:
                curr_out_size = out_size
                
            aug_out = model(aug_img, curr_out_size)
            
            if is_flipped:
                aug_out = transformsF.hflip(aug_out)
                
            aug_out_reversed = torch.rot90(aug_out, -k, dims=[-2, -1]) 
            
            outputs.append(aug_out_reversed)
            
        stacked_outputs = torch.stack(outputs)
        final_prediction, _ = torch.median(stacked_outputs, dim=0)
        
        return final_prediction

class LinearFitLossLR(lr_scheduler.LRScheduler):
    """
    Custom Learning Rate Scheduler that periodically fits a linear regression to the recent reconstruction errors (losses).
    If the standard deviation of the errors is greater than the slope by a certain factor, it divides the learning rate by 10.
    """
    def __init__(self, optimizer: optim.Adam, window_size=2048, slope_factor=20.0, min_lr=1e-6):
        super().__init__(optimizer)
        self.window_size = window_size
        self.slope_factor = slope_factor
        self.min_lr = min_lr
        self.losses = []

    def step(self, loss: float = None):
        if loss is None:
            return
        
        self.losses.append(loss)
        
        if len(self.losses) >= self.window_size:
            y = np.array(self.losses)
            x = np.arange(len(y))

            slope, _ = np.polyfit(x, y, deg=1)
            std_dev = np.std(y)

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