
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection, PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from data.datasets import AbstractSRDataset
from metrics.metrics import SRMetricSuite
from model.sr_resnet_model import SRResNet

class SRResNetRunner:
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