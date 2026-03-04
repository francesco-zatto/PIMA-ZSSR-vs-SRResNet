import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from model.zssr_model import ZSSRConvNet
from data.datasets import AbstractSRDataset
from data.utils import zssr_collate_fn

class ZSSRRunner:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.L1Loss()
        self.learning_rate = 1e-3

    def run(self, dataset: AbstractSRDataset, out_size: torch.Size, n_epochs=1, batch_size=16) -> torch.Tensor:
        """
        Trains the model on the internal patches of the test image and returns the predicted HR test image.
        """
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)#, collate_fn=zssr_collate_fn)
        model = ZSSRConvNet().to(self.device) 
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        scheduler = LinearFitLossLR(optimizer)
        print(f"--- Training ZSSR for {n_epochs} epochs ---")

        model.train()
        for epoch in range(n_epochs):
            for lr_patch, hr_patch in dataloader:                
                lr_patch, hr_true = lr_patch.to(self.device), hr_patch.to(self.device)
                hr_spatial_dims = hr_true.shape[-2:]
                
                optimizer.zero_grad()
                hr_pred = model(lr_patch, hr_spatial_dims)
                loss = self.criterion(hr_pred, hr_true)
                loss.backward()
                optimizer.step()
                scheduler.step(loss.item())
                
            print(f"End of epoch {epoch}, Loss: {loss.item():.6f}")

        test_img = dataset.strategy.base_img.unsqueeze(0).to(self.device)
        interpol = nn.functional.interpolate(test_img, out_size, mode="bicubic")
        return model(test_img, out_size).squeeze(0), interpol.squeeze(0)

class LinearFitLossLR(lr_scheduler.LRScheduler):
    """
    Custom Learning Rate Scheduler that periodically fits a linear regression to the recent reconstruction errors (losses).
    If the standard deviation of the errors is greater than the slope by a certain factor, it divides the learning rate by 10.
    """
    def __init__(self, optimizer: optim.Adam, window_size=2048, slope_factor=10.0, min_lr=1e-6):
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