import torch
import torch.nn as nn

from .sr_resnet_model import SRResNet
from .zssr_model import ZSSRConvNet

class HybridSRNet(nn.Module):
    def __init__(self, srresnet_config, zssr_config, integration_mode='distillation'):
        """
        Args:
            srresnet_config (dict): Configuration parameters for SRResNet.
            zssr_config (dict): Configuration parameters for ZSSR.
            integration_mode (str): 'distillation' or 'fusion_head'.
        """
        super(HybridSRNet, self).__init__()
        self.integration_mode = integration_mode
        
        # Initialize models

        self.srresnet = SRResNet(**srresnet_config)
        for param in self.srresnet.parameters():
            param.requires_grad = False

        self.zssr = ZSSRConvNet(**zssr_config)
        
        # Optional fusion head
        if self.integration_mode == 'fusion_head':
            self.fusion_head = nn.Sequential(
                nn.Conv2d(in_channels=6, out_channels=3, kernel_size=1, padding=0),
                nn.ReLU(),
                nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1)
            )

    def forward(self, x):
        """
        x: Low-resolution input patch/image
        """
        # Forward pass through the frozen SRResNet
        with torch.no_grad():
            sr_resnet_out = self.srresnet(x)
            
        # Forward pass through ZSSR
        zssr_out = self.zssr(x)
        
        # Integration
        
        if self.integration_mode == 'distillation':
            return zssr_out, sr_resnet_out
            
        elif self.integration_mode == 'fusion_head':
            # combined shape is (B, 6, H, W)
            combined = torch.cat((sr_resnet_out, zssr_out), dim=1)
            final_out = self.fusion_head(combined)
            return final_out
            
        else:
            raise ValueError(f"Unknown integration_mode: {self.integration_mode}")

    def load_srresnet_weights(self, checkpoint_path, device):
        """Helper to load only the pre-trained SRResNet weights"""
        checkpoint = torch.load(checkpoint_path, map_location=device)
        # Handle if the state dict is nested under a 'model_state_dict' key
        state_dict = checkpoint.get('model_state_dict', checkpoint) 
        self.srresnet.load_state_dict(state_dict)
        print(f"Loaded SRResNet weights from {checkpoint_path}")