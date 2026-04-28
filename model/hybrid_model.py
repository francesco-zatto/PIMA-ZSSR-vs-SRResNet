import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

from .sr_resnet_model import SRResNet
from .zssr_model import ZSSRConvNet

class HybridSRNet(nn.Module):
    def __init__(self, srresnet_config, zssr_config, integration_mode='cascade'):
        """
        Args:
            srresnet_config (dict): Configuration parameters for SRResNet.
            zssr_config (dict): Configuration parameters for ZSSR.
            integration_mode (str): 'distillation', 'fusion_head', or 'cascade'.
        """
        super(HybridSRNet, self).__init__()
        self.integration_mode = integration_mode
        
        # Initialize models
        self.srresnet = SRResNet(**srresnet_config)
        for param in self.srresnet.parameters():
            param.requires_grad = False

        self.zssr = ZSSRConvNet(**zssr_config)
        
        # Route the forward method and initialize mode-specific components
        if self.integration_mode == 'cascade':
            self._forward_strategy = self._forward_cascade
            
        elif self.integration_mode == 'distillation':
            self._forward_strategy = self._forward_distillation
            
        elif self.integration_mode == 'fusion_head':
            self.blend_weight = nn.Parameter(torch.tensor([0.5])) 
            self._forward_strategy = self._forward_fusion_head
            
        else:
            raise ValueError(f"Unknown integration_mode: {self.integration_mode}")

    def _run_srresnet(self, x):
        """Helper to run the frozen SRResNet with proper scaling."""
        with torch.no_grad():
            sr_resnet_out = self.srresnet(x)
        return torch.clamp(sr_resnet_out, 0.0, 1.0)

    def _align_srresnet(self, sr_resnet_out, target_tensor):
        """Helper to align SRResNet spatial dimensions to ZSSR's target."""
        target_h, target_w = target_tensor.shape[-2:]
        if sr_resnet_out.shape[-2:] != (target_h, target_w):
            return F.interpolate(
                sr_resnet_out, 
                size=(target_h, target_w), 
                mode='bicubic', 
                align_corners=False
            )
        return sr_resnet_out

    def _forward_cascade(self, x, out_size):
        sr_resnet_out = self._run_srresnet(x)
        out = self.zssr(sr_resnet_out, out_size)
        return out

    def _forward_distillation(self, x, out_size):
        sr_resnet_out = self._run_srresnet(x)
        zssr_out = self.zssr(x, out_size) if out_size else self.zssr(x)
        
        sr_resnet_out = self._align_srresnet(sr_resnet_out, zssr_out)
        return zssr_out, sr_resnet_out

    def _forward_fusion_head(self, x, out_size):
        sr_resnet_out = self._run_srresnet(x)
        zssr_out = self.zssr(x, out_size) if out_size else self.zssr(x)
        
        sr_resnet_out = self._align_srresnet(sr_resnet_out, zssr_out)
        
        alpha = torch.sigmoid(self.blend_weight) 
        final_out = (alpha * sr_resnet_out) + ((1.0 - alpha) * zssr_out)
        return final_out

    def forward(self, x, out_size):
        """
        x: Low-resolution input patch/image in range [0, 1]
        out_size: Tuple (H, W) indicating the spatial dimensions ZSSR is targeting
        """
        # Dynamically execute the chosen strategy
        return self._forward_strategy(x, out_size)
    
    def load_srresnet_weights(self, checkpoint_path, device):
        """Helper to load only the pre-trained SRResNet weights"""
        checkpoint = torch.load(checkpoint_path, map_location=device)
        # Handle if the state dict is nested under a 'model_state_dict' key
        state_dict = checkpoint.get('model_state_dict', checkpoint) 
        self.srresnet.load_state_dict(state_dict)
        print(f"Loaded SRResNet weights from {checkpoint_path}")