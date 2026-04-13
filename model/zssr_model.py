import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvReLUBlock(nn.Module):
    """
    A basic convolutional block consisting of a convolution followed by a ReLU activation.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv(x))

    def _init_weights(self) -> None:
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(self.conv.bias)

class FinalConvBlock(nn.Module):
    """
    Final convolutional block consisting of a tanh activation, a convolution and a sigmoid activation.
    """
    def __init__(self):
        super().__init__()
        self.tanh = nn.Tanh()
        self.conv = nn.Conv2d(3, 3, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()
        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sigmoid(self.conv(self.tanh(x)))

    def _init_weights(self) -> None:
        nn.init.xavier_normal_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

class ZSSRConvNet(nn.Module):
    """
    A simple convolutional neural network for Zero-Shot Super-Resolution.
    """
    def __init__(self, num_channels: int = 64, num_blocks: int = 8, 
                 sigmoid_mode: str = 'after', 
                 upsample_mode: str = 'bicubic',
                 scale_factor: int = 2):
        super().__init__()
        
        # Validate ablation modes
        assert sigmoid_mode in ['none', 'after', 'inside'], "sigmoid_mode must be 'none', 'after', or 'inside'"
        assert upsample_mode in ['bicubic', 'learnt'], "upsample_mode must be 'bicubic' or 'learnt'"
        
        self.sigmoid_mode = sigmoid_mode
        self.upsample_mode = upsample_mode
        self.scale_factor = scale_factor
        
        # Main Residual Network 
        layers = [ConvReLUBlock(3, num_channels)]
        for _ in range(num_blocks - 2):
            layers.append(ConvReLUBlock(num_channels, num_channels))
            
        last_conv = nn.Conv2d(num_channels, 3, kernel_size=3, padding=1)
        nn.init.zeros_(last_conv.weight)
        nn.init.zeros_(last_conv.bias)
        layers.append(last_conv)

        self.network = nn.Sequential(*layers)
        
        # Final Output Block
        if self.sigmoid_mode in ['after', 'inside']:
            self.final_conv = FinalConvBlock()
        else:
            self.final_conv = nn.Identity()

    def forward(self, x: torch.Tensor, out_size: torch.Size) -> torch.Tensor:
        # Upsampling
        x_up = F.interpolate(x, size=out_size, mode="bicubic", align_corners=False)

        # Step 2: Residual Feature Extraction 
        res = self.network(x_up)

        # Final optional sigmoid 
        if self.sigmoid_mode == 'none':
            return res + x_up    
        elif self.sigmoid_mode == 'inside':
            res = self.final_conv(res)
            return res + x_up            
        elif self.sigmoid_mode == 'after':
            out = res + x_up
            return self.final_conv(out)