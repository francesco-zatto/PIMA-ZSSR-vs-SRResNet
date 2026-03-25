import torch
import torch.nn as nn

class ConvBlock(torch.nn.Module):
    """
    A basic convolutional block consisting of a convolution followed by a PReLU activation
    with optional batch normalization.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, normalize=True, activation=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2)
        self.batch_norm = nn.BatchNorm2d(out_channels) if normalize else None
        self.prelu = nn.PReLU() if activation else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        return self.prelu(x)
    
class ResidualBlock(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, normalize=True):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, out_channels, kernel_size, normalize)
        self.conv2 = ConvBlock(out_channels, out_channels, kernel_size, normalize, activation=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        return x + residual

class SubPixelConvBlock(torch.nn.Module):
    def __init__(self, in_channels: int, upscale_factor: int = 2):
        super().__init__()
        out_channels = in_channels * (upscale_factor ** 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=upscale_factor)
        self.prelu = nn.PReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        return self.prelu(x)

class SRResNet(nn.Module):
    """
    SRResNet architecture based on the paper by Ledig et al. (2017).
    """
    def __init__(self, num_channels: int = 64, num_blocks: int = 16, upscale_factor: int = 4):
        super().__init__()
        self.num_channels = num_channels
        self.num_blocks = num_blocks
        self.upscale_factor = upscale_factor
        self.initial_conv = ConvBlock(in_channels=3, out_channels=num_channels, kernel_size=9, normalize=False)
        self.residual_blocks = nn.Sequential(*[ResidualBlock(in_channels=num_channels, out_channels=num_channels) for _ in range(num_blocks)])
        self.secondary_conv = ConvBlock(in_channels=num_channels, out_channels=num_channels, kernel_size=3, normalize=True, activation=False)
        
        if upscale_factor == 4:
            self.upsample = nn.Sequential(
                SubPixelConvBlock(num_channels, 2),
                SubPixelConvBlock(num_channels, 2)
            )
        else:
            self.upsample = SubPixelConvBlock(num_channels, upscale_factor)
        
        self.final_conv = ConvBlock(in_channels=num_channels, out_channels=3, kernel_size=9, normalize=False, activation=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.initial_conv(x)
        residual = x
        x = self.residual_blocks(x)
        x = self.secondary_conv(x)
        x = x + residual # Global skip connection
        x = self.upsample(x)
        x = self.final_conv(x)
        return torch.tanh(x)
