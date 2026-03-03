import torch
import torch.nn as nn

class ConvReLUBlock(torch.nn.Module):
    """
    A basic convolutional block consisting of a convolution followed by a ReLU activation.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super(ConvReLUBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv(x))

class ZSSRConvNet(nn.Module):
    """
    A simple convolutional neural network for Zero-Shot Super-Resolution.
    """
    def __init__(self, num_channels: int = 64, num_blocks: int = 8):
        super(ZSSRConvNet, self).__init__()
        layers = [ConvReLUBlock(3, num_channels)]
        for _ in range(num_blocks - 1):
            layers.append(ConvReLUBlock(num_channels, num_channels))
        layers.append(nn.Conv2d(num_channels, 3, kernel_size=3, padding=1))
        layers.append(nn.Sequential(
            nn.Tanh(), nn.Conv2d(3, 3, kernel_size=3, padding=1), nn.Sigmoid()
        ))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, out_size: torch.Size) -> torch.Tensor:
        x = nn.functional.interpolate(x, out_size, mode="bicubic")
        return self.network(x) + x