import torch
import numpy as np
from torch.utils.data import Dataset
from abc import ABC, abstractmethod
from PIL import Image

class AbstractSRDataLoader(Dataset, ABC):
    """
    Abstract base class for Super-Resolution Datasets.
    It defines the interface for both supervised and self-supervised datasets.
    """
    def __init__(self, scale_factor: int):
        super().__init__()
        self.scale_factor = scale_factor

    @abstractmethod
    def __len__(self) -> int:        
        """
        Returns the total number of samples in the dataset.
        """
        pass

    @abstractmethod
    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns a tuple: (lr_patch, hr_patch) for supervised datasets, both as PyTorch tensors.
        """
        pass

    def load_image(self, path: str) -> torch.Tensor:
        """
        Common utility method to load an image as RGB
        """
        return Image.open(path).convert('RGB')
    
    def to_tensor(self, img: Image.Image) -> torch.Tensor:
        """
        Converts a PIL Image to a PyTorch tensor and normalizes it to [0, 1].
        """
        return torch.from_numpy(np.array(img)).float() / 255.0
