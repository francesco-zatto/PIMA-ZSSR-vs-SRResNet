import torch
import numpy as np
import random
import glob
import os
import torchvision.transforms as transforms
import torchvision.transforms.functional as transformsF
from abc import ABC, abstractmethod
from PIL import Image

from data.utils import augment

class SRPreprocessingStrategy(ABC):
    """
    Interface for SR data handling. 
    """
    @abstractmethod
    def prepare(self, root_dir: str, ext: str) -> None:
        """Initializes the strategy's internal state."""
        pass

    @abstractmethod
    def sample(self, idx: int, scale_factor: float) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns a single (LR, HR) pair."""
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Returns the virtual length of the dataset."""
        pass

    @abstractmethod
    def update(self, new_hrs: list[torch.Tensor]) -> None:
        """
        Updates the dataset by adding the list of new HRs images.
        """
        pass

class ResNetPreprocessing(SRPreprocessingStrategy):
    # TODO
    pass

class ZSSRPreprocessing(SRPreprocessingStrategy):
    """
    Self-Supervised Strategy: Manages internal patch recurrence pools.
    """
    def __init__(self, num_patches=1000, num_hr_scales=6, crop_size=128):
        self.num_patches = num_patches
        self.num_hr_scales = num_hr_scales
        self.crop_size = crop_size
        self.pool_fathers: dict[float, list[torch.Tensor]] = {}
        self.sampling_probs: dict[float, float] = {}
        self.base_img = None

    def prepare(self, root_dir: str, ext: str):
        image_path = glob.glob(os.path.join(root_dir, ext))[0]
        self.base_img = self._load_image(image_path)
        self._add_downsampled_versions(self.base_img)

    def _load_image(self, image_path: str) -> torch.Tensor:
        return transformsF.to_tensor(Image.open(image_path).convert('RGB'))

    def _add_downsampled_versions(self, img: torch.Tensor):
        hr_scales = np.linspace(1.0, 0.8, self.num_hr_scales)
        _, h, w = img.shape
        for s in hr_scales:
            new_h, new_w = int(h * s), int(w * s)
            resized = transformsF.resize(img, (new_h, new_w), 
                                         interpolation=transforms.InterpolationMode.BICUBIC)
            self._add_to_pool(resized, (h, w))

    def _add_to_pool(self, hr_image: torch.Tensor, original_dims: tuple[int, int]):
        augmented_images = augment(hr_image)
        
        ratio = (hr_image.shape[1] * hr_image.shape[2]) / (original_dims[0] * original_dims[1])
        self.pool_fathers.setdefault(ratio, []).extend(augmented_images)
        
        ratios = np.array(list(self.pool_fathers.keys()))
        probs = ratios / np.sum(ratios)
        self.sampling_probs = dict(zip(ratios, probs))

    def sample(self, idx: int, scale_factor: float) -> tuple[torch.Tensor, torch.Tensor]:
        ratios = list(self.pool_fathers.keys())
        probs = [self.sampling_probs[r] for r in ratios]
        extracted_ratio = np.random.choice(ratios, p=probs)
        
        hr_father = random.choice(self.pool_fathers[extracted_ratio])
        hr_crop = self._crop(hr_father)
        
        lr_size = (int(hr_crop.shape[1] / scale_factor), int(hr_crop.shape[2] / scale_factor))
        lr_crop = transformsF.resize(hr_crop, lr_size, interpolation=transforms.InterpolationMode.BICUBIC)
        return lr_crop, hr_crop

    def _crop(self, hr_image: torch.Tensor) -> torch.Tensor:
        _, h, w = hr_image.shape
        if h < self.crop_size or w < self.crop_size:
            return hr_image
        i, j, th, tw = transforms.RandomCrop.get_params(hr_image, (self.crop_size, self.crop_size))
        return transformsF.crop(hr_image, i, j, th, tw)

    def __len__(self):
        return self.num_patches

    def update(self, new_hrs: list[torch.Tensor]) -> None:
        for hr in new_hrs:
            self._add_downsampled_versions(hr)

