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

class ResNetPreprocessing(SRPreprocessingStrategy):
    def __init__(self, crop_size=96, train=True):
        self.crop_size = crop_size
        self.train = train 
        self.img_paths = []

    def prepare(self, root_dir: str, ext: str):
        search_path = os.path.join(root_dir, "**", ext)
        self.img_paths = glob.glob(search_path, recursive=True)

    def sample(self, idx: int, scale_factor: float) -> tuple[torch.Tensor, torch.Tensor]:
        hr_img = Image.open(self.img_paths[idx]).convert('RGB')
        hr_tensor = transformsF.to_tensor(hr_img)

        if self.train:
            # use random crops
            i, j, h, w = transforms.RandomCrop.get_params(hr_tensor, (self.crop_size, self.crop_size))
            hr_target = transformsF.crop(hr_tensor, i, j, h, w)

        else:
            # use full image
            h, w = hr_tensor.shape[1], hr_tensor.shape[2]
            new_size = (h // int(scale_factor)) * int(scale_factor), (w // int(scale_factor)) * int(scale_factor)
            hr_target = transformsF.center_crop(hr_tensor, list(new_size))

         # Generate LR image
        lr_size = (hr_target.shape[1] // int(scale_factor), hr_target.shape[2] // int(scale_factor))
        lr_target = transformsF.resize(hr_target, list(lr_size), interpolation=transforms.InterpolationMode.BICUBIC)
        
        # Scale HR to [-1, 1] according to paper
        hr_target = (hr_target * 2.0) - 1.0

        return lr_target, hr_target
    
    def __len__(self):
        return len(self.img_paths)
       

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

    def prepare(self, root_dir: str, ext: str):
        image_path = glob.glob(os.path.join(root_dir, ext))[0]
        base_img = self._load_image(image_path)
        _, h, w = base_img.shape
        
        hr_scales = np.linspace(1.0, 0.5, self.num_hr_scales)
        for s in hr_scales:
            new_h, new_w = int(h * s), int(w * s)
            resized = transformsF.resize(base_img, (new_h, new_w), 
                                         interpolation=transforms.InterpolationMode.BICUBIC)
            self._add_to_pool(resized, (h, w))

    def _load_image(self, image_path: str) -> torch.Tensor:
        return transformsF.to_tensor(Image.open(image_path).convert('RGB'))

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
