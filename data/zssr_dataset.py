import torch
import numpy as np
import random
import torchvision.transforms as transforms
import torchvision.transforms.functional as transformsF
from data.abstract_sr_dataset import AbstractSRDataset
from data.utils import augment

class ZSSRDataset(AbstractSRDataset):
    """
    Zero-Shot Super-Resolution Dataset.
    This dataset generates LR-HR pairs from the test input image.
    """
    def __init__(self, image_path: str, scale_factor: float = 4.0, num_patches: int = 1000,
                 num_hr_scale_factors: int = 6, crop_size: int = 128):
        super().__init__(scale_factor)
        self.image_path = image_path
        self.image = self.to_tensor(self.load_image(image_path))
        self.num_patches = num_patches
        self.crop_size = crop_size
        self.random_cropper = transforms.RandomCrop(crop_size)
        _, self.h, self.w = self.image.shape
        self.current_scale_factor = scale_factor

        self.hr_scales = np.linspace(1.0, 0.5, num_hr_scale_factors)
        self.pool_fathers: dict[float, torch.Tensor] = {}
        self._create_inital_pool()
        self.sampling_probs = self._compute_sampling_probs()

    def _create_inital_pool(self):
        """Creates the initial pool of augmented HR images based on the input image and specified scales."""
        for scale in self.hr_scales:
            new_h, new_w = int(self.h * scale), int(self.w * scale)
            resized_image = transformsF.resize(self.image, (new_h, new_w))
            self._add_to_pool(resized_image)
    
    def _add_to_pool(self, hr_image: torch.Tensor):
        """Adds a new HR image to the pool of augmented images."""
        augmented_images = augment(hr_image)
        ratio = self._compute_ratio(hr_image.shape[1:], (self.h, self.w))
        self.pool_fathers.setdefault(ratio, []).extend(augmented_images)
        self.sampling_probs = self._compute_sampling_probs()
    
    def _compute_sampling_probs(self) -> dict[float, float]:
        """
        Compute sampling probabilities for the ratio, proportional to the ratio itself.
        """
        ratios = np.array(list(self.pool_fathers.keys()))
        probs = ratios / np.sum(ratios)
        return dict(zip(ratios, probs))

    def __len__(self) -> int:
        return self.num_patches
    
    def set_scale_factor(self, scale_factor: float):
        """
        Updates the current scale factor for generating LR-HR pairs."""
        self.current_scale_factor = scale_factor

    def _compute_ratio(self, new_shape: tuple[int, int], I_shape: tuple[int, int]) -> float:
        new_h, new_w = new_shape
        I_h, I_w = I_shape
        return new_h * new_w / (I_h * I_w)
    
    def update_pool(self, new_hr_image: torch.Tensor):
        """
        Adds new HR image to the pool of augmented images for training.
        """
        self._add_to_pool(new_hr_image) 

    def _crop(self, hr_image: torch.Tensor) -> torch.Tensor:
        """Crops a random patch from the HR image."""
        _, h, w = hr_image.shape
        if h < self.crop_size or w < self.crop_size:
            return hr_image
        return self.random_cropper(hr_image)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        extracted_ratio = np.random.choice(list(self.pool_fathers.keys()), p=list(self.sampling_probs.values()))
        hr_images = self.pool_fathers[extracted_ratio]
        hr_image = random.choice(hr_images)
        hr_crop = self._crop(hr_image)
        lr_size = (int(hr_crop.shape[1] / self.current_scale_factor), int(hr_crop.shape[2] / self.current_scale_factor))
        lr_crop = transformsF.resize(hr_crop, lr_size, interpolation=transforms.InterpolationMode.BICUBIC)
        return lr_crop, hr_crop

