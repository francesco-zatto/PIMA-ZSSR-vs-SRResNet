import torch
import numpy as np
import random
import glob
import os
import torchvision.transforms as transforms
import torchvision.transforms.functional as transformsF
import torch.nn.functional as F
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

    def update(self, new_hrs: list[torch.Tensor]) -> None:
        # No dynamic updates needed for ResNet strategy
        pass
       

class ZSSRPreprocessing(SRPreprocessingStrategy):
    """
    Self-Supervised Strategy: Manages internal patch recurrence pools.
    """
    def __init__(self, num_patches=1000, num_hr_scales=6, crop_size=128):
        self.num_patches = num_patches
        self.num_hr_scales = num_hr_scales
        self.crop_size = crop_size
        self.pool_fathers: list[torch.Tensor] = [] 
        self.father_weights: list[float] = []
        self.base_img: torch.Tensor = None 

        self.custom_kernel: torch.Tensor = None 

    def set_kernel(self, kernel: torch.Tensor) -> None:
        """
        Stores the estimated kernel. Expands it to 3 channels so it applies 
        identically to R, G, and B independently.
        """
        if kernel.shape[0] == 1:
            kernel = kernel.repeat(3, 1, 1, 1)

        self.custom_kernel = kernel.to('cpu').detach()

    def prepare(self, root_dir: str, ext: str):
        """
        Create initial dataset from test_img.
        """
        image_path = glob.glob(os.path.join(root_dir, ext))[0]
        self.base_img = self._load_image(image_path)
        self._add_downsampled_versions(self.base_img)

    def _load_image(self, image_path: str) -> torch.Tensor:
        """
        Read image as PIL.Image and then convert to tensor in [0, 1] range
        """
        return transformsF.to_tensor(Image.open(image_path).convert('RGB'))

    def _add_downsampled_versions(self, img: torch.Tensor):
        """
        Create self.num_hr_scales downsampled versions of img to increase the HRs pool.
        """
        hr_scales = np.linspace(1.0, 0.8, self.num_hr_scales)
        _, h, w = img.shape
        for s in hr_scales:
            new_h, new_w = int(h * s), int(w * s)
            resized = transformsF.resize(img, (new_h, new_w), 
                                         interpolation=transforms.InterpolationMode.BICUBIC)
            self._add_to_pool(resized)

    def _add_to_pool(self, hr_image: torch.Tensor):
        """
        Add an image to the pool of HR father images, computing its sampling probability.
        """
        augmented_images = augment(hr_image)
        
        hr_area = hr_image.shape[1] * hr_image.shape[2]
        base_area = self.base_img.shape[1] * self.base_img.shape[2]
        size_ratio = hr_area / base_area

        # Weight proportional to how close are to base_img size
        weight = size_ratio if size_ratio <= 1.0 else 1.0 / size_ratio
        
        self.pool_fathers.extend(augmented_images)
        self.father_weights.extend([weight] * len(augmented_images))
        
    def sample(self, idx: int, scale_factor: float) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample a HR father, extract a crop, simulate camera blur, and downsample correctly.
        """
        hr_father = random.choices(self.pool_fathers, weights=self.father_weights, k=1)[0]
        
        # Take a crop from the HR image
        hr_crop = self._crop(hr_father)
        
        if self.custom_kernel is not None:
            hr_batched = hr_crop.unsqueeze(0)
            k_size = self.custom_kernel.shape[-1]
            pad = k_size // 2
            
            with torch.no_grad():
                # Reflect padding prevents the dark zero-padding borders
                padded_hr = F.pad(hr_batched, (pad, pad, pad, pad), mode='reflect')
                
                # Apply convolution with the custom kernel to simulate camera blur
                blurred_hr = F.conv2d(
                    padded_hr, 
                    self.custom_kernel, 
                    padding=0, 
                    groups=3
                )
                
            hr_to_downscale = blurred_hr.squeeze(0)
            
            # Clamp out-of-bounds values caused by negative kernel weights
            hr_to_downscale = torch.clamp(hr_to_downscale, 0.0, 1.0)
        else:
            hr_to_downscale = hr_crop

        # Correct mathematical downsampling (Decimation)
        if int(scale_factor) == scale_factor:
            s = int(scale_factor)
            lr_crop = hr_to_downscale[:, ::s, ::s]
        else:
            # If fractional
            lr_size = (int(hr_crop.shape[1] / scale_factor), int(hr_crop.shape[2] / scale_factor))
            lr_crop = transformsF.resize(
                hr_to_downscale, 
                lr_size, 
                interpolation=transforms.InterpolationMode.NEAREST,
                antialias=False
            )        
            
        return torch.clamp(lr_crop, 0.0, 1.0), torch.clamp(hr_crop, 0.0, 1.0)

    def _crop(self, hr_image: torch.Tensor, min_variance: float = 0.005, max_tries: int = 10) -> torch.Tensor:
        """
        Take a random crop of HR image if HR is large enough, 
        ensuring the crop has enough variance to contain meaningful texture.
        """
        _, h, w = hr_image.shape
        
        # Fallback if the image is smaller than the crop size
        if h < self.crop_size or w < self.crop_size:
            return hr_image

        best_crop = None
        highest_variance = -1.0

        for _ in range(max_tries):
            # Sample a random crop
            i, j, th, tw = transforms.RandomCrop.get_params(hr_image, (self.crop_size, self.crop_size))
            crop = transformsF.crop(hr_image, i, j, th, tw)
            
            # Calculate the variance of the crop
            crop_variance = torch.var(crop).item()

            # If it meets the threshold, return immediately
            if crop_variance >= min_variance:
                return crop

            #  Otherwise, keep track of the best one we've seen so far
            if crop_variance > highest_variance:
                highest_variance = crop_variance
                best_crop = crop

        return best_crop

    def __len__(self):
        return self.num_patches

    def update(self, new_hrs: list[torch.Tensor]) -> None:
        for hr in new_hrs:
            self._add_downsampled_versions(hr)

