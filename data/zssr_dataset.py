import torch
import numpy as np
import torchvision.transforms.functional as transforms
from data.abstract_sr_dataset import AbstractSRDataset

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
        self.h, self.w = self.image.shape[1], self.image.shape[2]
        self.current_scale_factor = scale_factor

        self.hr_scales = np.linspace(1.0, 0.5, num_hr_scale_factors)
        self.pool_fathers: dict[float, torch.Tensor] = {}
        self._create_inital_pool()

    def _create_inital_pool(self):
        """Creates the initial pool of augmented HR images based on the input image and specified scales."""
        for scale in self.hr_scales:
            new_h, new_w = int(self.h * scale), int(self.w * scale)
            resized_image = transforms.resize(self.image, (new_h, new_w))
            self._add_to_pool(resized_image)
    
    def _add_to_pool(self, hr_image: torch.Tensor):
        """Adds a new HR image to the pool of augmented images."""
        augmented_images = self._augment(hr_image)
        ratio = self._compute_ratio(hr_image.shape[1:], (self.h, self.w))
        self.pool_fathers.setdefault(ratio, []).extend(augmented_images)

    def _augment(self, img: torch.Tensor) -> list[torch.Tensor]:
        """Applies rotations and horizontal flips to the input image to create 8 augmented versions."""
        rotations = [0, 90, 180, 270]
        ks = [angle // 90 for angle in rotations]
        augmented_images = []
        for k in ks:
            rotated_img = torch.rot90(img, k)
            hflip_img = transforms.hflip(rotated_img)
            augmented_images.extend([rotated_img, hflip_img])
        return augmented_images

    def __len__(self) -> int:
        return self.num_patches
    
    def set_scale_factor(self, scale_factor: float):
        """
        Updates the current scale factor for generating LR-HR pairs."""
        self.current_scale_factor = scale_factor

    # TODO maybe change the way ratio is computed
    def _compute_ratio(self, new_shape: tuple[int, int], I_shape: tuple[int, int]) -> float:
        new_h, new_w = new_shape
        I_h, I_w = I_shape
        return new_h * new_w / (I_h * I_w)
    
    def update_pool(self, new_hr_image: torch.Tensor):
        """
        Adds new HR image to the pool of augmented images for training.
        """
        self._add_to_pool(new_hr_image)    

    def set_scale_factor(self, scale_factor: float):
        self.current_scale_factor = scale_factor

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        hr_image = self.extract(self.augmented_images)

        return None, None
