import torch
import torchvision.transforms.functional as transforms
from data import AbstractSRDataset

class ZSSRDataset(AbstractSRDataset):
    """
    Zero-Shot Super-Resolution Dataset.
    This dataset generates LR-HR pairs from the test input image.
    """
    def __init__(self, image_path: str, scale_factor: int, num_patches: int = 1000):
        super().__init__(scale_factor)
        self.image_path = image_path
        self.image = self.load_image(image_path)
        self.num_patches = num_patches

        self.augmented_images = self._augment(self.to_tensor(self.image))

    def _augment(self, img: torch.Tensor) -> list[torch.Tensor]:
        rotations = [0, 90, 180, 270]
        augmented_images = []
        for angle in rotations:
            k = angle // 90
            rotated_img = torch.rot90(img, k)
            hflip_img = transforms.hflip(rotated_img)
            augmented_images.extend([rotated_img, hflip_img])
        return augmented_images

    def __len__(self) -> int:
        return self.num_patches

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img = self.augmented_images[idx % len(self.augmented_images)]
        hr_patch = self._random_crop(img, self.scale_factor * 2)
        lr_patch = transforms.resize(hr_patch, [hr_patch.shape[1] // self.scale_factor, hr_patch.shape[2] // self.scale_factor])
        return lr_patch, hr_patch
