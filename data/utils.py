import torch
import torchvision.transforms.functional as transformsF

def augment(img: torch.Tensor) -> list[torch.Tensor]:
    """Applies rotations and horizontal flips to the input image to create 8 augmented versions."""
    rotations = [0, 90, 180, 270]
    ks = [angle // 90 for angle in rotations]
    augmented_images = []
    for k in ks:
        rotated_img = torch.rot90(img, k, dims=[-2, -1])
        hflip_img = transformsF.hflip(rotated_img)
        augmented_images.extend([rotated_img, hflip_img])
    return augmented_images

def zssr_collate_fn(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Collate function to stack LR and HR patches into batches.
    Crop the HR images to the minimum batch size before stacking into a tensor.
    """
    lr_patches, hr_patches = zip(*batch)
    min_h = min(hr.shape[1] for hr in hr_patches)
    min_w = min(hr.shape[2] for hr in hr_patches)

    scale = lr_patches[0].shape[1] / hr_patches[0].shape[1]
    lr_size = (int(min_h * scale), int(min_w * scale))
    lr_patches = torch.stack([transformsF.resize(lr, lr_size) for lr in lr_patches])
    hr_patches = torch.stack([transformsF.resize(hr, (min_h, min_w)) for hr in hr_patches])
    return lr_patches, hr_patches