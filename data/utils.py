import torch
import torchvision.transforms.functional as transformsF
import torch.nn as nn
import torch.nn.functional as F

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

def estimate_michaeli_irani_kernel(
    image: torch.Tensor, 
    scale_factor: int = 2, 
    k_size: int = 11, 
    patch_size: int = 5, 
    sigma: float = None,      
    num_iters: int = 300,      
    lr: float = 0.05,          
    lambda_reg: float = 0.1,   
    max_queries: int = 1500
) -> torch.Tensor:
    """
    Estimates the optimal SR kernel directly from the low-res image 
    using a continuous gradient-descent adaptation of the MAP 
    objective from Michaeli & Irani (ICCV 2013).
    """
    device = image.device
    
    # Normalize image to [0, 1] if it's in [0, 255]
    if image.max() > 1.0:
        image = image / 255.0

    # Initialize the kernel k as a delta function 
    K = nn.Parameter(torch.zeros(1, 1, k_size, k_size, device=device))
    with torch.no_grad():
        K[0, 0, k_size // 2, k_size // 2] = 1.0
    
    optimizer = torch.optim.Adam([K], lr=lr)
    
    def extract_patches(img: torch.Tensor, p_size: int) -> torch.Tensor:
        patches = F.unfold(img, kernel_size=p_size)
        return patches.squeeze(0).t()
        
    # Extract fixed query patches (q_i) from the low-res image
    Q = extract_patches(image, patch_size)
    
    current_sigma = sigma

    for step in range(num_iters):
        optimizer.zero_grad()
        
        # Down-sample the image using the current estimate of k to generate candidate R_j
        pad = k_size // 2
        padded_img = F.pad(image, (pad, pad, pad, pad), mode='reflect')
        kernel_rgb = K.repeat(3, 1, 1, 1)
        L_down = F.conv2d(padded_img, kernel_rgb, stride=scale_factor, groups=3)
        
        # Candidate "parent" patches from the downsampled image (R_j * k)
        R_down = extract_patches(L_down, patch_size)
        
        # Randomly sample a subset of query patches to save memory
        if Q.size(0) > max_queries:
            idx = torch.randperm(Q.size(0), device=device)[:max_queries]
            Q_sub = Q[idx]
        else:
            Q_sub = Q
            
        # Compute distances securely using expanded formula (a^2 + b^2 - 2ab)
        q_sq = Q_sub.pow(2).sum(dim=1, keepdim=True)
        r_sq = R_down.pow(2).sum(dim=1)
        cross_term = torch.mm(Q_sub, R_down.t())
        
        # Clamp at 0 to avoid tiny negative values from floating point inaccuracies 
        dist = torch.clamp(q_sq + r_sq - 2 * cross_term, min=0.0) 
        
        # Dynamically estimate sigma to ensure gradients flow smoothly
        if step == 0 and current_sigma is None:
            with torch.no_grad():
                min_dist, _ = dist.min(dim=1)
                # Use a multiple of the median distance to set the softmax scale
                current_sigma = torch.sqrt(min_dist.median() + 1e-8).item() * 2.0
                current_sigma = max(current_sigma, 0.01)

        # Compute the Data Loss
        S = -dist / (2 * current_sigma ** 2)
        loss_data = -torch.logsumexp(S, dim=1).mean()
        
        # Compute the Regularization Loss
        grad_x = K[:, :, :, 1:] - K[:, :, :, :-1]
        grad_y = K[:, :, 1:, :] - K[:, :, :-1, :]
        loss_reg = (grad_x ** 2).mean() + (grad_y ** 2).mean()
        
        # Total loss and backprop
        loss = loss_data + lambda_reg * loss_reg
        loss.backward()
        optimizer.step()
        
        # Enforce kernel constraints
        with torch.no_grad():
            # Allow minor negative values (characteristic of optimal SR kernels) 
            K.data = torch.clamp(K.data, min=-0.2) 
            # Ensure it sums to 1 to maintain brightness
            K.data /= K.data.sum()
            
    return K.data.detach()