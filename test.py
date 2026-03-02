import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from data.datasets import Urban100Dataset
from data.preprocessing import ZSSRPreprocessing
from torch.utils.data import DataLoader

from data.utils import zssr_collate_fn

import matplotlib.pyplot as plt

NUM_PATCHES = 1024
BATCH_SIZE = 16
STEPS_PER_EPOCH = NUM_PATCHES // BATCH_SIZE


if __name__ == "__main__":

    # Example usage of ZSSRDataset
    image_path = "report/images/LR"
    dataset = Urban100Dataset(root_dir=image_path, scale_factor=4, strategy=ZSSRPreprocessingStrategy())
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, drop_last=True, collate_fn=zssr_collate_fn)

    for i in range(STEPS_PER_EPOCH):
        lr_batch, hr_batch = next(iter(dataloader))
        print(f"LR batch shape: {lr_batch.shape}, HR batch shape: {hr_batch.shape}")
        if i % 4 == 0:  
            lr_patch = lr_batch[0]
            hr_patch = hr_batch[0]

            # Visualize the first LR-HR pair
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            plt.title(f"Batch {i+1}")
            axes[0].imshow(lr_patch.permute(1, 2, 0).numpy())
            axes[0].set_title("LR Patch")
            axes[0].axis("off")
            axes[1].imshow(hr_patch.permute(1, 2, 0).numpy())
            axes[1].set_title("HR Patch")
            axes[1].axis("off")
            plt.show()