import os
import shutil
import tempfile
import zipfile
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from data.datasets import Urban100Dataset
from data.preprocessing import ZSSRPreprocessing
from runner.zssr_runner import ZSSRRunner

class ZSSRPipeline:
    def __init__(self, dataset_zip_path: str, datasets_dir: str, output_dir: str, scale_factor: float = 4.0):
        self.dataset_zip_path = Path(dataset_zip_path)
        self.datasets_dir = Path(datasets_dir)
        self.output_dir = Path(output_dir)
        self.scale_factor = scale_factor

    def extract_dataset(self) -> Path:
        """Unzips the dataset into the datasets folder if not already extracted."""
        self.datasets_dir.mkdir(parents=True, exist_ok=True)
        extract_path = self.datasets_dir / self.dataset_zip_path.stem
        
        if not extract_path.exists():
            print(f"Extracting {self.dataset_zip_path.name} to {extract_path}...")
            with zipfile.ZipFile(self.dataset_zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
        else:
            print(f"Dataset already extracted at: {extract_path}")
        
        return extract_path

    def process_image(self, img_path: Path):
        """Trains ZSSR on a single image using a temporary directory."""
        print(f"\n--- Processing: {img_path.name} ---")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            target_img_path = temp_dir_path / img_path.name
            shutil.copy(img_path, target_img_path)
            
            strategy = ZSSRPreprocessing(num_patches=64)
            dataset = Urban100Dataset(
                root_dir=str(temp_dir_path), 
                scale_factor=self.scale_factor, 
                strategy=strategy
            )
        
            _, h, w = strategy.base_img.shape
            out_size = (int(h * self.scale_factor), int(w * self.scale_factor))
            
            runner = ZSSRRunner()
            sr_output = runner.run(dataset, out_size, n_epochs=10)
            print(sr_output.shape)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        sr_img_np = np.clip(sr_output.detach().cpu().permute(1, 2, 0).numpy(), 0.0, 1.0)
        save_path = self.output_dir / f"{img_path.stem}_zssr_x{int(self.scale_factor)}.png"
        
        plt.imsave(str(save_path), sr_img_np)
        print(f"Saved prediction to {save_path}")

    def run(self):
        """Executes the full pipeline."""
        extracted_dir = self.extract_dataset()
        
        image_paths = list(extracted_dir.rglob("*.png"))
        
        if not image_paths:
            print(f"No images found in {extracted_dir}!")
            return

        print(f"Found {len(image_paths)} images. Starting ZSSR inference loop...")
        for img_path in image_paths:
            self.process_image(img_path)
            
        print("\nPipeline completed successfully! All predictions saved.")