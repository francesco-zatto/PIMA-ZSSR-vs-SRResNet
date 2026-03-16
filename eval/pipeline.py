import os
import csv
import shutil
import tempfile
import zipfile
import torch
from pathlib import Path
from PIL import Image
import torchvision.transforms.functional as transformsF

from data.datasets import Urban100Dataset
from data.preprocessing import ZSSRPreprocessing, ResNetPreprocessing
from runner.runners import AbstractRunner

class SRPipeline:
    def __init__(self, runner: AbstractRunner, dataset_zip_path: str, datasets_dir: str, output_dir: str, scale_factor: float = 4.0):
        self.runner = runner
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
        
        return extract_path

    def process_image(self, lr_img_path: Path, hr_img_path: Path, csv_writer, **kwargs):
        """Prepares a single image environment and delegates to the Runner's evaluate method."""
        print(f"\n--- Evaluating: {lr_img_path.name} ---")
        
        is_zssr = "ZSSR" in self.runner.__class__.__name__
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            
            if is_zssr:
                results = self._process_zssr(temp_dir_path, lr_img_path, hr_img_path)
            else:                
                results = self._process_zssr(temp_dir_path, hr_img_path)

        # Log Metrics
        psnr_val = results['psnr'].item()
        ssim_val = results['ssim'].item()
        print(f"Metrics -> PSNR: {psnr_val:.2f} dB, SSIM: {ssim_val:.4f}")

        if csv_writer:
            csv_writer.writerow([lr_img_path.name, f"{psnr_val:.4f}", f"{ssim_val:.4f}"])

    def run(self, **kwargs):
        """Executes the pipeline focused solely on orchestrating evaluation loop over images."""
        
        # Extract Data and filter for images
        extracted_dir = self.extract_dataset()
        
        lr_image_paths = list(extracted_dir.rglob("*LR*.png"))
        if not lr_image_paths:
            lr_image_paths = list(extracted_dir.rglob("*x4*.png"))
        if not lr_image_paths:
            lr_image_paths = list(extracted_dir.rglob("**/LR/**/*.png"))
            
        if not lr_image_paths:
            print(f"Could not locate LR x4 images. Please check your dataset formatting.")
            return

        print(f"Found {len(lr_image_paths)} LR images. Starting Evaluation Loop...")
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = self.output_dir / f"{self.runner.__class__.__name__.lower()}_evaluation_results.csv"
        
        # Iterate, Evaluate, and Record
        with open(csv_path, mode='w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(["Image_Name", "PSNR", "SSIM"])
            
            for lr_path in lr_image_paths:
                # Attempt to pair LR with Ground Truth HR
                hr_name = lr_path.name.replace('LR', 'HR').replace('x4', '')
                hr_path_candidates = list(extracted_dir.rglob(hr_name))
                
                if not hr_path_candidates and lr_path.parent.name == "LR":
                    possible_hr = lr_path.parent.parent / "HR" / lr_path.name
                    if possible_hr.exists():
                        hr_path_candidates.append(possible_hr)
                        
                if not hr_path_candidates:
                    print(f"Warning: Could not find HR ground truth for {lr_path.name}. Skipping.")
                    continue
                    
                hr_path = hr_path_candidates[0]
                self.process_image(lr_path, hr_path, csv_writer, **kwargs)
                
        print(f"\nPipeline evaluation completed successfully! Results saved to {csv_path}")

    def _process_zssr(self, temp_dir_path: Path, lr_img_path: Path, hr_img_path: Path, **kwargs) -> dict:
        """Handles the specific zero-shot training and evaluation loop for ZSSR."""
        # ZSSR needs the LR image for its internal dynamic dataset
        target_img_path = temp_dir_path / lr_img_path.name
        shutil.copy(lr_img_path, target_img_path)
        
        strategy = ZSSRPreprocessing(num_patches=64) 
        dataset = Urban100Dataset(root_dir=str(temp_dir_path), scale_factor=self.scale_factor, strategy=strategy)
        
        _, h, w = strategy.base_img.shape
        out_size = (int(h * self.scale_factor), int(w * self.scale_factor))
        
        # ZSSR MUST train on the specific image every time before predicting
        print(f"Training ZSSR locally on {lr_img_path.name}...")
        self.runner.train(dataset, out_size=out_size, **kwargs)
            
        # Load Ground Truth Tensor for ZSSR's evaluation signature
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        hr_img = Image.open(hr_img_path).convert('RGB')
        hr_true = transformsF.to_tensor(hr_img).to(device).unsqueeze(0)
        
        # Align spatial dimensions (Dataset rounding fallback)
        min_h = min(self.runner.out_size[0], hr_true.shape[-2])
        min_w = min(self.runner.out_size[1], hr_true.shape[-1])
        hr_true = hr_true[..., :min_h, :min_w]
        
        # Call evaluate (ZSSR style)
        results, hr_pred = self.runner.evaluate(hr_true=hr_true, save_hr=True)
        
        # Save the ZSSR Prediction
        pred_tensor = hr_pred.squeeze(0).cpu().clamp(0, 1)
        pred_pil = transformsF.to_pil_image(pred_tensor)
        
        save_filename = f"{lr_img_path.stem}_ZSSR_pred.png"
        save_path = self.output_dir / save_filename
        pred_pil.save(save_path)
        print(f"Saved ZSSR prediction to: {save_path.name}")
        
        return results

    def _process_resnet(self, temp_dir_path: Path, hr_img_path: Path) -> dict:
        """Handles standard evaluation for pre-trained models like ResNet."""
        # ResNet preprocessing generates the LR internally from the HR image, so we pass HR
        target_img_path = temp_dir_path / hr_img_path.name
        shutil.copy(hr_img_path, target_img_path)
        
        strategy = ResNetPreprocessing(train=False)
        dataset = Urban100Dataset(root_dir=str(temp_dir_path), scale_factor=self.scale_factor, strategy=strategy)
        
        # SRResNet is already trained globally, so just evaluate
        results = self.runner.evaluate(dataset)
        
        return results