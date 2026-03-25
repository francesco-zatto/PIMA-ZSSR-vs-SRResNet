import sys
import argparse
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).parent))

from data.datasets import Urban100Dataset, BSD100Dataset
from data.preprocessing import ResNetPreprocessing
from runner.runners import SRResNetRunner
from eval.pipeline import SRPipeline

from config import URBAN100_DIR, BSD100_DIR, URBAN100_ZIP, BSD100_ZIP, SRRESNET_OUTPUT_DIR

if __name__ == "__main__":
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Evaluate SRResNet model on benchmark datasets")
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        required=True,
        help="Path to checkpoint file (required for evaluation)"
    )
    parser.add_argument(
        "--dataset", 
        type=str, 
        choices=["urban100", "bsd100", "both"],
        default="both",
        help="Dataset to evaluate on (default: both)"
    )
    parser.add_argument(
        "--scale-factor", 
        type=int, 
        default=4,
        help="Super-resolution scale factor (default: 4)"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default=str(SRRESNET_OUTPUT_DIR),
        help=f"Output directory for results (default: {SRRESNET_OUTPUT_DIR})"
    )
    parser.add_argument(
        "--pipeline", 
        action="store_true",
        help="Run full pipeline (save predicted images + CSV). Default: metrics only"
    )
    args = parser.parse_args()
    
    # Verify checkpoint exists
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        sys.exit(1)
    
    print(f"Loading checkpoint from {checkpoint_path}...")
    
    # Initialize runner
    runner = SRResNetRunner()
    
    # Set output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Evaluate on selected datasets
    datasets_to_eval = []
    if args.dataset in ["urban100", "both"]:
        datasets_to_eval.append(("Urban100", Urban100Dataset, URBAN100_DIR, URBAN100_ZIP))
    if args.dataset in ["bsd100", "both"]:
        datasets_to_eval.append(("BSD100", BSD100Dataset, BSD100_DIR, BSD100_ZIP))
    
    results_summary = {}
    
    for dataset_name, dataset_class, data_dir, data_zip in datasets_to_eval:
        print(f"\n{'='*50}")
        print(f"Evaluating on {dataset_name}")
        print(f"{'='*50}")
        
        # Check if dataset directory exists
        if not data_dir.exists():
            print(f"Warning: {data_dir} not found.")
            if data_zip.exists():
                print(f"Extracting {data_zip}...")
                import zipfile
                with zipfile.ZipFile(data_zip, 'r') as zip_ref:
                    zip_ref.extractall(data_dir.parent)
            else:
                print(f"Error: Dataset not found at {data_dir} and no zip at {data_zip}")
                continue
        
        # Option 1: Simple evaluation (metrics only)
        test_dataset = dataset_class(
            root_dir=str(data_dir), 
            scale_factor=args.scale_factor, 
            strategy=ResNetPreprocessing(train=False)
        )
        
        results, _ = runner.evaluate(
            dataset=test_dataset, 
            checkpoint_path=checkpoint_path
        )
        
        results_summary[dataset_name] = results
        print(f"\n{dataset_name} Results -> PSNR: {results['psnr']:.2f} dB, SSIM: {results['ssim']:.4f}")
        
        # Option 2: Full pipeline (saves predicted images + CSV)
        if args.pipeline:
            print(f"\nRunning full pipeline for {dataset_name} (saving predictions)...")
            
            dataset_output = output_dir / dataset_name.lower()
            dataset_output.mkdir(parents=True, exist_ok=True)
            
            pipeline = SRPipeline(
                runner=runner,
                dataset_zip_path=str(data_zip),
                datasets_dir=str(data_dir.parent),
                output_dir=str(dataset_output),
                scale_factor=float(args.scale_factor)
            )
            pipeline.run()
    
    # Print summary
    print(f"\n{'='*50}")
    print("EVALUATION SUMMARY")
    print(f"{'='*50}")
    print(f"Checkpoint: {checkpoint_path.name}")
    print(f"Scale Factor: {args.scale_factor}x")
    print(f"{'='*50}")
    for dataset_name, results in results_summary.items():
        print(f"{dataset_name:12} -> PSNR: {results['psnr']:6.2f} dB, SSIM: {results['ssim']:.4f}")
    print(f"{'='*50}")  
    
    if args.pipeline:
        print(f"\nPredictions saved to: {output_dir}")
    
    print("\nEvaluation complete!")