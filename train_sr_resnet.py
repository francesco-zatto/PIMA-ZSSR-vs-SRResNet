import sys
import argparse
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).parent))

import matplotlib.pyplot as plt

from data.datasets import Urban100Dataset, BSD100Dataset, COCODataset
from data.preprocessing import ResNetPreprocessing

from runner.runners import SRResNetRunner
from eval.pipeline import SRPipeline

from config import COCO_DIR, CHECKPOINT_DIR, SRRESNET_OUTPUT_DIR

BATCH_SIZE = 16
TOTAL_ITERATIONS = 1e6

if __name__ == "__main__":
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train SRResNet model")
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        default=None,
        help="Path to checkpoint file to resume training (optional)"
    )
    parser.add_argument(
        "--checkpoint-dir", 
        type=str, 
        default=str(CHECKPOINT_DIR),
        help="Directory to save checkpoints (default: from config)"
    )
    args = parser.parse_args()
    
    # TRAINING

    # Initialize training dataset
    train_image_dir = str(COCO_DIR)
    print(f"Preparing dataset from {train_image_dir}...")
    train_dataset = COCODataset(
        root_dir=train_image_dir, 
        scale_factor=4, 
        strategy=ResNetPreprocessing()
    )
    
    # instantiate the SRResNet pipeline
    print("Initializing SRResNet Runner...")
    runner = SRResNetRunner()
    
    # Determine checkpoint load path
    checkpoint_load_path = None
    if args.checkpoint:
        checkpoint_load_path = Path(args.checkpoint)
        print(f"Resuming from checkpoint: {checkpoint_load_path}")
    else:
        print("Starting training from scratch (no checkpoint specified)")
    
    # run the training process
    print("Starting training...")
    checkpoint_save_path = Path(args.checkpoint_dir)
    train_output = runner.train(
        dataset=train_dataset, 
        total_iterations=TOTAL_ITERATIONS, 
        batch_size=BATCH_SIZE, 
        checkpoint_load_path=checkpoint_load_path, 
        checkpoint_save_path=checkpoint_save_path
    )
    
    print("Training complete.")

    # TESTING 

    # pipeline = SRPipeline(
    #     runner=runner,
    #     dataset_zip_path="datasets/Urban100.zip",
    #     datasets_dir="datasets",
    #     output_dir="outputs/srresnet_urban100",
    #     scale_factor=4.0)

    # # Assuming you loaded weights into resnet_runner or zssr_runner via checkpoint 
    # pipeline.run()
    
    # Initialize test dataset
    #test_image_dir = "test_images"
    #print(f"Preparing dataset from {test_image_dir}...")
    #test_dataset = Urban100Dataset(
    #    root_dir=test_image_dir, 
    #    scale_factor=4, 
    #    strategy=ResNetPreprocessing(train=False)
    #)

    #test_output = runner.evaluate(dataset=test_dataset)

    # TODO: convert model output from [-1, 1] to [0, 1]
    # TODO: visualize test output
    # TODO: calculate PSNR and SSIM for training and testing output

    #print("Testing complete.")


