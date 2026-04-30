import sys
import argparse
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).parent))

import matplotlib.pyplot as plt

from data.datasets import COCODataset
from data.preprocessing import ResNetPreprocessing

from runner.runners import SRResNetRunner
from eval.pipeline import SRPipeline

from config import COCO_DIR, CHECKPOINT_DIR

BATCH_SIZE = 16
TOTAL_ITERATIONS = 1e6/2

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
    
    # New Architectural Flags (Default is False unless specified in terminal)
    parser.add_argument(
        "--use-batch-norm", 
        action='store_true',
        help="Enable Batch Normalization in the network"
    )
    parser.add_argument(
        "--scale-lr", 
        action='store_true',
        help="Scale LR images to [-1, 1] range"
    )
    parser.add_argument(
        "--final-activation", 
        action='store_true',
        help="Enable Tanh final activation bounding output to [-1, 1]"
    )
    
    args = parser.parse_args()
    
    # TRAINING

    # Initialize training dataset
    train_image_dir = str(COCO_DIR)
    print(f"Preparing dataset from {train_image_dir}...")
    
    # Pass the scale_lr argument to the strategy
    train_dataset = COCODataset(
        root_dir=train_image_dir, 
        scale_factor=4, 
        strategy=ResNetPreprocessing(train=True, scale_LR=args.scale_lr)
    )
    
    # instantiate the SRResNet pipeline
    print("Initializing SRResNet Runner...")
    
    # Pass the architectural toggles to runner 
    runner = SRResNetRunner(
        use_batch_norm=args.use_batch_norm,
        final_activation=args.final_activation,
        scale_lr=args.scale_lr
    )
    
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
