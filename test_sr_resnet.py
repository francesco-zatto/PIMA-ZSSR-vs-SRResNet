import sys
from pathlib import Path

import torch

from runner.zssr_runner import ZSSRRunner
sys.path.append(str(Path(__file__).parent))

import matplotlib.pyplot as plt

from data.datasets import Urban100Dataset, BSD100Dataset, COCODataset
from data.preprocessing import ResNetPreprocessing

from runner.sr_resnet_runner import SRResNetRunner

BATCH_SIZE = 16
TOTAL_ITERATIONS = 1e6

if __name__ == "__main__":
    
    # TRAINING

    # Initialize training dataset
    train_image_dir = "train2014"
    print(f"Preparing dataset from {train_image_dir}...")
    train_dataset = COCODataset(
        root_dir=train_image_dir, 
        scale_factor=4, 
        strategy=ResNetPreprocessing()
    )
    
    # instantiate the SRResNet pipeline
    print("Initializing SRResNet Runner...")
    runner = SRResNetRunner()

    # run the training process
    print("Starting training...")
    train_output = runner.train(dataset=train_dataset, total_iterations=TOTAL_ITERATIONS, batch_size=BATCH_SIZE)
    
    print("Training complete.")


    # TESTING 
    
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


