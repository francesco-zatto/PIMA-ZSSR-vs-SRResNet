# Self-Supervised “Zero-Shot” SR vs Supervised SRResNet

## Overview
This repository provides a PyTorch implementation comparing two distinct approaches to Super-Resolution (SR):
* **SRResNet**: This is a supervised deep learning model based on the architecture by Ledig et al. (2017).
* **ZSSR (Zero-Shot Super-Resolution)**: This is a self-supervised model that trains directly on the internal patches of the test image across multiple scale factors.

## Architecture Details

### SRResNet
* The network consists of a sequence of Residual Blocks featuring Convolutional layers and PReLU activations.
* Upsampling is handled using Sub-Pixel Convolution blocks with a `PixelShuffle` layer.
* The evaluation pipeline includes built-in metric collection for both PSNR and SSIM.

### ZSSR
* The architecture utilizes a lightweight, fully-convolutional network built with `ConvReLUBlock` layers.
* During training, random Gaussian noise is added to the low-resolution patches to improve robustness.
* The learning rate is dynamically controlled by a custom `LinearFitLossLR` scheduler, which adjusts the learning rate by fitting a linear regression to recent losses and decaying when the standard deviation of errors is greater than the slope by a given factor.
* Final image prediction incorporates geometric self-ensembling (data augmentation), aggregating the outputs of 8 rotated and flipped transformations by taking the median pixel value.

## Configuration and Datasets
* The global configuration file manages paths for datasets like `Urban100` and `BSD100`.
* When running the code, specific output directories (`zssr`, `srresnet`, and `checkpoints`) are automatically created if they do not already exist.

## Usage Example

The main entry point to run the super-resolution pipeline is `test.py`.

To start the pipeline, simply run:

```bash
python test.py
