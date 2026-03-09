import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import matplotlib.pyplot as plt

from data.datasets import Urban100Dataset
from data.preprocessing import ZSSRPreprocessing

from runner.zssr_runner import ZSSRRunner

import torch
import torchvision.transforms.functional as transformsF
from PIL import Image
import numpy as np

def print_cuda_info():
    # 1. Check if CUDA is actually available
    is_available = torch.cuda.is_available()
    print(f"CUDA Available: {is_available}")

    if is_available:
        # 2. Get the number of available GPUs
        device_count = torch.cuda.device_count()
        print(f"Number of GPUs: {device_count}")

        # 3. Iterate through all available devices and print their info
        for i in range(device_count):
            print(f"\n--- GPU {i} ---")
            print(f"Device Name: {torch.cuda.get_device_name(i)}")
            
            # 4. Get detailed properties (VRAM, compute capability, etc.)
            props = torch.cuda.get_device_properties(i)
            print(f"Compute Capability: {props.major}.{props.minor}")
            # Convert Total Memory from Bytes to Gigabytes
            print(f"Total VRAM: {props.total_memory / (1024**3):.2f} GB")
            print(f"Multi-Processor Count: {props.multi_processor_count}")

        # 5. Show which device PyTorch is currently targeting by default
        current_device = torch.cuda.current_device()
        print(f"\nCurrently Selected Device ID: {current_device}")
        
    else:
        print("PyTorch cannot find a CUDA-enabled GPU. It is defaulting to CPU.")

def save_training_plot(history: dict, filepath: str = "training_metrics.png"):
    """
    Saves the raw training loss and gradient magnitude history as an image file.
    Does not display the plot.
    """
    loss = history.get('loss', [])
    grad_mag = history.get('grad_mag', [])

    # Set up the figure side-by-side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # --- Subplot 1: Loss ---
    ax1.plot(loss, alpha=0.8, color='blue', label='Raw Loss')
    ax1.set_title("Training Loss")
    ax1.set_xlabel("Iteration (Batch)")
    ax1.set_ylabel("L1 Loss")
    ax1.set_yscale("log") # Log scale helps visualize the rapid initial drop
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)

    # --- Subplot 2: Gradient Magnitude ---
    ax2.plot(grad_mag, alpha=0.8, color='red', label='Raw Grad Mag')
    ax2.set_title("Global Gradient Magnitude")
    ax2.set_xlabel("Iteration (Batch)")
    ax2.set_ylabel("L2 Norm of Gradients")
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    
    # Save the figure to disk and close it to free memory
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Training plot successfully saved to: {filepath}")

NUM_PATCHES = 512
BATCH_SIZE = 1
SCALE_FACTOR = 4
N_EPOCHS = 10

if __name__ == "__main__":
    
    # print_cuda_info()
    
    # 1. Initialize the Dataset
    image_dir = "report/images/test_eye"
    gt = transformsF.to_tensor(Image.open("report/images/HR/HR_SISR_example.png").convert('RGB'))
    print(f"Preparing dataset from {image_dir}...")
    dataset = Urban100Dataset(
        root_dir=image_dir, 
        scale_factor=SCALE_FACTOR, 
        strategy=ZSSRPreprocessing(num_patches=NUM_PATCHES)
    )

    # 2. Instantiate the ZSSR pipeline
    print("Initializing ZSSR Runner...")
    runner = ZSSRRunner()

    # 3. Run the training process
    print("Starting training...")
    out_size = tuple(int(dim * SCALE_FACTOR) for dim in dataset.strategy.base_img.shape[-2:])
    print(out_size)
    sr_output, sr_output_interpol, model = runner.run(dataset, out_size, n_epochs=N_EPOCHS)

    sr_img = np.clip(sr_output.detach().cpu().permute(1, 2, 0).numpy(), 0.0, 1.0) 
    sr_img_interpol = np.clip(sr_output_interpol.detach().cpu().permute(1, 2, 0).numpy(), 0.0, 1.0)
    gt_np = gt.permute(1, 2, 0).numpy()
    # diff_with_gt = np.clip(sr_img - gt_np, 0.0, 1.0)
    # interpol_diff_with_gt = np.clip(sr_img_interpol - gt_np, 0.0, 1.0) 

    last_layer = model.network[-1]
    print(f"\nTotal sum of weights: {last_layer.weight.sum().item()}")
    print(f"Total sum of biases:  {last_layer.bias.sum().item()}")

    
    # 4. (Optional) Visualize the final output if your run() method returns the tensor
    if sr_output is not None:
        plt.imsave("out_zssr_interpol.png", sr_img_interpol)
        plt.imsave("out_zssr.png", sr_img)
        # plt.imsave("diff_model.png", diff_with_gt)
        # plt.imsave("diff_interpol.png", interpol_diff_with_gt)
        save_training_plot(runner.history, filepath="training_metrics_no_res_conn.png")
    else:
        print("Training complete.")