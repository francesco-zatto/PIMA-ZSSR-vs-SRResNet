import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import matplotlib.pyplot as plt

from data.datasets import Urban100Dataset
from data.preprocessing import ZSSRPreprocessing

from runner.zssr_runner import ZSSRRunner

NUM_PATCHES = 1024
BATCH_SIZE = 16
N_EPOCHS = 10

if __name__ == "__main__":
    
    # 1. Initialize the Dataset
    image_dir = "report/images/LR"
    print(f"Preparing dataset from {image_dir}...")
    dataset = Urban100Dataset(
        root_dir=image_dir, 
        scale_factor=4, 
        strategy=ZSSRPreprocessing(num_patches=NUM_PATCHES)
    )

    # 2. Instantiate the ZSSR pipeline
    print("Initializing ZSSR Runner...")
    runner = ZSSRRunner()

    # 3. Run the training process
    print("Starting training...")
    sr_output = runner.run(dataset=dataset, n_epochs=N_EPOCHS, batch_size=BATCH_SIZE)
    
    # 4. (Optional) Visualize the final output if your run() method returns the tensor
    if sr_output is not None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.imshow(sr_output.detach().cpu().permute(1, 2, 0).numpy())
        ax.set_title("ZSSR Output (x4)")
        ax.axis("off")
        plt.show()
    else:
        print("Training complete.")