import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from vizconfig import models, URBAN100_DIR # Import your new config
from data.preprocessing import ResNetPreprocessing
from runner.runners import SRResNetRunner


def visualize_feature_maps(model, input_tensor, model_name, model_type, overlay=False):
    """
    Registers hooks, runs a forward pass, and plots activations.
    Now includes the Input LR image as the first row.
    """
    activations = {}

    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook

    target_layers = {
        "Initial Conv": model.initial_conv,
        "Res Block 8": model.residual_blocks[7],
        "Res Block 16": model.residual_blocks[15], 
        "Output": model.final_conv
    }

    hooks = [layer.register_forward_hook(get_activation(name)) for name, layer in target_layers.items()]

    model.eval()
    with torch.no_grad():
        _ = model(input_tensor)

    # Prepare input image for Row 0 and potential Overlay
    # Revert [-1, 1] normalization to [0, 1] for display if necessary
    input_img_display = input_tensor[0].permute(1, 2, 0).cpu().numpy()
    if input_img_display.min() < 0:
        input_img_display = (input_img_display + 1) / 2
    input_img_display = np.clip(input_img_display, 0, 1)

    # Total rows = Feature Map Layers + 1 for the Input Row
    num_feat_layers = len(activations)
    num_total_rows = num_feat_layers + 1
    
    fig, axes = plt.subplots(num_total_rows, 8, figsize=(24, num_total_rows * 3))

    # --- ROW 0: INPUT CHANNELS (RGB) ---
    input_channels = ["Input (R)", "Input (G)", "Input (B)"]
    for j in range(8):
        ax = axes[0, j]
        if j < 3:
            # Plot individual RGB channels
            im = ax.imshow(input_img_display[:, :, j], cmap='viridis')
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.05, format='%.2f')
            ax.set_title(input_channels[j], fontsize=10)
        else:
            ax.axis('off')
        
        if j == 0:
            ax.set_ylabel("Input (LR)", fontsize=12, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])

    # --- ROWS 1 to N: FEATURE MAPS ---
    for i, (name, act) in enumerate(activations.items()):
        row_idx = i + 1  # Offset by 1 for the input row
        act = act.squeeze(0) 
        
        for j in range(min(act.shape[0], 8)):
            ax = axes[row_idx, j]
            channel_data = act[j].cpu().numpy()

            if overlay:
                upsampled = F.interpolate(
                    act[j].unsqueeze(0).unsqueeze(0), 
                    size=(input_tensor.shape[2], input_tensor.shape[3]), 
                    mode='bilinear', 
                    align_corners=False
                ).squeeze().cpu().numpy()
                ax.imshow(input_img_display)
                im = ax.imshow(upsampled, cmap='jet', alpha=0.5)
            else:
                vmin = np.percentile(channel_data, 1)
                vmax = np.percentile(channel_data, 99)
                im = ax.imshow(channel_data, cmap='viridis', vmin=vmin, vmax=vmax if vmax > vmin else None)

            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.05, format='%.2f')
            cbar.ax.tick_params(labelsize=8)

            if j == 0:
                ax.set_ylabel(name, fontsize=12, fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])

        # Hide unused columns
        for j in range(min(act.shape[0], 8), 8):
            axes[row_idx, j].axis('off')

    for h in hooks:
        h.remove()

    plt.tight_layout(pad=2.0, w_pad=3.5, h_pad=1.0)
    plt.suptitle(f"{model_type} ({model_name}): Feature maps for 8 first channels", fontsize=18, y=0.98, fontweight='bold')
    
    save_dir = Path("../report/images/feature_maps/")
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_dir / f"{model_type}_{model_name}.png")
    plt.close()


# --- CONFIGURATION ---
for model_id, cfg in models.items():
    print(f"\nProcessing {model_id}...")

    # 1. Initialize Runner using flags from vizconfig
    runner = SRResNetRunner(
        use_batch_norm=cfg['use_batch_norm'], 
        final_activation=cfg['final_activation'],
        scale_lr=cfg['scale_lr']
    )

    # 2. Load Weights
    if cfg['checkpoint'].exists():
        checkpoint = torch.load(cfg['checkpoint'], map_location='cpu')
        runner.model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print(f"Skipping {model_id}: Checkpoint not found at {cfg['checkpoint']}")
        continue

    # 3. Setup Preprocessing & Sample
    preproc = ResNetPreprocessing(train=False, scale_LR=runner.scale_lr)
    preproc.prepare(str(URBAN100_DIR), "*.png")
    
    index_to_sample = 10
    lr_tensor, _ = preproc.sample(index_to_sample, scale_factor=4)
    lr_input = lr_tensor.unsqueeze(0).to(runner.device)

    # 4. Run Visualization
    visualize_feature_maps(
        runner.model, 
        lr_input, 
        model_name=cfg['short_name'], 
        model_type=cfg['model_type'], 
        overlay=False
    )