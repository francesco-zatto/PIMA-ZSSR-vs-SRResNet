import torch
import matplotlib.pyplot as plt
import torchvision.transforms.functional as transformsF
from PIL import Image
import numpy as np
from pathlib import Path
from data.preprocessing import ResNetPreprocessing
from runner.runners import SRResNetRunner
from config import BSD100_DIR, URBAN100_DIR
from model.sr_resnet_model import SRResNet


def visualize_feature_maps(model, input_tensor):
    """
    Registers hooks to specific layers, runs a forward pass, and plots activations.
    """
    activations = {}

    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook

    # register hooks
    hooks = []

    # print model architecture to identify layer names and positions
    #print(model)

    target_layers = {
        "Initial Conv": model.initial_conv,
        "Res Block 8": model.residual_blocks[7],
        "Res Block 16": model.residual_blocks[15], 
        "Final Output": model.final_conv
    }

    for name, layer in target_layers.items():
        hooks.append(layer.register_forward_hook(get_activation(name)))

    # forward pass (input_tensor is [1, 3, H, W])
    model.eval()
    with torch.no_grad():
        _ = model(input_tensor)

    # plot the first 8 channels of each layer
    num_layers = len(activations)
    fig, axes = plt.subplots(num_layers, 8, figsize=(20, num_layers * 2.5))

    for i, (name, act) in enumerate(activations.items()):
        # normalize for visualization: [min, max] -> [0, 1]
        act = act.squeeze(0) # remove batch dim
        for j in range(min(act.shape[0], 8)):
            ax = axes[i, j]
            # plot the j-th channel
            channel_data = act[j].cpu().numpy()

            # robust scaling to remove outliers
            vmin = np.percentile(channel_data, 1)
            vmax = np.percentile(channel_data, 99)
            channel_img = act[j].cpu().numpy()
            if (vmax - vmin) < 1e-6:
                ax.imshow(channel_data, cmap='viridis')
            else:
                ax.imshow(channel_data, cmap='viridis', vmin=vmin, vmax=vmax)
            if j == 0:
                ax.set_title(name, loc='left', fontsize=12, fontweight='bold')

        # hide any remaining unused subplots in the current row (if there are fewer than 8 channels)
        for j in range(min(act.shape[0], 8), 8):
            ax = axes[i, j]
            ax.axis('off')

    # remove hooks to keep the model clean
    for h in hooks:
        h.remove()

    plt.tight_layout()
    plt.show()


use_batch_norm = False
scale_LR = True
final_activation = False
index_to_sample = 10

preproc = ResNetPreprocessing(train=False, scale_LR=scale_LR)
preproc.prepare(str(URBAN100_DIR), "*.png")
lr_tensor, hr_tensor = preproc.sample(index_to_sample, scale_factor=4)
lr_input = lr_tensor.unsqueeze(0).to('cpu')
model = SRResNet(use_batch_norm=use_batch_norm, final_activation=final_activation)

visualize_feature_maps(model, lr_input)