import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from IPython.display import display

def generate_comparison_plots(folder_paths, output_dir="outputs/plots"):
    """
    Reads CSVs and generates comparative plots for any number of ZSSR model variants.
    
    Expected input format:
    folder_paths = {
        'Dataset1': {'ModelA': 'path/to/A', 'ModelB': 'path/to/B', 'ModelC': 'path/to/C'},
        'Dataset2': {'ModelA': 'path/to/A', 'ModelB': 'path/to/B', 'ModelC': 'path/to/C'}
    }
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    if not folder_paths:
        print("No data provided.")
        return

    datasets = list(folder_paths.keys())
    # Dynamically extract all model names from the first dataset
    models = list(folder_paths[datasets[0]].keys())
    num_models = len(models)
    
    # Dynamic Data Containers
    metrics = {'PSNR': {}, 'SSIM': {}}
    for metric in metrics:
        metrics[metric]['means'] = {m: [] for m in models}
        metrics[metric]['data'] = {m: [] for m in models}
    
    # Load and Prepare Data
    for dataset_name, dirs in folder_paths.items():
        for model_name in models:
            try:
                model_dir = Path(dirs[model_name])
                csv_file = list(model_dir.glob("*.csv"))[0]
                df = pd.read_csv(csv_file)
                
                # Ensure numeric data
                df['PSNR'] = pd.to_numeric(df['PSNR'], errors='coerce')
                df['SSIM'] = pd.to_numeric(df['SSIM'], errors='coerce')
                
                # Store Means
                metrics['PSNR']['means'][model_name].append(df['PSNR'].mean())
                metrics['SSIM']['means'][model_name].append(df['SSIM'].mean())
                
                # Store Raw Data for boxplots (dropping NaNs)
                metrics['PSNR']['data'][model_name].append(df['PSNR'].dropna().values)
                metrics['SSIM']['data'][model_name].append(df['SSIM'].dropna().values)
                
            except Exception as e:
                print(f"Skipping {dataset_name} - {model_name} due to error: {e}")
                # Append NaNs/Empty arrays to maintain shape if a file is missing
                metrics['PSNR']['means'][model_name].append(np.nan)
                metrics['SSIM']['means'][model_name].append(np.nan)
                metrics['PSNR']['data'][model_name].append([])
                metrics['SSIM']['data'][model_name].append([])

    # Set up colors (using a matplotlib colormap to support N models dynamically)
    cmap = plt.get_cmap('tab10')
    colors = [cmap(i) for i in range(num_models)]

    # ==========================================
    # 1. Bar Plots (Averages)
    # ==========================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(datasets))  
    total_group_width = 0.8
    bar_width = total_group_width / num_models

    for i, model_name in enumerate(models):
        # Calculate offset so bars center nicely over the dataset tick
        offset = (i - num_models / 2 + 0.5) * bar_width
        
        axes[0].bar(x + offset, metrics['PSNR']['means'][model_name], 
                    bar_width, label=model_name, color=colors[i])
        axes[1].bar(x + offset, metrics['SSIM']['means'][model_name], 
                    bar_width, label=model_name, color=colors[i])

    for i, title in enumerate(['PSNR', 'SSIM']):
        axes[i].set_ylabel(f'Average {title}')
        axes[i].set_title(f'Average {title} Comparison')
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(datasets)
        axes[i].legend()
        axes[i].grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(Path(output_dir) / "comparison_barplot.png", dpi=300)
    plt.show()
    plt.close()

    # ==========================================
    # 2. Box Plots (Distributions)
    # ==========================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Calculate group spacing: num_models + 1 (for visual gap between datasets)
    group_width = num_models + 1
    base_positions = np.arange(len(datasets)) * group_width

    for i, model_name in enumerate(models):
        # Shift each model's boxplot by 'i' within its group
        pos = base_positions + i + 1 
        
        bplot_psnr = axes[0].boxplot(metrics['PSNR']['data'][model_name], positions=pos, 
                                     widths=0.6, patch_artist=True)
        bplot_ssim = axes[1].boxplot(metrics['SSIM']['data'][model_name], positions=pos, 
                                     widths=0.6, patch_artist=True)
        
        # Color the boxes
        for patch in bplot_psnr['boxes']: patch.set_facecolor(colors[i])
        for patch in bplot_ssim['boxes']: patch.set_facecolor(colors[i])

    # Formatting Boxplots
    center_offset = (num_models + 1) / 2
    tick_positions = base_positions + center_offset

    for i, title in enumerate(['PSNR', 'SSIM']):
        axes[i].set_title(f'{title} Distribution')
        axes[i].set_xticks(tick_positions)
        axes[i].set_xticklabels(datasets)
        axes[i].grid(axis='y', linestyle='--', alpha=0.7)
        
        # Custom legend for Boxplots
        handles = [plt.Rectangle((0,0),1,1, color=colors[idx]) for idx in range(num_models)]
        axes[i].legend(handles, models)

    plt.tight_layout()
    plt.savefig(Path(output_dir) / "comparison_boxplot.png", dpi=300)
    plt.show()
    plt.close()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
plt.close(fig) 

def plot_layer_distributions(epoch: int, layer_stats: dict[str, list[float]]):
    ax1.clear()
    ax2.clear()
    
    # Plot Weight Histogram
    weights = layer_stats['weights'][epoch]
    ax1.hist(weights, bins=50, color='royalblue', alpha=0.7)
    ax1.set_title(f'Weight Distribution (Epoch {epoch})')
    ax1.set_xlabel('Weight Value')
    ax1.set_ylabel('Frequency')
    ax1.grid(axis='y', alpha=0.3)
    
    # Plot Gradient Histogram
    grads = layer_stats['gradients'][epoch]
    ax2.hist(grads, bins=50, color='darkorange', alpha=0.7)
    ax2.set_title(f'Gradient Distribution (Epoch {epoch})')
    ax2.set_xlabel('Gradient Value')
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    display(fig)