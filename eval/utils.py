import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from IPython.display import display

import config

def generate_comparison_plots(folder_paths, output_dir="outputs/plots"):
    """
    Reads CSVs and generates comparative plots for normal vs variant models of zssr  
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    datasets = []
    
    # Data containers for Bar Plots
    psnr_means_norm, psnr_means_var = [], []
    ssim_means_norm, ssim_means_var = [], []
    
    # Data containers for Box Plots 
    psnr_data_norm, psnr_data_var = [], []
    ssim_data_norm, ssim_data_var = [], []
    
    # Data containers for Scatter Plots 
    scatter_psnr = {}
    scatter_ssim = {}
    
    # Load and Prepare Data
    for dataset_name, dirs in folder_paths.items():
        try:
            normal_dir = Path(dirs['zssr'])
            variant_dir = Path(dirs['sigmoid'])
            
            normal_csv = list(normal_dir.glob("*.csv"))[0]
            variant_csv = list(variant_dir.glob("*.csv"))[0]
            
            df_norm = pd.read_csv(normal_csv)
            df_var = pd.read_csv(variant_csv)
            
            # Ensure numeric data
            df_norm['PSNR'] = pd.to_numeric(df_norm['PSNR'], errors='coerce')
            df_norm['SSIM'] = pd.to_numeric(df_norm['SSIM'], errors='coerce')
            df_var['PSNR'] = pd.to_numeric(df_var['PSNR'], errors='coerce')
            df_var['SSIM'] = pd.to_numeric(df_var['SSIM'], errors='coerce')
            
            datasets.append(dataset_name)
            
            # Store Means
            psnr_means_norm.append(df_norm['PSNR'].mean())
            psnr_means_var.append(df_var['PSNR'].mean())
            ssim_means_norm.append(df_norm['SSIM'].mean())
            ssim_means_var.append(df_var['SSIM'].mean())
            
            # Store Raw Data for boxplots (dropping NaNs)
            psnr_data_norm.append(df_norm['PSNR'].dropna().values)
            psnr_data_var.append(df_var['PSNR'].dropna().values)
            ssim_data_norm.append(df_norm['SSIM'].dropna().values)
            ssim_data_var.append(df_var['SSIM'].dropna().values)
            
            # Merge for 1-to-1 scatter plot
            merged = pd.merge(df_norm, df_var, on='Image_Name', suffixes=('_norm', '_var'))
            scatter_psnr[dataset_name] = (merged['PSNR_norm'], merged['PSNR_var'])
            scatter_ssim[dataset_name] = (merged['SSIM_norm'], merged['SSIM_var'])
            
        except Exception as e:
            print(f"Skipping {dataset_name} due to error: {e}")

    if not datasets:
        print("No data found to plot.")
        return

    # Bar Plots (Averages)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(datasets))  
    width = 0.35                  

    # PSNR Bar Plot
    axes[0].bar(x - width/2, psnr_means_norm, width, label='ZSSR (Normal)', color='steelblue')
    axes[0].bar(x + width/2, psnr_means_var, width, label='ZSSR (Sigmoid)', color='darkorange')
    axes[0].set_ylabel('Average PSNR')
    axes[0].set_title('Average PSNR Comparison')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(datasets)
    axes[0].legend()
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)

    # SSIM Bar Plot
    axes[1].bar(x - width/2, ssim_means_norm, width, label='ZSSR (Normal)', color='steelblue')
    axes[1].bar(x + width/2, ssim_means_var, width, label='ZSSR (Sigmoid)', color='darkorange')
    axes[1].set_ylabel('Average SSIM')
    axes[1].set_title('Average SSIM Comparison')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(datasets)
    axes[1].legend()
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    bar_path = Path(output_dir) / "comparison_barplot.png"
    plt.savefig(bar_path, dpi=300)
    plt.show()
    plt.close()

    # Box Plots (Distributions)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Calculate offset positions for grouped boxplots
    pos_norm = np.array(range(1, len(datasets) * 3, 3))
    pos_var = pos_norm + 1

    box_colors = ['lightblue', 'lightsalmon']

    # PSNR Box Plot
    bplot1_norm = axes[0].boxplot(psnr_data_norm, positions=pos_norm, widths=0.6, patch_artist=True)
    bplot1_var = axes[0].boxplot(psnr_data_var, positions=pos_var, widths=0.6, patch_artist=True)
    
    for patch in bplot1_norm['boxes']: patch.set_facecolor(box_colors[0])
    for patch in bplot1_var['boxes']: patch.set_facecolor(box_colors[1])
    
    axes[0].set_title('PSNR Distribution')
    axes[0].set_xticks(pos_norm + 0.5)
    axes[0].set_xticklabels(datasets)
    axes[0].legend([bplot1_norm["boxes"][0], bplot1_var["boxes"][0]], ['ZSSR (Normal)', 'ZSSR (Sigmoid)'])
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)

    # SSIM Box Plot
    bplot2_norm = axes[1].boxplot(ssim_data_norm, positions=pos_norm, widths=0.6, patch_artist=True)
    bplot2_var = axes[1].boxplot(ssim_data_var, positions=pos_var, widths=0.6, patch_artist=True)

    plt.tight_layout()
    bar_path = Path(output_dir) / "comparison_boxplot.png"
    plt.savefig(bar_path, dpi=300)
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