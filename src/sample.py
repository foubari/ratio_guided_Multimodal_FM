"""
Generate pairs (x, y) via guided sampling.

Usage:
    # Baseline (independent flow)
    python src/sample.py --transform_type rotate90 --guidance_method none --num_samples 64

    # MC-guided (Feng et al.)
    python src/sample.py --transform_type rotate90 --guidance_method mc_feng --guidance_strength 0.5 --num_samples 64
"""
import os
# Fix OpenMP conflict on Windows (numpy/torch/matplotlib)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import sys

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.flow_matching import FlowMatchingModel
from src.models.unet import FlowMatchingUNet
from src.models.ratio_estimator import RatioEstimator
from src.data.mnist_dataset import get_flow_dataloader
from src.utils.flow_utils import sample_bimodal_guided
from src.utils.path_utils import get_checkpoint_path
from src.utils import set_seed


def visualize_pairs(samples_x, samples_y, save_path, transform_type='rotate90', num_cols=8, num_rows=8):
    """
    Display generated pairs as columns: each column shows (x, y) stacked vertically.
    
    Layout: 
        Col1    Col2    Col3 ...
        [x1]    [x2]    [x3]
        [y1]    [y2]    [y3]
        [x5]    [x6]    [x7]
        [y5]    [y6]    [y7]
        ...
    
    Args:
        samples_x, samples_y: Generated samples [N, 1, 28, 28]
        save_path: Path to save the figure
        transform_type: Name of the transform for title
        num_cols: Number of columns (pairs per row)
        num_rows: Number of pair-rows (each pair-row = 2 image rows)
    """
    num_pairs = min(num_cols * num_rows, len(samples_x))
    
    # Figure size: each image is ~1 inch, pairs are stacked (2 rows per pair-row)
    fig_width = num_cols * 1.2
    fig_height = num_rows * 2 * 1.2 + 0.5  # 2 images per pair-row
    
    fig, axes = plt.subplots(num_rows * 2, num_cols, figsize=(fig_width, fig_height))
    
    # Handle case where axes is 1D
    if num_rows * 2 == 1:
        axes = axes.reshape(1, -1)
    if num_cols == 1:
        axes = axes.reshape(-1, 1)
    
    for idx in range(num_pairs):
        col = idx % num_cols
        pair_row = idx // num_cols
        
        # Row indices: x goes in even rows, y goes in odd rows
        row_x = pair_row * 2
        row_y = pair_row * 2 + 1
        
        # Normalize to [0, 1]
        img_x = (samples_x[idx, 0].cpu().numpy() + 1) / 2
        img_y = (samples_y[idx, 0].cpu().numpy() + 1) / 2
        
        # Plot x
        axes[row_x, col].imshow(img_x, cmap='gray', vmin=0, vmax=1)
        axes[row_x, col].axis('off')
        
        # Plot y
        axes[row_y, col].imshow(img_y, cmap='gray', vmin=0, vmax=1)
        axes[row_y, col].axis('off')
    
    # Turn off remaining axes
    for idx in range(num_pairs, num_cols * num_rows):
        col = idx % num_cols
        pair_row = idx // num_cols
        row_x = pair_row * 2
        row_y = pair_row * 2 + 1
        axes[row_x, col].axis('off')
        axes[row_y, col].axis('off')
    
    # Add row labels on the left
    for pair_row in range(num_rows):
        row_x = pair_row * 2
        row_y = pair_row * 2 + 1
        axes[row_x, 0].set_ylabel('x', fontsize=10, rotation=0, labelpad=15, va='center')
        axes[row_y, 0].set_ylabel('y', fontsize=10, rotation=0, labelpad=15, va='center')
        axes[row_x, 0].yaxis.set_visible(True)
        axes[row_y, 0].yaxis.set_visible(True)
    
    fig.suptitle(f'Generated Pairs (x, y) — Transform: {transform_type}', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Sample bimodal pairs')
    parser.add_argument('--transform_type', type=str, default='rotate90',
                        help='Transformation type')
    parser.add_argument('--guidance_method', type=str, default='none',
                        choices=['none', 'mc_feng'],
                        help='Guidance method (none=independent, mc_feng=Feng MC guidance)')
    parser.add_argument('--guidance_strength', type=float, default=0.5,
                        help='Guidance strength (0=independent, 1=full guidance)')
    parser.add_argument('--mc_batch_size', type=int, default=128,
                        help='Number of MC samples for guidance')
    parser.add_argument('--loss_type', type=str, default='disc',
                        help='Loss type for ratio estimator')
    parser.add_argument('--num_samples', type=int, default=64,
                        help='Number of samples to generate')
    parser.add_argument('--num_steps', type=int, default=100,
                        help='Number of ODE integration steps')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda/cpu)')
    parser.add_argument('--model', type=str, default='unet',
                        choices=['unet', 'original'],
                        help='Model architecture')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    args = parser.parse_args()

    # Set seed for reproducibility
    set_seed(args.seed)
    print(f"Random seed: {args.seed}")

    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load Flow Matching models
    print("Loading FM models...")
    if args.model == 'unet':
        fm_x = FlowMatchingUNet().to(device)
        fm_y = FlowMatchingUNet().to(device)
    else:
        fm_x = FlowMatchingModel().to(device)
        fm_y = FlowMatchingModel().to(device)

    path_x = get_checkpoint_path('flow', 'x', None, 'best')
    path_y = get_checkpoint_path('flow', 'y', args.transform_type, 'best')

    if not os.path.exists(path_x):
        print(f"ERROR: FM_x checkpoint not found: {path_x}")
        print("Please train FM_x first: python src/train_flow.py --modality x")
        return

    if not os.path.exists(path_y):
        print(f"ERROR: FM_y checkpoint not found: {path_y}")
        print(f"Please train FM_y first: python src/train_flow.py --modality y --transform_type {args.transform_type}")
        return

    fm_x.load_state_dict(torch.load(path_x, map_location=device))
    fm_y.load_state_dict(torch.load(path_y, map_location=device))
    print(f"  Loaded FM_x from: {path_x}")
    print(f"  Loaded FM_y from: {path_y}")

    # Load ratio estimator (if guided)
    ratio_estimator = None
    if args.guidance_method != 'none':
        print("Loading ratio estimator...")
        ratio_estimator = RatioEstimator(loss_type=args.loss_type).to(device)

        path_ratio = get_checkpoint_path('ratio', args.loss_type, args.transform_type, 'best')

        if not os.path.exists(path_ratio):
            print(f"ERROR: Ratio estimator checkpoint not found: {path_ratio}")
            print(f"Please train ratio estimator first: python src/train_ratio.py --loss_type {args.loss_type} --transform_type {args.transform_type}")
            return

        ratio_estimator.load_state_dict(torch.load(path_ratio, map_location=device))
        print(f"  Loaded ratio estimator from: {path_ratio}")

    # Load data loader for MC guidance
    data_loader = None
    if args.guidance_method == 'mc_feng':
        print("Loading dataset for MC guidance...")
        data_loader = get_flow_dataloader(
            transform_type=args.transform_type,
            batch_size=args.mc_batch_size,
            train=True
        )

    # Sample
    print(f"\nSampling {args.num_samples} pairs...")
    print(f"  Guidance method: {args.guidance_method}")
    if args.guidance_method != 'none':
        print(f"  Guidance strength: {args.guidance_strength}")
        print(f"  MC batch size: {args.mc_batch_size}")
    print(f"  Integration steps: {args.num_steps}")

    samples_x, samples_y = sample_bimodal_guided(
        fm_x=fm_x,
        fm_y=fm_y,
        ratio_estimator=ratio_estimator,
        guidance_method=args.guidance_method,
        guidance_strength=args.guidance_strength,
        num_samples=args.num_samples,
        num_steps=args.num_steps,
        device=device,
        data_loader=data_loader,
        mc_batch_size=args.mc_batch_size
    )

    # Visualize
    os.makedirs('outputs', exist_ok=True)
    save_path = f"outputs/samples_{args.guidance_method}_gamma{args.guidance_strength}_{args.transform_type}.png"
    visualize_pairs(samples_x, samples_y, save_path, args.transform_type)

    print("\nSampling complete!")


if __name__ == '__main__':
    main()
