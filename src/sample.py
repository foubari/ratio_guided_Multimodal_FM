"""
Generate pairs (x, y) via guided sampling.

Usage:
    # Baseline (independent flow)
    python src/sample.py --transform_type rotate90 --guidance_method none --num_samples 64

    # Ratio-guided
    python src/sample.py --transform_type rotate90 --guidance_method grad_log_ratio --guidance_strength 2.0 --num_samples 64
"""
import argparse
import torch
import matplotlib.pyplot as plt
import os
import sys

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.flow_matching import FlowMatchingModel
from src.models.ratio_estimator import RatioEstimator
from src.utils.flow_utils import sample_bimodal_guided
from src.utils.path_utils import get_checkpoint_path


def visualize_pairs(samples_x, samples_y, save_path):
    """Display generated pairs (8x8 grid)."""
    num_samples = min(64, len(samples_x))
    num_rows = 8
    num_cols = 16

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(16, 8))

    for i in range(num_samples):
        row = i // 8
        col_x = (i % 8) * 2
        col_y = col_x + 1

        # Denormalize from [-1, 1] to [0, 1]
        img_x = (samples_x[i, 0].cpu().numpy() + 1) / 2
        img_y = (samples_y[i, 0].cpu().numpy() + 1) / 2

        axes[row, col_x].imshow(img_x, cmap='gray', vmin=0, vmax=1)
        axes[row, col_x].axis('off')

        axes[row, col_y].imshow(img_y, cmap='gray', vmin=0, vmax=1)
        axes[row, col_y].axis('off')

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
                        choices=['none', 'grad_log_ratio'],
                        help='Guidance method')
    parser.add_argument('--guidance_strength', type=float, default=1.0,
                        help='Guidance strength (gamma)')
    parser.add_argument('--loss_type', type=str, default='disc',
                        help='Loss type for ratio estimator')
    parser.add_argument('--num_samples', type=int, default=64,
                        help='Number of samples to generate')
    parser.add_argument('--num_steps', type=int, default=100,
                        help='Number of ODE integration steps')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda/cpu)')
    args = parser.parse_args()

    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load Flow Matching models
    print("Loading FM models...")
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

    # Sample
    print(f"\nSampling {args.num_samples} pairs...")
    print(f"  Guidance method: {args.guidance_method}")
    if args.guidance_method != 'none':
        print(f"  Guidance strength: {args.guidance_strength}")
    print(f"  Integration steps: {args.num_steps}")

    samples_x, samples_y = sample_bimodal_guided(
        fm_x=fm_x,
        fm_y=fm_y,
        ratio_estimator=ratio_estimator,
        guidance_method=args.guidance_method,
        guidance_strength=args.guidance_strength,
        num_samples=args.num_samples,
        num_steps=args.num_steps,
        device=device
    )

    # Visualize
    os.makedirs('outputs', exist_ok=True)
    save_path = f"outputs/samples_{args.guidance_method}_gamma{args.guidance_strength}_{args.transform_type}.png"
    visualize_pairs(samples_x, samples_y, save_path)

    print("\nSampling complete!")


if __name__ == '__main__':
    main()
