"""
Generate pairs (MNIST, SVHN) via guided sampling.

Usage:
    # Baseline (independent flow)
    python src/sample_mnist_svhn.py --guidance_method none --num_samples 64

    # MC-guided (Feng et al.)
    python src/sample_mnist_svhn.py --guidance_method mc_feng --guidance_strength 0.5 --num_samples 64
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.unet_flexible import FlowMatchingUNetMNIST, FlowMatchingUNetSVHN
from src.models.ratio_flexible import RatioEstimatorMNISTSVHN
from src.utils import set_seed, load_checkpoint
import math


def log_gaussian_density(x, mu, sigma):
    """Compute log probability under isotropic Gaussian."""
    dim = x[0].numel()
    diff = (x - mu).view(x.shape[0], -1)
    log_prob = -0.5 * (diff ** 2).sum(dim=1) / (sigma ** 2 + 1e-8)
    log_prob = log_prob - 0.5 * dim * math.log(2 * math.pi) - dim * math.log(sigma + 1e-8)
    return log_prob


def sample_bimodal_guided_mnist_svhn(
    fm_mnist,
    fm_svhn,
    ratio_estimator=None,
    guidance_method='none',
    guidance_strength=0.0,
    num_samples=16,
    num_steps=100,
    device='cuda',
    mc_batch_size=64,
):
    """
    Sample bimodal pairs (MNIST, SVHN) with optional guidance.
    
    Args:
        fm_mnist: Flow matching model for MNIST (1x32x32)
        fm_svhn: Flow matching model for SVHN (3x32x32)
        ratio_estimator: RatioEstimatorMNISTSVHN
        guidance_method: 'none' or 'mc_feng'
        guidance_strength: γ in [0, 1]
        num_samples: Number of pairs to generate
        num_steps: Number of ODE integration steps
        device: cuda/cpu
        mc_batch_size: Number of MC samples (generated from flows, NOT from dataset)
        
    Returns:
        samples_mnist: [N, 1, 32, 32]
        samples_svhn: [N, 3, 32, 32]
    """
    fm_mnist.eval()
    fm_svhn.eval()
    if ratio_estimator is not None:
        ratio_estimator.eval()

    # Initialize from prior
    x_t = torch.randn(num_samples, 1, 32, 32, device=device)  # MNIST
    y_t = torch.randn(num_samples, 3, 32, 32, device=device)  # SVHN

    dt = 1.0 / num_steps
    eps = 1e-3

    # Pre-load MC samples if using guidance
    mc_x1_samples = None
    mc_y1_samples = None
    mc_ratios = None

    if guidance_method == 'mc_feng' and ratio_estimator is not None:
        print(f"  Generating {mc_batch_size} independent MC samples from flows...")
        
        # Generate MNIST samples independently
        mc_x1_samples = torch.randn(mc_batch_size, 1, 32, 32, device=device)
        for step in range(num_steps):
            t = step * dt
            t_batch = torch.full((mc_batch_size,), t, device=device)
            with torch.no_grad():
                v = fm_mnist(mc_x1_samples, t_batch)
            mc_x1_samples = mc_x1_samples + v * dt
        
        # Generate SVHN samples independently
        mc_y1_samples = torch.randn(mc_batch_size, 3, 32, 32, device=device)
        for step in range(num_steps):
            t = step * dt
            t_batch = torch.full((mc_batch_size,), t, device=device)
            with torch.no_grad():
                v = fm_svhn(mc_y1_samples, t_batch)
            mc_y1_samples = mc_y1_samples + v * dt
        
        print(f"  Generated MC samples: x shape={mc_x1_samples.shape}, y shape={mc_y1_samples.shape}")

        # Pre-compute ratios for all combinations
        with torch.no_grad():
            log_r = ratio_estimator.log_ratio(mc_x1_samples, mc_y1_samples)
            mc_ratios = log_r.exp()
            print(f"  MC ratios: min={mc_ratios.min():.4f}, max={mc_ratios.max():.4f}, mean={mc_ratios.mean():.4f}")

    for step in tqdm(range(num_steps), desc="Sampling"):
        t = step * dt
        t_batch_mnist = torch.full((num_samples,), t, device=device)
        t_batch_svhn = torch.full((num_samples,), t, device=device)

        with torch.no_grad():
            v_x = fm_mnist(x_t, t_batch_mnist)
            v_y = fm_svhn(y_t, t_batch_svhn)

        # MC Guidance
        if guidance_method == 'mc_feng' and mc_x1_samples is not None and t > eps:
            with torch.no_grad():
                N_mc = mc_x1_samples.shape[0]
                sigma_t = (1 - t + eps)

                # Expand for broadcasting
                x_t_exp = x_t.unsqueeze(1)  # [B, 1, 1, 32, 32]
                y_t_exp = y_t.unsqueeze(1)  # [B, 1, 3, 32, 32]
                mc_x1_exp = mc_x1_samples.unsqueeze(0)  # [1, N_mc, 1, 32, 32]
                mc_y1_exp = mc_y1_samples.unsqueeze(0)  # [1, N_mc, 3, 32, 32]

                mu_x = t * mc_x1_exp
                mu_y = t * mc_y1_exp

                # Log probabilities
                diff_x = (x_t_exp - mu_x).view(num_samples, N_mc, -1)
                diff_y = (y_t_exp - mu_y).view(num_samples, N_mc, -1)

                log_p_x = -0.5 * (diff_x ** 2).sum(dim=-1) / (sigma_t ** 2)
                log_p_y = -0.5 * (diff_y ** 2).sum(dim=-1) / (sigma_t ** 2)
                log_p_joint = log_p_x + log_p_y

                # Normalize
                log_p_max = log_p_joint.max(dim=1, keepdim=True)[0]
                p_joint = (log_p_joint - log_p_max).exp()

                p_bar = p_joint.mean(dim=1, keepdim=True) + 1e-10
                mc_ratios_exp = mc_ratios.unsqueeze(0)
                Z_bar = (mc_ratios_exp * p_joint).mean(dim=1, keepdim=True) + 1e-10

                # Weights
                weights = (mc_ratios_exp / Z_bar) * (p_joint / p_bar)
                weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-10)

                # Conditional velocities
                v_cond_x = (mc_x1_exp.squeeze(0) - x_t.unsqueeze(1)) / (1 - t + eps)
                v_cond_y = (mc_y1_exp.squeeze(0) - y_t.unsqueeze(1)) / (1 - t + eps)

                # Weighted sum
                weights_exp_x = weights.view(num_samples, N_mc, 1, 1, 1)
                weights_exp_y = weights.view(num_samples, N_mc, 1, 1, 1)

                g_x = (weights_exp_x * v_cond_x).sum(dim=1)
                g_y = (weights_exp_y * v_cond_y).sum(dim=1)

                # Apply guidance
                v_x = (1 - guidance_strength) * v_x + guidance_strength * g_x
                v_y = (1 - guidance_strength) * v_y + guidance_strength * g_y

        # Euler step
        x_t = x_t + v_x * dt
        y_t = y_t + v_y * dt

    return x_t, y_t


def visualize_pairs_mnist_svhn(samples_mnist, samples_svhn, save_path, num_cols=8, num_rows=4):
    """
    Display generated MNIST-SVHN pairs.
    
    Layout: Each column shows (MNIST, SVHN) stacked vertically.
    """
    num_pairs = min(num_cols * num_rows, len(samples_mnist))

    fig_width = num_cols * 1.5
    fig_height = num_rows * 2 * 1.5 + 0.5

    fig, axes = plt.subplots(num_rows * 2, num_cols, figsize=(fig_width, fig_height))

    if num_rows * 2 == 1:
        axes = axes.reshape(1, -1)
    if num_cols == 1:
        axes = axes.reshape(-1, 1)

    for idx in range(num_pairs):
        col = idx % num_cols
        pair_row = idx // num_cols

        row_mnist = pair_row * 2
        row_svhn = pair_row * 2 + 1

        # MNIST (grayscale)
        img_mnist = (samples_mnist[idx, 0].cpu().numpy() + 1) / 2
        img_mnist = np.clip(img_mnist, 0, 1)

        # SVHN (RGB) - shape is [3, 32, 32], need to transpose to [32, 32, 3]
        img_svhn = (samples_svhn[idx].cpu().numpy() + 1) / 2
        img_svhn = np.clip(img_svhn, 0, 1)
        img_svhn = img_svhn.transpose(1, 2, 0)  # [H, W, C]

        axes[row_mnist, col].imshow(img_mnist, cmap='gray', vmin=0, vmax=1)
        axes[row_mnist, col].axis('off')

        axes[row_svhn, col].imshow(img_svhn)
        axes[row_svhn, col].axis('off')

    # Turn off remaining axes
    for idx in range(num_pairs, num_cols * num_rows):
        col = idx % num_cols
        pair_row = idx // num_cols
        row_mnist = pair_row * 2
        row_svhn = pair_row * 2 + 1
        axes[row_mnist, col].axis('off')
        axes[row_svhn, col].axis('off')

    # Add row labels
    for pair_row in range(num_rows):
        row_mnist = pair_row * 2
        row_svhn = pair_row * 2 + 1
        axes[row_mnist, 0].set_ylabel('MNIST', fontsize=9, rotation=0, labelpad=25, va='center')
        axes[row_svhn, 0].set_ylabel('SVHN', fontsize=9, rotation=0, labelpad=25, va='center')
        axes[row_mnist, 0].yaxis.set_visible(True)
        axes[row_svhn, 0].yaxis.set_visible(True)

    fig.suptitle('Generated Pairs (MNIST, SVHN)', fontsize=12, fontweight='bold')

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Sample MNIST-SVHN pairs')
    parser.add_argument('--guidance_method', type=str, default='none',
                        choices=['none', 'mc_feng'],
                        help='Guidance method')
    parser.add_argument('--guidance_strength', type=float, default=0.5,
                        help='Guidance strength (0=independent, 1=full guidance)')
    parser.add_argument('--mc_batch_size', type=int, default=256,
                        help='Number of MC samples')
    parser.add_argument('--loss_type', type=str, default='disc',
                        help='Loss type for ratio estimator')
    parser.add_argument('--num_samples', type=int, default=32,
                        help='Number of samples to generate')
    parser.add_argument('--num_steps', type=int, default=100,
                        help='Number of ODE integration steps')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    args = parser.parse_args()

    set_seed(args.seed)
    print(f"Random seed: {args.seed}")

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load Flow Matching models
    print("Loading FM models...")

    fm_mnist = FlowMatchingUNetMNIST(img_size=32).to(device)
    fm_svhn = FlowMatchingUNetSVHN().to(device)

    path_mnist = 'checkpoints/flow_mnist32_best.pth'
    path_svhn = 'checkpoints/flow_svhn_best.pth'

    if not os.path.exists(path_mnist):
        print(f"ERROR: FM_mnist checkpoint not found: {path_mnist}")
        print("Please train first: python src/train_flow_mnist32.py")
        return

    if not os.path.exists(path_svhn):
        print(f"ERROR: FM_svhn checkpoint not found: {path_svhn}")
        print("Please train first: python src/train_flow_svhn.py")
        return

    load_checkpoint(fm_mnist, path_mnist, device)
    load_checkpoint(fm_svhn, path_svhn, device)
    print(f"  Loaded FM_mnist from: {path_mnist}")
    print(f"  Loaded FM_svhn from: {path_svhn}")

    # Load ratio estimator (if guided)
    ratio_estimator = None
    if args.guidance_method != 'none':
        print("Loading ratio estimator...")
        ratio_estimator = RatioEstimatorMNISTSVHN(loss_type=args.loss_type).to(device)

        path_ratio = f'checkpoints/ratio_{args.loss_type}_mnist_svhn_best.pth'

        if not os.path.exists(path_ratio):
            print(f"ERROR: Ratio estimator not found: {path_ratio}")
            print("Please train first: python src/train_ratio_mnist_svhn.py")
            return

        ratio_estimator.load_state_dict(torch.load(path_ratio, map_location=device))
        print(f"  Loaded ratio estimator from: {path_ratio}")

    # Sample
    print(f"\nSampling {args.num_samples} pairs...")
    print(f"  Guidance method: {args.guidance_method}")
    if args.guidance_method != 'none':
        print(f"  Guidance strength: {args.guidance_strength}")

    samples_mnist, samples_svhn = sample_bimodal_guided_mnist_svhn(
        fm_mnist=fm_mnist,
        fm_svhn=fm_svhn,
        ratio_estimator=ratio_estimator,
        guidance_method=args.guidance_method,
        guidance_strength=args.guidance_strength,
        num_samples=args.num_samples,
        num_steps=args.num_steps,
        device=device,
        mc_batch_size=args.mc_batch_size
    )

    # Visualize
    os.makedirs('outputs/mnist_svhn', exist_ok=True)
    save_path = f"outputs/mnist_svhn/samples_{args.guidance_method}_gamma{args.guidance_strength}.png"
    visualize_pairs_mnist_svhn(samples_mnist, samples_svhn, save_path)

    print("\nSampling complete!")


if __name__ == '__main__':
    main()
