"""
Train Flow Matching model for MNIST (32x32 version for MNIST-SVHN experiments).

This trains FM_mnist32: prior → q_mnist (1x32x32 grayscale)

Usage:
    python src/train_flow_mnist32.py --epochs 50
"""
import argparse
import torch
import torch.nn.functional as F
import sys
import os
from tqdm import tqdm

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.unet_flexible import FlowMatchingUNetMNIST
from src.data.mnist_svhn_dataset import get_mnist32_flow_dataloader
from src.utils import set_seed
from torch.utils.data import DataLoader


def train_epoch(model, dataloader, optimizer, device):
    """Train one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in tqdm(dataloader, desc="Training FM_mnist32"):
        x_1 = batch['x'].to(device)  # [B, 1, 32, 32]
        B = x_1.shape[0]

        # Sample timesteps
        t = torch.rand(B, device=device)

        # Sample prior
        x_0 = torch.randn_like(x_1)

        # Linear interpolation
        t_exp = t.view(B, 1, 1, 1)
        x_t = (1 - t_exp) * x_0 + t_exp * x_1

        # Target velocity
        u_t_target = x_1 - x_0

        # Predict velocity
        v_t = model(x_t, t)

        # MSE loss
        loss = F.mse_loss(v_t, u_t_target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def main():
    parser = argparse.ArgumentParser(description='Train Flow Matching for MNIST 32x32')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda/cpu)')
    parser.add_argument('--save_every', type=int, default=10,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)
    print(f"Random seed: {args.seed}")

    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Dataset
    dataloader = get_mnist32_flow_dataloader(batch_size=args.batch_size, train=True)
    print(f"MNIST 32x32 training samples: {len(dataloader.dataset)}")

    # Model (MNIST 32x32)
    model = FlowMatchingUNetMNIST(img_size=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    start_epoch = 0
    best_loss = float('inf')
    
    # Resume from checkpoint if specified
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch']
            if 'best_loss' in checkpoint:
                best_loss = checkpoint['best_loss']
            print(f"  Resumed from epoch {start_epoch}, best_loss={best_loss:.4f}")
        else:
            model.load_state_dict(checkpoint)
            print(f"  Loaded model weights (old format)")

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Training loop
    patience_counter = 0

    os.makedirs('checkpoints', exist_ok=True)

    for epoch in range(start_epoch, args.epochs):
        avg_loss = train_epoch(model, dataloader, optimizer, device)

        print(f"Epoch {epoch+1}/{args.epochs} - Loss: {avg_loss:.4f}")

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            path = 'checkpoints/flow_mnist32_best.pth'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_loss': best_loss,
            }, path)
            print(f"  → Saved best model: {path}")
        else:
            patience_counter += 1

        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            path = f'checkpoints/flow_mnist32_epoch{epoch+1}.pth'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_loss': best_loss,
            }, path)
            print(f"  → Saved checkpoint: {path}")

        # Early stopping
        if patience_counter >= args.patience:
            print(f"\nEarly stopping after {epoch+1} epochs")
            break

    print("\nTraining complete!")
    print(f"Best loss: {best_loss:.4f}")


if __name__ == '__main__':
    main()
