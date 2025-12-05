"""
Train Flow Matching model for SVHN.

This trains FM_svhn: prior → q_svhn (32x32 RGB images)

Usage:
    python src/train_flow_svhn.py --epochs 50
"""
import argparse
import torch
import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.unet_flexible import FlowMatchingUNetSVHN
from src.data.mnist_svhn_dataset import get_svhn_flow_dataloader
from src.utils.flow_utils import CFMSchedule
from src.utils import set_seed


def train_epoch(model, dataloader, optimizer, schedule, device):
    """Train one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    from tqdm import tqdm
    import torch.nn.functional as F

    for batch in tqdm(dataloader, desc="Training FM_svhn"):
        x_1 = batch['x'].to(device)  # [B, 3, 32, 32]
        B = x_1.shape[0]

        # Sample timesteps
        t = torch.rand(B, device=device)

        # Get interpolated samples and targets
        x_t, u_t_target = schedule.add_noise_flexible(x_1, t)

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


class CFMScheduleFlexible(CFMSchedule):
    """Extended CFM schedule for flexible image sizes."""
    
    def add_noise_flexible(self, x_1, t):
        """
        Interpolate x_0 → x_1 at time t for any image shape.
        """
        B = x_1.shape[0]
        device = x_1.device

        # Sample prior
        x_0 = torch.randn_like(x_1)

        # Reshape t for broadcasting
        t_shape = [B] + [1] * (len(x_1.shape) - 1)
        t = t.view(*t_shape)

        # Linear interpolation
        x_t = (1 - t) * x_0 + t * x_1

        # Conditional velocity
        u_t_target = x_1 - x_0

        return x_t, u_t_target


def main():
    parser = argparse.ArgumentParser(description='Train Flow Matching model for SVHN')
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
                        help='Random seed for reproducibility')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from (e.g., checkpoints/flow_svhn_best.pth)')
    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)
    print(f"Random seed: {args.seed}")

    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Dataset
    dataloader = get_svhn_flow_dataloader(
        batch_size=args.batch_size,
        train=True,
        num_workers=4
    )
    print(f"SVHN training samples: {len(dataloader.dataset)}")

    # Model
    model = FlowMatchingUNetSVHN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    schedule = CFMScheduleFlexible()
    
    start_epoch = 0
    best_loss = float('inf')
    
    # Resume from checkpoint if specified
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        
        # Handle both old format (just state_dict) and new format (full checkpoint)
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
            # Old format: just model state dict
            model.load_state_dict(checkpoint)
            print(f"  Loaded model weights (old format, no optimizer state)")

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Training loop
    patience_counter = 0

    os.makedirs('checkpoints', exist_ok=True)

    for epoch in range(start_epoch, args.epochs):
        avg_loss = train_epoch(model, dataloader, optimizer, schedule, device)

        print(f"Epoch {epoch+1}/{args.epochs} - Loss: {avg_loss:.4f}")

        # Save best model (full checkpoint for resume)
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            path = 'checkpoints/flow_svhn_best.pth'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_loss': best_loss,
            }, path)
            print(f"  → Saved best model: {path}")
        else:
            patience_counter += 1

        # Save checkpoint every N epochs
        if (epoch + 1) % args.save_every == 0:
            path = f'checkpoints/flow_svhn_epoch{epoch+1}.pth'
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
