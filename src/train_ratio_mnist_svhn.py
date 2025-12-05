"""
Train ratio estimator for MNIST-SVHN pairs.

The ratio estimator learns r_1(mnist, svhn) = q(mnist, svhn) / p_ind(mnist, svhn)
where pairs with same digit have high ratio, different digits have low ratio.

Usage:
    python src/train_ratio_mnist_svhn.py --epochs 30
"""
import argparse
import torch
import torch.nn.functional as F
import sys
import os
from tqdm import tqdm

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.ratio_flexible import RatioEstimatorMNISTSVHN
from src.data.mnist_svhn_dataset import get_mnist_svhn_ratio_dataloader
from src.utils.losses import get_ratio_loss
from src.utils import set_seed


def train_epoch(model, dataloader, loss_fn, optimizer, device):
    """Train one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    num_batches = 0

    for batch in tqdm(dataloader, desc="Training ratio"):
        x = batch['x'].to(device)       # [B, 1, 32, 32] MNIST
        y = batch['y'].to(device)       # [B, 3, 32, 32] SVHN
        is_real = batch['is_real'].to(device).float()  # [B]

        # Forward
        scores = model(x, y)

        # Separate real and fake scores for the loss function
        real_mask = is_real == 1
        fake_mask = is_real == 0
        
        scores_real = scores[real_mask]
        scores_fake = scores[fake_mask]
        
        # Compute loss (returns tuple: loss, metrics)
        if len(scores_real) > 0 and len(scores_fake) > 0:
            loss, _ = loss_fn(scores_real, scores_fake)
        elif len(scores_real) > 0:
            # Only real samples in batch
            loss = F.binary_cross_entropy_with_logits(scores_real, torch.ones_like(scores_real))
        else:
            # Only fake samples in batch
            loss = F.binary_cross_entropy_with_logits(scores_fake, torch.zeros_like(scores_fake))

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accuracy
        preds = (torch.sigmoid(scores) > 0.5).float()
        correct += (preds == is_real).sum().item()
        total += len(is_real)

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches
    accuracy = correct / total

    return {'loss': avg_loss, 'accuracy': accuracy}


def main():
    parser = argparse.ArgumentParser(description='Train ratio estimator for MNIST-SVHN')
    parser.add_argument('--loss_type', type=str, default='disc',
                        choices=['disc', 'rulsif'],
                        help='Loss type')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--real_fake_ratio', type=float, default=0.5,
                        help='Proportion of real pairs')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda/cpu)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)
    print(f"Random seed: {args.seed}")

    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Dataset
    dataloader = get_mnist_svhn_ratio_dataloader(
        batch_size=args.batch_size,
        real_fake_ratio=args.real_fake_ratio,
        train=True,
        num_workers=4
    )
    print(f"Training samples: {len(dataloader.dataset)}")
    print(f"Real/fake ratio: {args.real_fake_ratio}")

    # Model
    model = RatioEstimatorMNISTSVHN(loss_type=args.loss_type).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss
    loss_fn = get_ratio_loss(loss_type=args.loss_type)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    best_loss = float('inf')
    patience = 5
    patience_counter = 0

    os.makedirs('checkpoints', exist_ok=True)

    for epoch in range(args.epochs):
        metrics = train_epoch(model, dataloader, loss_fn, optimizer, device)

        metrics_str = ' - '.join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        print(f"Epoch {epoch+1}/{args.epochs} - {metrics_str}")

        # Save best model
        if metrics['loss'] < best_loss:
            best_loss = metrics['loss']
            patience_counter = 0
            path = f'checkpoints/ratio_{args.loss_type}_mnist_svhn_best.pth'
            torch.save(model.state_dict(), path)
            print(f"  → Saved best model: {path}")
        else:
            patience_counter += 1

        # Regular checkpoint
        if (epoch + 1) % 10 == 0:
            path = f'checkpoints/ratio_{args.loss_type}_mnist_svhn_epoch{epoch+1}.pth'
            torch.save(model.state_dict(), path)
            print(f"  → Saved checkpoint: {path}")

        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping after {epoch+1} epochs")
            break

    print("\nTraining complete!")
    print(f"Best loss: {best_loss:.4f}")


if __name__ == '__main__':
    main()
