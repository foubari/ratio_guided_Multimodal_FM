"""
Train ratio estimator r̂(x, y) = q(x,y) / p_1^ind(x,y).

Usage:
    python src/train_ratio.py --loss_type disc --transform_type rotate90 --epochs 30
"""
import argparse
import torch
import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.ratio_estimator import RatioEstimator
from src.data.mnist_dataset import get_ratio_dataloader
from src.utils.losses import get_ratio_loss
from src.utils.trainer import RatioTrainer
from src.utils.path_utils import get_checkpoint_path
from src.utils import set_seed


def main():
    parser = argparse.ArgumentParser(description='Train ratio estimator')
    parser.add_argument('--loss_type', type=str, default='disc',
                        choices=['disc', 'rulsif'],
                        help='Loss type')
    parser.add_argument('--transform_type', type=str, default='rotate90',
                        help='Transformation type for y modality')
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

    # Loss-specific hyperparameters
    parser.add_argument('--rulsif_alpha', type=float, default=0.2,
                        help='RuLSIF alpha parameter')
    parser.add_argument('--lambda_penalty', type=float, default=0.1,
                        help='RuLSIF lambda penalty')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    args = parser.parse_args()

    # Set seed for reproducibility
    set_seed(args.seed)
    print(f"Random seed: {args.seed}")

    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Dataset
    dataloader = get_ratio_dataloader(
        transform_type=args.transform_type,
        batch_size=args.batch_size,
        real_fake_ratio=args.real_fake_ratio
    )
    print(f"Transform type: {args.transform_type}")
    print(f"Loss type: {args.loss_type}")
    print(f"Real/fake ratio: {args.real_fake_ratio}")

    # Model
    model = RatioEstimator(loss_type=args.loss_type).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss
    loss_fn = get_ratio_loss(
        loss_type=args.loss_type,
        alpha=args.rulsif_alpha,
        lambda_penalty=args.lambda_penalty
    )

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Trainer
    trainer = RatioTrainer(model, loss_fn, optimizer, device)

    # Training loop
    best_loss = float('inf')
    patience = 5
    patience_counter = 0

    for epoch in range(args.epochs):
        metrics = trainer.train_epoch(dataloader)

        # Print metrics
        metrics_str = ' - '.join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        print(f"Epoch {epoch+1}/{args.epochs} - {metrics_str}")

        # Save best model
        if metrics['loss'] < best_loss:
            best_loss = metrics['loss']
            patience_counter = 0
            path = get_checkpoint_path(
                'ratio', args.loss_type, args.transform_type, 'best'
            )
            torch.save(model.state_dict(), path)
            print(f"  → Saved best model: {path}")
        else:
            patience_counter += 1

        # Regular checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            path = get_checkpoint_path(
                'ratio', args.loss_type, args.transform_type, f'epoch{epoch+1}'
            )
            torch.save(model.state_dict(), path)
            print(f"  → Saved checkpoint: {path}")

        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping after {epoch+1} epochs (patience={patience})")
            break

    print("\nTraining complete!")
    print(f"Best loss: {best_loss:.4f}")


if __name__ == '__main__':
    main()
