"""
Train two independent Flow Matching models:
- FM_x: prior → q_x (standard MNIST images)
- FM_y: prior → q_y (transformed MNIST images)

Usage:
    python src/train_flow.py --modality x --epochs 50
    python src/train_flow.py --modality y --transform_type rotate90 --epochs 50
"""
import argparse
import torch
import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.flow_matching import FlowMatchingModel
from src.models.unet import FlowMatchingUNet
from src.data.mnist_dataset import get_flow_dataloader
from src.utils.flow_utils import CFMSchedule, train_flow_matching_epoch
from src.utils.path_utils import get_checkpoint_path
from src.utils import set_seed


def main():
    parser = argparse.ArgumentParser(description='Train Flow Matching model')
    parser.add_argument('--modality', type=str, required=True, choices=['x', 'y'],
                        help='Modality to train (x=standard, y=transformed)')
    parser.add_argument('--transform_type', type=str, default='rotate90',
                        help='Transformation type for y modality')
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
    parser.add_argument('--model', type=str, default='unet',
                        choices=['unet', 'original'],
                        help='Model architecture (unet=lightweight, original=encoder-decoder)')
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
    transform_for_loader = args.transform_type if args.modality == 'y' else None
    dataloader = get_flow_dataloader(
        transform_type=transform_for_loader,
        batch_size=args.batch_size,
        train=True
    )
    print(f"Training modality: {args.modality}")
    if args.modality == 'y':
        print(f"Transform type: {args.transform_type}")

    # Model
    if args.model == 'unet':
        model = FlowMatchingUNet().to(device)
        print(f"Using FlowMatchingUNet (lightweight U-Net)")
    else:
        model = FlowMatchingModel().to(device)
        print(f"Using FlowMatchingModel (original encoder-decoder)")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    schedule = CFMSchedule()

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Training loop
    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(args.epochs):
        avg_loss = train_flow_matching_epoch(
            model, dataloader, optimizer, schedule, device, modality=args.modality
        )

        print(f"Epoch {epoch+1}/{args.epochs} - Loss: {avg_loss:.4f}")

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            path = get_checkpoint_path(
                'flow', args.modality, transform_for_loader, 'best'
            )
            torch.save(model.state_dict(), path)
            print(f"  → Saved best model: {path}")
        else:
            patience_counter += 1

        # Save checkpoint every N epochs
        if (epoch + 1) % args.save_every == 0:
            path = get_checkpoint_path(
                'flow', args.modality, transform_for_loader, f'epoch{epoch+1}'
            )
            torch.save(model.state_dict(), path)
            print(f"  → Saved checkpoint: {path}")

        # Early stopping
        if patience_counter >= args.patience:
            print(f"\nEarly stopping after {epoch+1} epochs (patience={args.patience})")
            break

    print("\nTraining complete!")
    print(f"Best loss: {best_loss:.4f}")


if __name__ == '__main__':
    main()
