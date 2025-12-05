"""
Train classifiers for MNIST-SVHN evaluation.

Trains:
1. MNIST classifier (32x32 grayscale)
2. SVHN classifier (32x32 RGB)

Usage:
    python src/train_classifiers_mnist_svhn.py
"""
import argparse
import torch
import torch.nn.functional as F
import sys
import os
from tqdm import tqdm
from torch.utils.data import DataLoader

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.svhn_classifier import SVHNClassifier, MNISTClassifier32
from src.data.mnist_svhn_dataset import MNISTSVHNDataset
from src.utils import set_seed
from torchvision import datasets, transforms


def get_mnist32_dataloader(batch_size=128, train=True):
    """Get MNIST resized to 32x32."""
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    dataset = datasets.MNIST(
        root='./data',
        train=train,
        transform=transform,
        download=True
    )
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=4)


def get_svhn_dataloader(batch_size=128, train=True):
    """Get SVHN dataloader."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    dataset = datasets.SVHN(
        root='./data',
        split='train' if train else 'test',
        transform=transform,
        download=True
    )
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=4)


def train_classifier(model, train_loader, test_loader, device, epochs=10, lr=1e-3, name=""):
    """Train a classifier."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    best_acc = 0.0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for x, y in tqdm(train_loader, desc=f"[{name}] Epoch {epoch+1}/{epochs}"):
            x, y = x.to(device), y.to(device)
            
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += len(y)
        
        train_acc = correct / total
        
        # Evaluation
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += len(y)
        
        test_acc = correct / total
        
        print(f"  Train acc: {train_acc:.4f}, Test acc: {test_acc:.4f}")
        
        if test_acc > best_acc:
            best_acc = test_acc
    
    return best_acc


def main():
    parser = argparse.ArgumentParser(description='Train classifiers for MNIST-SVHN')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    os.makedirs('checkpoints', exist_ok=True)

    # ========== Train MNIST 32x32 Classifier ==========
    print("\n" + "="*60)
    print("Training MNIST 32x32 Classifier")
    print("="*60)
    
    mnist_train = get_mnist32_dataloader(args.batch_size, train=True)
    mnist_test = get_mnist32_dataloader(args.batch_size, train=False)
    
    mnist_classifier = MNISTClassifier32().to(device)
    print(f"Parameters: {sum(p.numel() for p in mnist_classifier.parameters()):,}")
    
    best_acc = train_classifier(
        mnist_classifier, mnist_train, mnist_test,
        device, epochs=args.epochs, lr=args.lr, name="MNIST32"
    )
    
    path = 'checkpoints/mnist32_classifier.pth'
    torch.save(mnist_classifier.state_dict(), path)
    print(f"Saved MNIST 32x32 classifier: {path} (best acc: {best_acc:.4f})")

    # ========== Train SVHN Classifier ==========
    print("\n" + "="*60)
    print("Training SVHN Classifier")
    print("="*60)
    
    svhn_train = get_svhn_dataloader(args.batch_size, train=True)
    svhn_test = get_svhn_dataloader(args.batch_size, train=False)
    
    svhn_classifier = SVHNClassifier().to(device)
    print(f"Parameters: {sum(p.numel() for p in svhn_classifier.parameters()):,}")
    
    best_acc = train_classifier(
        svhn_classifier, svhn_train, svhn_test,
        device, epochs=args.epochs, lr=args.lr, name="SVHN"
    )
    
    path = 'checkpoints/svhn_classifier.pth'
    torch.save(svhn_classifier.state_dict(), path)
    print(f"Saved SVHN classifier: {path} (best acc: {best_acc:.4f})")

    print("\n" + "="*60)
    print("All classifiers trained!")
    print("="*60)


if __name__ == '__main__':
    main()
