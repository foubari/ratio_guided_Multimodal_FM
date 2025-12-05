"""
MNIST-SVHN paired dataset for bimodal flow matching.

Pairs MNIST and SVHN images by their digit label (0-9).
MNIST: 1x28x28 grayscale -> resized to 1x32x32
SVHN: 3x32x32 RGB
"""
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import numpy as np


class MNISTSVHNDataset(Dataset):
    """
    Dataset of aligned pairs (mnist, svhn) where both show the same digit.

    Args:
        root: Path to data folder
        train: Train or test split
        download: Download if not present
    """

    def __init__(self, root='./data', train=True, download=True):
        self.train = train

        # MNIST transforms: resize to 32x32, normalize to [-1, 1]
        mnist_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # [-1, 1]
        ])

        # SVHN transforms: normalize to [-1, 1]
        svhn_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # [-1, 1]
        ])

        # Load datasets
        self.mnist = datasets.MNIST(
            root=root,
            train=train,
            transform=mnist_transform,
            download=download
        )

        self.svhn = datasets.SVHN(
            root=root,
            split='train' if train else 'test',
            transform=svhn_transform,
            download=download
        )

        # Build index by label for efficient pairing
        self._build_label_indices()

    def _build_label_indices(self):
        """Build dictionary mapping label -> list of indices for each dataset."""
        # MNIST indices by label
        self.mnist_by_label = {i: [] for i in range(10)}
        for idx, (_, label) in enumerate(self.mnist):
            self.mnist_by_label[label].append(idx)

        # SVHN indices by label
        self.svhn_by_label = {i: [] for i in range(10)}
        for idx in range(len(self.svhn)):
            label = int(self.svhn.labels[idx])
            self.svhn_by_label[label].append(idx)

        # Convert to numpy for faster random access
        for i in range(10):
            self.mnist_by_label[i] = np.array(self.mnist_by_label[i])
            self.svhn_by_label[i] = np.array(self.svhn_by_label[i])

        # Print stats
        print(f"MNIST-SVHN Dataset ({'train' if self.train else 'test'}):")
        for i in range(10):
            print(f"  Digit {i}: MNIST={len(self.mnist_by_label[i])}, SVHN={len(self.svhn_by_label[i])}")

    def __len__(self):
        # Use MNIST size as base (it's smaller: 60k vs 73k)
        return len(self.mnist)

    def __getitem__(self, idx):
        """
        Returns:
            dict with keys:
                'x': [1, 32, 32] - MNIST image (grayscale, resized)
                'y': [3, 32, 32] - SVHN image (RGB)
                'label': int - digit class (0-9)
        """
        # Get MNIST sample
        mnist_img, label = self.mnist[idx]

        # Get random SVHN sample with same label
        svhn_indices = self.svhn_by_label[label]
        svhn_idx = np.random.choice(svhn_indices)
        svhn_img, svhn_label = self.svhn[svhn_idx]

        return {
            'x': mnist_img,  # [1, 32, 32]
            'y': svhn_img,   # [3, 32, 32]
            'label': label
        }


class MNISTSVHNRatioDataset(Dataset):
    """
    Dataset for training ratio estimator on MNIST-SVHN pairs.

    Generates:
    - Real pairs: (mnist, svhn) with same label
    - Fake pairs: (mnist, svhn) with different labels (product of marginals)

    Args:
        base_dataset: MNISTSVHNDataset instance
        real_fake_ratio: Proportion of real pairs (default=0.5)
    """

    def __init__(self, base_dataset, real_fake_ratio=0.5):
        self.dataset = base_dataset
        self.real_fake_ratio = real_fake_ratio
        self.size = len(base_dataset)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        """
        Returns:
            dict with keys:
                'x': [1, 32, 32] - MNIST
                'y': [3, 32, 32] - SVHN
                'is_real': 0 or 1
                'label_x': int
                'label_y': int
        """
        is_real = torch.rand(1).item() < self.real_fake_ratio

        if is_real:
            # Real pair: same label
            sample = self.dataset[idx]
            return {
                'x': sample['x'],
                'y': sample['y'],
                'is_real': 1,
                'label_x': sample['label'],
                'label_y': sample['label']
            }
        else:
            # Fake pair: random SVHN from different label
            mnist_img, label_x = self.dataset.mnist[idx]

            # Choose different label
            other_labels = [l for l in range(10) if l != label_x]
            label_y = np.random.choice(other_labels)

            # Get random SVHN with that label
            svhn_indices = self.dataset.svhn_by_label[label_y]
            svhn_idx = np.random.choice(svhn_indices)
            svhn_img, _ = self.dataset.svhn[svhn_idx]

            return {
                'x': mnist_img,
                'y': svhn_img,
                'is_real': 0,
                'label_x': label_x,
                'label_y': label_y
            }


def get_mnist_svhn_dataloader(batch_size=128, train=True, root='./data', num_workers=4):
    """
    Get dataloader for MNIST-SVHN pairs.

    Args:
        batch_size: Batch size
        train: Train or test split
        root: Path to data folder
        num_workers: Number of data loading workers

    Returns:
        DataLoader yielding {'x': [B,1,32,32], 'y': [B,3,32,32], 'label': [B]}
    """
    dataset = MNISTSVHNDataset(root=root, train=train, download=True)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )

    return dataloader


def get_mnist_svhn_ratio_dataloader(batch_size=128, real_fake_ratio=0.5,
                                     train=True, root='./data', num_workers=4):
    """
    Get dataloader for training ratio estimator on MNIST-SVHN.

    Args:
        batch_size: Batch size
        real_fake_ratio: Proportion of real pairs
        train: Train or test split
        root: Path to data folder
        num_workers: Number of workers

    Returns:
        DataLoader yielding ratio training samples
    """
    base_dataset = MNISTSVHNDataset(root=root, train=train, download=True)
    ratio_dataset = MNISTSVHNRatioDataset(base_dataset, real_fake_ratio=real_fake_ratio)

    dataloader = DataLoader(
        ratio_dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )

    return dataloader


class SVHNFlowDataset(Dataset):
    """SVHN dataset wrapper returning dict format for flow training."""
    
    def __init__(self, root='./data', train=True, download=True):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        self.svhn = datasets.SVHN(
            root=root,
            split='train' if train else 'test',
            transform=self.transform,
            download=download
        )

    def __len__(self):
        return len(self.svhn)

    def __getitem__(self, idx):
        img, label = self.svhn[idx]
        return {
            'x': img,  # [3, 32, 32]
            'y': img,  # Same (for compatibility)
            'label': label
        }


def get_svhn_flow_dataloader(batch_size=128, train=True, root='./data', num_workers=4):
    """
    Get dataloader for training flow matching on SVHN only.

    Returns batches in same format as MNIST loader for compatibility.

    Args:
        batch_size: Batch size
        train: Train or test split
        root: Path to data folder
        num_workers: Number of workers

    Returns:
        DataLoader yielding {'x': [B,3,32,32], 'y': [B,3,32,32], 'label': [B]}
    """
    dataset = SVHNFlowDataset(root=root, train=train, download=True)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )

    return dataloader


class MNIST32FlowDataset(Dataset):
    """MNIST dataset resized to 32x32 for flow training."""
    
    def __init__(self, root='./data', train=True, download=True):
        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        self.mnist = datasets.MNIST(
            root=root,
            train=train,
            transform=self.transform,
            download=download
        )

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        img, label = self.mnist[idx]
        return {
            'x': img,  # [1, 32, 32]
            'label': label
        }


def get_mnist32_flow_dataloader(batch_size=128, train=True, root='./data', num_workers=4):
    """
    Get dataloader for training flow matching on MNIST 32x32.

    Args:
        batch_size: Batch size
        train: Train or test split
        root: Path to data folder
        num_workers: Number of workers

    Returns:
        DataLoader yielding {'x': [B,1,32,32], 'label': [B]}
    """
    dataset = MNIST32FlowDataset(root=root, train=train, download=True)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )

    return dataloader


if __name__ == '__main__':
    # Test the dataset
    print("Testing MNIST-SVHN dataset...")
    
    dataset = MNISTSVHNDataset(train=True)
    print(f"\nDataset size: {len(dataset)}")
    
    sample = dataset[0]
    print(f"MNIST shape: {sample['x'].shape}")  # [1, 32, 32]
    print(f"SVHN shape: {sample['y'].shape}")   # [3, 32, 32]
    print(f"Label: {sample['label']}")
    
    # Test dataloader
    dataloader = get_mnist_svhn_dataloader(batch_size=32)
    batch = next(iter(dataloader))
    print(f"\nBatch shapes:")
    print(f"  x: {batch['x'].shape}")  # [32, 1, 32, 32]
    print(f"  y: {batch['y'].shape}")  # [32, 3, 32, 32]
    print(f"  label: {batch['label'].shape}")
    
    print("\nTest passed!")
