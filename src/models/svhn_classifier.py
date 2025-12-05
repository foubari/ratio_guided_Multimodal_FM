"""
SVHN Classifier for evaluation.

Used to evaluate coherence of generated MNIST-SVHN pairs.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SVHNClassifier(nn.Module):
    """
    CNN for SVHN classification (32x32 RGB).

    Architecture matches typical SVHN classifiers.
    """

    def __init__(self):
        super().__init__()

        # Conv layers
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)

        # After 4 poolings: 32 -> 16 -> 8 -> 4 -> 2
        # But we only do 2 poolings here: 32 -> 16 -> 8
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 10)

        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        """
        Args:
            x: [B, 3, 32, 32] - normalized to [-1, 1]

        Returns:
            logits: [B, 10]
        """
        # Conv block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)  # 32 -> 16

        # Conv block 2
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)  # 16 -> 8

        # Conv block 3
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Conv block 4
        x = F.relu(self.bn4(self.conv4(x)))

        # Flatten
        x = x.view(x.size(0), -1)

        # FC layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class MNISTClassifier32(nn.Module):
    """
    MNIST classifier for 32x32 images (resized from 28x28).
    
    Used to classify MNIST in MNIST-SVHN experiments.
    """

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)

        # 32 -> 16 -> 8
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        """
        Args:
            x: [B, 1, 32, 32] - normalized to [-1, 1]

        Returns:
            logits: [B, 10]
        """
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)  # 32 -> 16

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)  # 16 -> 8

        x = F.relu(self.conv3(x))

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


if __name__ == '__main__':
    print("Testing classifiers...\n")
    
    # Test SVHN classifier
    model_svhn = SVHNClassifier()
    x = torch.randn(4, 3, 32, 32)
    out = model_svhn(x)
    params = sum(p.numel() for p in model_svhn.parameters())
    print(f"SVHN Classifier: input={x.shape}, output={out.shape}, params={params:,}")
    
    # Test MNIST 32x32 classifier
    model_mnist = MNISTClassifier32()
    x = torch.randn(4, 1, 32, 32)
    out = model_mnist(x)
    params = sum(p.numel() for p in model_mnist.parameters())
    print(f"MNIST32 Classifier: input={x.shape}, output={out.shape}, params={params:,}")
    
    print("\nAll tests passed!")
