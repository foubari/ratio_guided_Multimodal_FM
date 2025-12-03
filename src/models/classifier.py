"""
Simple MNIST classifier for evaluation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MNISTClassifier(nn.Module):
    """
    Simple CNN for MNIST classification.

    Used to evaluate coherence of generated pairs:
    - Classify both x and y
    - Check if predictions match (coherence metric)
    """

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)

        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        """
        Args:
            x: [B, 1, 28, 28] - normalized to [-1, 1]

        Returns:
            logits: [B, 10]
        """
        # Conv layers
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)  # 28 -> 14

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)  # 14 -> 7

        # Flatten
        x = x.view(x.size(0), -1)

        # FC layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x
