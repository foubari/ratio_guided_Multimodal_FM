"""
Flexible Ratio Estimator for different image configurations.

Supports:
- Same modality: both x and y have same channels/size (e.g., MNIST transforms)
- Cross-modality: x and y have different channels (e.g., MNIST-SVHN)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageEncoder(nn.Module):
    """
    Convolutional encoder for images of various sizes.
    Uses adaptive pooling to handle different spatial sizes.
    """

    def __init__(self, in_channels=1, feature_dim=256):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, 32, 3, padding=1)
        self.gn1 = nn.GroupNorm(8, 32)
        self.pool1 = nn.MaxPool2d(2)  # /2

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.gn2 = nn.GroupNorm(8, 64)
        self.pool2 = nn.MaxPool2d(2)  # /4

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.gn3 = nn.GroupNorm(8, 128)
        self.pool3 = nn.MaxPool2d(2)  # /8

        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.gn4 = nn.GroupNorm(8, 128)

        # Adaptive pooling handles any spatial size
        self.pool_final = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Linear(128, feature_dim)

    def forward(self, img):
        """
        Args:
            img: [B, C, H, W] - any size image

        Returns:
            features: [B, feature_dim]
        """
        x = F.silu(self.gn1(self.conv1(img)))
        x = self.pool1(x)

        x = F.silu(self.gn2(self.conv2(x)))
        x = self.pool2(x)

        x = F.silu(self.gn3(self.conv3(x)))
        x = self.pool3(x)

        x = F.silu(self.gn4(self.conv4(x)))

        x = self.pool_final(x)
        x = x.view(x.size(0), -1)

        x = self.fc(x)

        return x


class FlexibleRatioEstimator(nn.Module):
    """
    Estimates log-ratio r_1(x,y) for paired modalities.

    Supports different channel counts for x and y modalities.

    Args:
        x_channels: Number of channels for x modality (default=1 for MNIST)
        y_channels: Number of channels for y modality (default=1 for MNIST, 3 for SVHN)
        feature_dim: Dimension of image features
        hidden_dim: Dimension of hidden layers
        loss_type: 'disc' or 'rulsif'
    """

    def __init__(
        self,
        x_channels=1,
        y_channels=1,
        feature_dim=256,
        hidden_dim=512,
        loss_type='disc'
    ):
        super().__init__()

        self.x_channels = x_channels
        self.y_channels = y_channels
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.loss_type = loss_type

        # Separate encoders for each modality
        self.encoder_x = ImageEncoder(in_channels=x_channels, feature_dim=feature_dim)
        self.encoder_y = ImageEncoder(in_channels=y_channels, feature_dim=feature_dim)

        # Score network
        self.score_net = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x, y):
        """
        Compute score T_θ(x, y).

        Args:
            x: [B, x_channels, H, W] - modality 1
            y: [B, y_channels, H, W] - modality 2

        Returns:
            scores: [B] - scalar scores
        """
        feat_x = self.encoder_x(x)
        feat_y = self.encoder_y(y)

        combined = torch.cat([feat_x, feat_y], dim=1)
        scores = self.score_net(combined).squeeze(-1)

        return scores

    def log_ratio(self, x, y):
        """
        Return log r̂_1(x,y) according to loss type.

        Args:
            x: [B, x_channels, H, W]
            y: [B, y_channels, H, W]

        Returns:
            [B] - log r̂_1(x,y)
        """
        scores = self.forward(x, y)

        if self.loss_type == 'disc':
            return F.logsigmoid(scores) - F.logsigmoid(-scores)
        elif self.loss_type == 'rulsif':
            w = F.softplus(scores)
            return torch.log(w + 1e-8)
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")


# Convenience classes

class RatioEstimatorMNIST(FlexibleRatioEstimator):
    """Ratio estimator for MNIST transforms (both x and y are 1x28x28)."""
    
    def __init__(self, loss_type='disc'):
        super().__init__(
            x_channels=1,
            y_channels=1,
            feature_dim=256,
            hidden_dim=512,
            loss_type=loss_type
        )


class RatioEstimatorMNISTSVHN_old(FlexibleRatioEstimator):
    """Ratio estimator for MNIST-SVHN (x=1x32x32, y=3x32x32). OLD VERSION ~944K params."""
    
    def __init__(self, loss_type='disc'):
        super().__init__(
            x_channels=1,
            y_channels=3,
            feature_dim=256,
            hidden_dim=512,
            loss_type=loss_type
        )


class MNISTEncoder(nn.Module):
    """
    Encoder for MNIST 32x32 grayscale images.
    Lighter encoder since MNIST is simpler.
    """
    
    def __init__(self, feature_dim=256):
        super().__init__()
        
        # 32x32 -> 16x16 -> 8x8 -> 4x4
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, feature_dim)
        
    def forward(self, x):
        """
        Args:
            x: [B, 1, 32, 32]
        Returns:
            [B, feature_dim]
        """
        h = F.silu(self.bn1(self.conv1(x)))
        h = F.max_pool2d(h, 2)  # 16x16
        
        h = F.silu(self.bn2(self.conv2(h)))
        h = F.max_pool2d(h, 2)  # 8x8
        
        h = F.silu(self.bn3(self.conv3(h)))
        h = F.max_pool2d(h, 2)  # 4x4
        
        h = F.silu(self.bn4(self.conv4(h)))
        
        h = self.pool(h)
        h = h.view(h.size(0), -1)
        h = self.fc(h)
        
        return h


class SVHNEncoder(nn.Module):
    """
    Encoder for SVHN 32x32 RGB images.
    More powerful encoder since SVHN is more complex (real photos with clutter).
    """
    
    def __init__(self, feature_dim=256):
        super().__init__()
        
        # Block 1: 32x32 -> 16x16
        self.conv1a = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1a = nn.BatchNorm2d(64)
        self.conv1b = nn.Conv2d(64, 64, 3, padding=1)
        self.bn1b = nn.BatchNorm2d(64)
        
        # Block 2: 16x16 -> 8x8
        self.conv2a = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2a = nn.BatchNorm2d(128)
        self.conv2b = nn.Conv2d(128, 128, 3, padding=1)
        self.bn2b = nn.BatchNorm2d(128)
        
        # Block 3: 8x8 -> 4x4
        self.conv3a = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3a = nn.BatchNorm2d(256)
        self.conv3b = nn.Conv2d(256, 256, 3, padding=1)
        self.bn3b = nn.BatchNorm2d(256)
        
        # Block 4: 4x4 -> 2x2
        self.conv4a = nn.Conv2d(256, 256, 3, padding=1)
        self.bn4a = nn.BatchNorm2d(256)
        self.conv4b = nn.Conv2d(256, 256, 3, padding=1)
        self.bn4b = nn.BatchNorm2d(256)
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, feature_dim)
        
    def forward(self, x):
        """
        Args:
            x: [B, 3, 32, 32]
        Returns:
            [B, feature_dim]
        """
        # Block 1
        h = F.silu(self.bn1a(self.conv1a(x)))
        h = F.silu(self.bn1b(self.conv1b(h)))
        h = F.max_pool2d(h, 2)  # 16x16
        
        # Block 2
        h = F.silu(self.bn2a(self.conv2a(h)))
        h = F.silu(self.bn2b(self.conv2b(h)))
        h = F.max_pool2d(h, 2)  # 8x8
        
        # Block 3
        h = F.silu(self.bn3a(self.conv3a(h)))
        h = F.silu(self.bn3b(self.conv3b(h)))
        h = F.max_pool2d(h, 2)  # 4x4
        
        # Block 4
        h = F.silu(self.bn4a(self.conv4a(h)))
        h = F.silu(self.bn4b(self.conv4b(h)))
        h = F.max_pool2d(h, 2)  # 2x2
        
        h = self.pool(h)
        h = h.view(h.size(0), -1)
        h = self.fc(h)
        
        return h


class RatioEstimatorMNISTSVHN(nn.Module):
    """
    Ratio estimator for MNIST-SVHN pairs with asymmetric encoders.
    
    - MNIST encoder: lighter (~150K params)
    - SVHN encoder: heavier (~1.4M params) 
    - Joint scoring network (~400K params)
    - Total: ~2M params
    
    Args:
        feature_dim: Output dimension for each encoder
        hidden_dim: Dimension of scoring network hidden layers
        loss_type: 'disc' or 'rulsif'
    """
    
    def __init__(self, feature_dim=256, hidden_dim=512, loss_type='disc'):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.loss_type = loss_type
        
        # Asymmetric encoders
        self.encoder_mnist = MNISTEncoder(feature_dim=feature_dim)
        self.encoder_svhn = SVHNEncoder(feature_dim=feature_dim)
        
        # Joint scoring network
        self.score_net = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, x, y):
        """
        Compute score T_θ(x, y).
        
        Args:
            x: [B, 1, 32, 32] - MNIST
            y: [B, 3, 32, 32] - SVHN
            
        Returns:
            scores: [B]
        """
        feat_mnist = self.encoder_mnist(x)
        feat_svhn = self.encoder_svhn(y)
        
        combined = torch.cat([feat_mnist, feat_svhn], dim=1)
        scores = self.score_net(combined).squeeze(-1)
        
        return scores
    
    def log_ratio(self, x, y):
        """
        Return log r̂_1(x,y) according to loss type.
        
        Args:
            x: [B, 1, 32, 32]
            y: [B, 3, 32, 32]
            
        Returns:
            [B] - log r̂_1(x,y)
        """
        scores = self.forward(x, y)
        
        if self.loss_type == 'disc':
            return F.logsigmoid(scores) - F.logsigmoid(-scores)
        elif self.loss_type == 'rulsif':
            w = F.softplus(scores)
            return torch.log(w + 1e-8)
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")


if __name__ == '__main__':
    print("Testing FlexibleRatioEstimator...\n")
    
    # Test MNIST (same channels)
    model_mnist = RatioEstimatorMNIST()
    x = torch.randn(4, 1, 28, 28)
    y = torch.randn(4, 1, 28, 28)
    scores = model_mnist(x, y)
    log_r = model_mnist.log_ratio(x, y)
    params = sum(p.numel() for p in model_mnist.parameters())
    print(f"MNIST: scores={scores.shape}, log_ratio={log_r.shape}, params={params:,}")
    
    # Test MNIST-SVHN (different channels) - NEW VERSION
    model_svhn = RatioEstimatorMNISTSVHN()
    x = torch.randn(4, 1, 32, 32)
    y = torch.randn(4, 3, 32, 32)
    scores = model_svhn(x, y)
    log_r = model_svhn.log_ratio(x, y)
    params = sum(p.numel() for p in model_svhn.parameters())
    print(f"MNIST-SVHN (new): scores={scores.shape}, log_ratio={log_r.shape}, params={params:,}")
    
    # Breakdown
    mnist_params = sum(p.numel() for p in model_svhn.encoder_mnist.parameters())
    svhn_params = sum(p.numel() for p in model_svhn.encoder_svhn.parameters())
    score_params = sum(p.numel() for p in model_svhn.score_net.parameters())
    print(f"  MNIST encoder: {mnist_params:,}")
    print(f"  SVHN encoder: {svhn_params:,}")
    print(f"  Score network: {score_params:,}")
    
    print("\nAll tests passed!")
