"""
Ratio estimator for bimodal flow matching.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal time embeddings (reused from flow_matching)."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        """
        Args:
            time: [B] - timesteps

        Returns:
            [B, dim] - time embeddings
        """
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ImageEncoder(nn.Module):
    """
    Convolutional encoder for 28x28 images.

    Architecture:
        Conv2d layers with GroupNorm + SiLU
        Global pooling + Linear projection
    """

    def __init__(self, in_channels=1, feature_dim=256):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, 32, 3, padding=1)
        self.gn1 = nn.GroupNorm(8, 32)
        self.pool1 = nn.MaxPool2d(2)  # 28 -> 14

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.gn2 = nn.GroupNorm(8, 64)
        self.pool2 = nn.MaxPool2d(2)  # 14 -> 7

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.gn3 = nn.GroupNorm(8, 128)
        self.pool3 = nn.MaxPool2d(2)  # 7 -> 3

        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.gn4 = nn.GroupNorm(8, 128)

        # Adaptive pooling to handle variable spatial sizes
        self.pool_final = nn.AdaptiveAvgPool2d(1)

        # Project to feature_dim
        self.fc = nn.Linear(128, feature_dim)

    def forward(self, img):
        """
        Args:
            img: [B, in_channels, 28, 28]

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

        # Global pooling
        x = self.pool_final(x)
        x = x.view(x.size(0), -1)

        # Project to feature dim
        x = self.fc(x)

        return x


class RatioEstimator(nn.Module):
    """
    Estimates the log-ratio r_1(x,y) = log(q(x,y) / p_1^ind(x,y)).

    Simplification: Estimates only the TERMINAL ratio (t=1), not time-dependent.

    Architecture:
        encoder_x : ImageEncoder for modality x
        encoder_y : ImageEncoder for modality y
        score_net : MLP([256+256] → 512 → 256 → 1)

    Args:
        feature_dim: Dimension of image features (default=256)
        hidden_dim: Dimension of hidden layers (default=512)
        loss_type: 'disc' or 'rulsif' (determines how to interpret T)
    """

    def __init__(self, feature_dim=256, hidden_dim=512, loss_type='disc'):
        super().__init__()

        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.loss_type = loss_type

        # Two-stream encoders
        self.encoder_x = ImageEncoder(in_channels=1, feature_dim=feature_dim)
        self.encoder_y = ImageEncoder(in_channels=1, feature_dim=feature_dim)

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
            x: [B, 1, 28, 28] - modality 1
            y: [B, 1, 28, 28] - modality 2

        Returns:
            scores: [B] - scalar scores T_θ(x, y)
        """
        # Encode both images
        feat_x = self.encoder_x(x)
        feat_y = self.encoder_y(y)

        # Concatenate features
        combined = torch.cat([feat_x, feat_y], dim=1)

        # Score network
        scores = self.score_net(combined).squeeze(-1)

        return scores

    def log_ratio(self, x, y):
        """
        Return log r̂_1(x,y) according to loss type.

        For discriminator:
            T(x,y) = logit of discriminator
            log_ratio = F.logsigmoid(T) - F.logsigmoid(-T)
                      = log(D/(1-D)) but numerically stable

        For rulsif:
            w(x,y) = softplus(T(x,y))
            log_ratio = log(w) = log(softplus(T))

        Args:
            x: [B, 1, 28, 28]
            y: [B, 1, 28, 28]

        Returns:
            [B] - log r̂_1(x,y)
        """
        scores = self.forward(x, y)

        if self.loss_type == 'disc':
            # log(σ(T) / (1-σ(T))) = T - log(1+exp(T))
            # More stable: logsigmoid(T) - logsigmoid(-T)
            return F.logsigmoid(scores) - F.logsigmoid(-scores)
        elif self.loss_type == 'rulsif':
            # log(softplus(T))
            w = F.softplus(scores)
            return torch.log(w + 1e-8)  # epsilon for stability
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")
