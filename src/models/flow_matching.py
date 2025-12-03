"""
Flow Matching model for MNIST (Conditional Flow Matching - CFM).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal time embeddings."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        """
        Args:
            time: [B] - timesteps in [0, 1]

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
    """Convolutional encoder for 28x28 images."""

    def __init__(self, in_channels=1, feature_dim=256):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, 32, 3, stride=1, padding=1)
        self.gn1 = nn.GroupNorm(8, 32)

        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)  # 28 -> 14
        self.gn2 = nn.GroupNorm(8, 64)

        self.conv3 = nn.Conv2d(64, 128, 3, stride=2, padding=1)  # 14 -> 7
        self.gn3 = nn.GroupNorm(8, 128)

        self.conv4 = nn.Conv2d(128, 256, 3, stride=1, padding=1)  # 7 -> 7 (refine, no downsampling)
        self.gn4 = nn.GroupNorm(8, 256)

        # Final spatial size is 7x7 (28 -> 14 -> 7)
        self.fc = nn.Linear(256 * 7 * 7, feature_dim)

    def forward(self, x):
        """
        Args:
            x: [B, C, 28, 28]

        Returns:
            [B, feature_dim]
        """
        x = F.silu(self.gn1(self.conv1(x)))
        x = F.silu(self.gn2(self.conv2(x)))
        x = F.silu(self.gn3(self.conv3(x)))
        x = F.silu(self.gn4(self.conv4(x)))  # Stays at 7x7

        # Flatten
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class VelocityDecoder(nn.Module):
    """Decoder to reconstruct velocity field."""

    def __init__(self, feature_dim=256, time_emb_dim=128, out_channels=1):
        super().__init__()

        # Combine image features + time embeddings
        combined_dim = feature_dim + time_emb_dim

        # MLP to spatial features (start at 7x7 to match encoder output)
        self.fc1 = nn.Linear(combined_dim, 256 * 7 * 7)

        # Transposed convolutions: 7 -> 14 -> 28
        self.deconv1 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)  # 7 -> 14
        self.gn1 = nn.GroupNorm(8, 128)

        self.deconv2 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)  # 14 -> 28
        self.gn2 = nn.GroupNorm(8, 64)

        self.conv3 = nn.Conv2d(64, 32, 3, padding=1)  # 28 -> 28 (refine)
        self.gn3 = nn.GroupNorm(8, 32)

        # Final output layer
        self.conv_out = nn.Conv2d(32, out_channels, 3, padding=1)

    def forward(self, features, time_emb):
        """
        Args:
            features: [B, feature_dim]
            time_emb: [B, time_emb_dim]

        Returns:
            [B, out_channels, 28, 28] - velocity field
        """
        # Combine features and time
        x = torch.cat([features, time_emb], dim=1)

        # MLP to spatial (7x7)
        x = self.fc1(x)
        x = x.view(x.size(0), 256, 7, 7)

        # Deconvolutions: 7 -> 14 -> 28
        x = F.silu(self.gn1(self.deconv1(x)))
        x = F.silu(self.gn2(self.deconv2(x)))
        x = F.silu(self.gn3(self.conv3(x)))

        # Output
        x = self.conv_out(x)

        return x


class FlowMatchingModel(nn.Module):
    """
    Flow Matching unimodal model (Conditional Flow Matching - CFM).

    Args:
        img_channels: Number of image channels (1 for MNIST)
        feature_dim: Dimension of image features (default=256)
        time_emb_dim: Dimension of time embeddings (default=128)
    """

    def __init__(self, img_channels=1, feature_dim=256, time_emb_dim=128):
        super().__init__()

        self.img_channels = img_channels
        self.feature_dim = feature_dim
        self.time_emb_dim = time_emb_dim

        # Time embedding
        self.time_embed = SinusoidalPositionEmbeddings(time_emb_dim)

        # Image encoder
        self.encoder = ImageEncoder(img_channels, feature_dim)

        # Velocity decoder
        self.decoder = VelocityDecoder(feature_dim, time_emb_dim, img_channels)

    def forward(self, x_t, t):
        """
        Predict velocity field v_t.

        Args:
            x_t: [B, img_channels, 28, 28] - image at time t
            t: [B] - timesteps in [0, 1]

        Returns:
            v_t: [B, img_channels, 28, 28] - velocity field
        """
        # Time embedding
        t_emb = self.time_embed(t)

        # Encode image
        features = self.encoder(x_t)

        # Decode velocity
        v_t = self.decoder(features, t_emb)

        return v_t
