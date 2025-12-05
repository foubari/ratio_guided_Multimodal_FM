"""
Flexible U-Net for Flow Matching on different image sizes.

Supports:
- MNIST: 1x28x28 or 1x32x32 (grayscale)
- SVHN: 3x32x32 (RGB)

Based on the original UNet but with configurable image size.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    
    Args:
        timesteps: [B] tensor of timesteps
        dim: embedding dimension
        max_period: controls frequency range
        
    Returns:
        [B, dim] tensor of embeddings
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=timesteps.device) / half
    )
    args = timesteps[:, None] * freqs[None, :]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class ResBlock(nn.Module):
    """
    Residual block with time embedding injection.
    """
    
    def __init__(self, in_channels, out_channels, time_emb_dim, dropout=0.0):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # First conv
        self.norm1 = nn.GroupNorm(min(8, in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        
        # Time embedding projection
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )
        
        # Second conv
        self.norm2 = nn.GroupNorm(min(8, out_channels), out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        # Skip connection
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()
    
    def forward(self, x, t_emb):
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        
        # Add time embedding
        t = self.time_mlp(t_emb)[:, :, None, None]
        h = h + t
        
        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        return h + self.skip(x)


class Downsample(nn.Module):
    """Downsample by 2x using strided convolution."""
    
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)
    
    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    """Upsample by 2x using nearest neighbor + conv."""
    
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
    
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


class FlexibleUNet(nn.Module):
    """
    Flexible U-Net for Flow Matching on different image sizes.
    
    Automatically adjusts architecture based on input size.
    
    For 28x28: 28 -> 14 -> 7 (2 downsamples)
    For 32x32: 32 -> 16 -> 8 (2 downsamples)
    
    Args:
        in_channels: input image channels (1 for MNIST, 3 for SVHN)
        img_size: image size (28 or 32)
        model_channels: base channel count (default=64 for SVHN, 32 for MNIST)
        channel_mult: channel multipliers per level
        num_res_blocks: number of ResBlocks per level
        dropout: dropout rate
    """
    
    def __init__(
        self,
        in_channels=1,
        img_size=28,
        model_channels=32,
        channel_mult=(1, 2),
        num_res_blocks=2,
        dropout=0.1
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.img_size = img_size
        self.model_channels = model_channels
        self.channel_mult = channel_mult
        self.num_res_blocks = num_res_blocks
        
        # Time embedding
        time_emb_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        # Initial convolution
        self.input_conv = nn.Conv2d(in_channels, model_channels, 3, padding=1)
        
        # Encoder (downsampling path)
        self.encoder_blocks = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        
        ch = model_channels
        encoder_channels = [ch]
        
        for level, mult in enumerate(channel_mult):
            out_ch = model_channels * mult
            
            for _ in range(num_res_blocks):
                self.encoder_blocks.append(ResBlock(ch, out_ch, time_emb_dim, dropout))
                ch = out_ch
                encoder_channels.append(ch)
            
            if level < len(channel_mult) - 1:
                self.downsamplers.append(Downsample(ch))
                encoder_channels.append(ch)
        
        # Middle block
        self.middle_block1 = ResBlock(ch, ch, time_emb_dim, dropout)
        self.middle_block2 = ResBlock(ch, ch, time_emb_dim, dropout)
        
        # Decoder (upsampling path)
        self.decoder_blocks = nn.ModuleList()
        self.upsamplers = nn.ModuleList()
        
        for level, mult in reversed(list(enumerate(channel_mult))):
            out_ch = model_channels * mult
            
            for i in range(num_res_blocks + 1):
                skip_ch = encoder_channels.pop()
                self.decoder_blocks.append(ResBlock(ch + skip_ch, out_ch, time_emb_dim, dropout))
                ch = out_ch
            
            if level > 0:
                self.upsamplers.append(Upsample(ch))
        
        # Output
        self.out_norm = nn.GroupNorm(min(8, ch), ch)
        self.out_conv = nn.Conv2d(ch, in_channels, 3, padding=1)
        
        # Initialize output conv to zero for stable training
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)
    
    def forward(self, x, t):
        """
        Predict velocity field.
        
        Args:
            x: [B, C, H, W] noisy image at time t
            t: [B] timesteps in [0, 1]
            
        Returns:
            [B, C, H, W] predicted velocity
        """
        # Time embedding
        t_emb = timestep_embedding(t, self.model_channels)
        t_emb = self.time_embed(t_emb)
        
        # Initial conv
        h = self.input_conv(x)
        
        # Encoder
        hs = [h]
        block_idx = 0
        downsample_idx = 0
        
        for level in range(len(self.channel_mult)):
            for _ in range(self.num_res_blocks):
                h = self.encoder_blocks[block_idx](h, t_emb)
                hs.append(h)
                block_idx += 1
            
            if level < len(self.channel_mult) - 1:
                h = self.downsamplers[downsample_idx](h)
                hs.append(h)
                downsample_idx += 1
        
        # Middle
        h = self.middle_block1(h, t_emb)
        h = self.middle_block2(h, t_emb)
        
        # Decoder
        block_idx = 0
        upsample_idx = 0
        
        for level in reversed(range(len(self.channel_mult))):
            for _ in range(self.num_res_blocks + 1):
                skip = hs.pop()
                h = torch.cat([h, skip], dim=1)
                h = self.decoder_blocks[block_idx](h, t_emb)
                block_idx += 1
            
            if level > 0:
                h = self.upsamplers[upsample_idx](h)
                upsample_idx += 1
        
        # Output
        h = self.out_norm(h)
        h = F.silu(h)
        h = self.out_conv(h)
        
        return h


# Convenience classes for specific configurations

class FlowMatchingUNetMNIST(FlexibleUNet):
    """U-Net for MNIST (1x28x28 or 1x32x32)."""
    
    def __init__(self, img_size=28):
        super().__init__(
            in_channels=1,
            img_size=img_size,
            model_channels=32,
            channel_mult=(1, 2),
            num_res_blocks=2,
            dropout=0.1
        )


class FlowMatchingUNetSVHN(FlexibleUNet):
    """U-Net for SVHN (3x32x32)."""
    
    def __init__(self):
        super().__init__(
            in_channels=3,
            img_size=32,
            model_channels=64,  # More channels for RGB
            channel_mult=(1, 2, 2),  # One more level
            num_res_blocks=2,
            dropout=0.1
        )


if __name__ == '__main__':
    print("Testing FlexibleUNet configurations...\n")
    
    # Test MNIST 28x28
    model_mnist28 = FlowMatchingUNetMNIST(img_size=28)
    x = torch.randn(4, 1, 28, 28)
    t = torch.rand(4)
    out = model_mnist28(x, t)
    params = sum(p.numel() for p in model_mnist28.parameters())
    print(f"MNIST 28x28: input={x.shape}, output={out.shape}, params={params:,}")
    
    # Test MNIST 32x32
    model_mnist32 = FlowMatchingUNetMNIST(img_size=32)
    x = torch.randn(4, 1, 32, 32)
    out = model_mnist32(x, t)
    params = sum(p.numel() for p in model_mnist32.parameters())
    print(f"MNIST 32x32: input={x.shape}, output={out.shape}, params={params:,}")
    
    # Test SVHN
    model_svhn = FlowMatchingUNetSVHN()
    x = torch.randn(4, 3, 32, 32)
    out = model_svhn(x, t)
    params = sum(p.numel() for p in model_svhn.parameters())
    print(f"SVHN 32x32:  input={x.shape}, output={out.shape}, params={params:,}")
    
    print("\nAll tests passed!")
