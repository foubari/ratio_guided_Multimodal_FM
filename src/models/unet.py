"""
Lightweight U-Net for Flow Matching on MNIST.

Based on torchcfm (https://github.com/atong01/conditional-flow-matching)
Architecture designed for 28x28 images with ~300K parameters.
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
    
    Args:
        in_channels: input channels
        out_channels: output channels  
        time_emb_dim: time embedding dimension
        dropout: dropout rate
    """
    
    def __init__(self, in_channels, out_channels, time_emb_dim, dropout=0.0):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # First conv
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        
        # Time embedding projection
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )
        
        # Second conv
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        # Skip connection
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()
    
    def forward(self, x, t_emb):
        """
        Args:
            x: [B, C, H, W] input
            t_emb: [B, time_emb_dim] time embedding
            
        Returns:
            [B, out_channels, H, W]
        """
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


class UNetMNIST(nn.Module):
    """
    Lightweight U-Net for MNIST Flow Matching.
    
    Architecture (for 28x28):
        - Encoder: 28 -> 14 -> 7
        - Middle block at 7x7
        - Decoder: 7 -> 14 -> 28
        - Skip connections between encoder and decoder
    
    Args:
        in_channels: input image channels (1 for MNIST)
        model_channels: base channel count (default=32)
        channel_mult: channel multipliers per level (default=(1, 2, 2))
        num_res_blocks: number of ResBlocks per level (default=2)
        dropout: dropout rate (default=0.0)
    """
    
    def __init__(
        self,
        in_channels=1,
        model_channels=32,
        channel_mult=(1, 2, 2),
        num_res_blocks=2,
        dropout=0.0
    ):
        super().__init__()
        
        self.in_channels = in_channels
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
        encoder_channels = [ch]  # Track channels for skip connections
        
        for level, mult in enumerate(channel_mult):
            out_ch = model_channels * mult
            
            # ResBlocks at this level
            for _ in range(num_res_blocks):
                self.encoder_blocks.append(ResBlock(ch, out_ch, time_emb_dim, dropout))
                ch = out_ch
                encoder_channels.append(ch)
            
            # Downsample (except at last level)
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
            
            # ResBlocks at this level (with skip connection from encoder)
            for i in range(num_res_blocks + 1):
                skip_ch = encoder_channels.pop()
                self.decoder_blocks.append(ResBlock(ch + skip_ch, out_ch, time_emb_dim, dropout))
                ch = out_ch
            
            # Upsample (except at first level)
            if level > 0:
                self.upsamplers.append(Upsample(ch))
        
        # Output
        self.out_norm = nn.GroupNorm(8, ch)
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
            # ResBlocks
            for _ in range(self.num_res_blocks):
                h = self.encoder_blocks[block_idx](h, t_emb)
                hs.append(h)
                block_idx += 1
            
            # Downsample
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
            # ResBlocks with skip connections
            for _ in range(self.num_res_blocks + 1):
                skip = hs.pop()
                h = torch.cat([h, skip], dim=1)
                h = self.decoder_blocks[block_idx](h, t_emb)
                block_idx += 1
            
            # Upsample
            if level > 0:
                h = self.upsamplers[upsample_idx](h)
                upsample_idx += 1
        
        # Output
        h = self.out_norm(h)
        h = F.silu(h)
        h = self.out_conv(h)
        
        return h


# Wrapper to match the original FlowMatchingModel interface
class FlowMatchingUNet(UNetMNIST):
    """
    Flow Matching model using U-Net architecture.
    Drop-in replacement for FlowMatchingModel with same interface.
    
    Default config: ~460K parameters (vs ~9M for the original)
    Based on torchcfm recommendations for MNIST.
    """
    
    def __init__(
        self,
        img_channels=1,
        model_channels=32,
        channel_mult=(1, 2),  # 2 levels: 28->14
        num_res_blocks=2,
        dropout=0.1
    ):
        super().__init__(
            in_channels=img_channels,
            model_channels=model_channels,
            channel_mult=channel_mult,
            num_res_blocks=num_res_blocks,
            dropout=dropout
        )


if __name__ == '__main__':
    # Test the model
    model = FlowMatchingUNet()
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Test forward pass
    x = torch.randn(4, 1, 28, 28)
    t = torch.rand(4)
    out = model(x, t)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    assert out.shape == x.shape, "Output shape mismatch!"
    print("Test passed!")
