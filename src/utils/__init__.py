"""Utilities module."""
import random
import numpy as np
import torch


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed (default: 42)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # For fully deterministic results (may reduce performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_checkpoint(model, path, device='cpu'):
    """
    Load model checkpoint, handling both old and new formats.
    
    Old format: just state_dict
    New format: {'model_state_dict': ..., 'optimizer_state_dict': ..., 'epoch': ..., 'best_loss': ...}
    
    Args:
        model: PyTorch model to load weights into
        path: Path to checkpoint file
        device: Device to load to
        
    Returns:
        dict with 'epoch' and 'best_loss' if available, else empty dict
    """
    checkpoint = torch.load(path, map_location=device)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        return {
            'epoch': checkpoint.get('epoch', 0),
            'best_loss': checkpoint.get('best_loss', float('inf'))
        }
    else:
        # Old format
        model.load_state_dict(checkpoint)
        return {}

