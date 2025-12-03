"""
Checkpoint path management utilities.
"""
import os


def get_checkpoint_path(model_type, *args):
    """
    Get checkpoint path for a model.

    Args:
        model_type: 'flow', 'ratio', 'classifier'
        *args: Identifiers (modality, loss_type, epoch, etc.)

    Returns:
        path: str (e.g., 'checkpoints/flow_x_none_epoch50.pth')

    Examples:
        get_checkpoint_path('flow', 'x', 'none', 'best')
        → 'checkpoints/flow_x_none_best.pth'

        get_checkpoint_path('ratio', 'disc', 'rotate90', 30)
        → 'checkpoints/ratio_disc_rotate90_30.pth'
    """
    base_dir = 'checkpoints'
    os.makedirs(base_dir, exist_ok=True)

    # Filter out None values and convert to strings
    parts = [str(arg) for arg in args if arg is not None]

    filename = f"{model_type}_{'_'.join(parts)}.pth"
    return os.path.join(base_dir, filename)
