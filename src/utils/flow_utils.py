"""
Utilities for Conditional Flow Matching (CFM).
"""
import torch
import torch.nn.functional as F
from tqdm import tqdm


class CFMSchedule:
    """
    Manages interpolation and sampling for Conditional Flow Matching.

    Uses rectified flow (noise-free linear interpolation):
        x_t = (1-t) * x_0 + t * x_1
    where:
        x_0 ~ N(0, I) (prior)
        x_1 ~ q (data)

    The conditional velocity field is:
        u_t(x_t | x_0, x_1) = x_1 - x_0  (constant along trajectory)
    """

    def __init__(self):
        pass

    def add_noise(self, x_1, t):
        """
        Interpolate x_0 → x_1 at time t (rectified flow).

        Args:
            x_1: [B, C, H, W] - real data
            t: [B] - timesteps in [0, 1]

        Returns:
            x_t: [B, C, H, W] - interpolated sample
            u_t_target: [B, C, H, W] - true conditional velocity
        """
        B = x_1.shape[0]
        device = x_1.device

        # Sample prior
        x_0 = torch.randn_like(x_1)

        # Reshape t for broadcasting: [B] -> [B, 1, 1, 1]
        t = t.view(B, 1, 1, 1)

        # Linear interpolation (rectified flow, no noise)
        x_t = (1 - t) * x_0 + t * x_1

        # Conditional velocity (constant along the path)
        u_t_target = x_1 - x_0

        return x_t, u_t_target

    def sample(self, model, num_samples, num_steps=100, device='cuda'):
        """
        Sample from prior via ODE integration.

        Args:
            model: FlowMatchingModel
            num_samples: Number of images to generate
            num_steps: Number of integration steps

        Returns:
            samples: [num_samples, C, H, W]
        """
        model.eval()

        # Initialize from prior N(0, I)
        # Assume MNIST: [1, 28, 28]
        x_t = torch.randn(num_samples, 1, 28, 28, device=device)

        dt = 1.0 / num_steps

        with torch.no_grad():
            for step in range(num_steps):
                t = step * dt
                t_batch = torch.full((num_samples,), t, device=device)

                # Predict velocity
                v_t = model(x_t, t_batch)

                # Euler step
                x_t = x_t + v_t * dt

        return x_t


def train_flow_matching_epoch(model, dataloader, optimizer, schedule, device):
    """
    Train one epoch of CFM.

    For each batch (x_1):
        1. Sample t ~ U(0,1)
        2. Generate x_t, u_t_target = schedule.add_noise(x_1, t)
        3. Predict v_t = model(x_t, t)
        4. Loss = MSE(v_t, u_t_target)
        5. Backprop

    Args:
        model: FlowMatchingModel
        dataloader: DataLoader yielding {'x': ..., 'y': ...}
        optimizer: Torch optimizer
        schedule: CFMSchedule
        device: 'cuda' or 'cpu'

    Returns:
        avg_loss: float
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in tqdm(dataloader, desc="Training FM"):
        # Get data (use either 'x' or 'y' depending on modality)
        # For simplicity, we use the first available key
        if 'x' in batch:
            x_1 = batch['x'].to(device)
        elif 'y' in batch:
            x_1 = batch['y'].to(device)
        else:
            raise KeyError("Batch must contain 'x' or 'y'")

        B = x_1.shape[0]

        # Sample timesteps uniformly
        t = torch.rand(B, device=device)

        # Get interpolated samples and target velocities
        x_t, u_t_target = schedule.add_noise(x_1, t)

        # Predict velocity
        v_t = model(x_t, t)

        # MSE loss
        loss = F.mse_loss(v_t, u_t_target)

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches
    return avg_loss


def sample_bimodal_guided(
    fm_x,
    fm_y,
    ratio_estimator=None,
    guidance_method='none',
    guidance_strength=0.0,
    num_samples=16,
    num_steps=100,
    device='cuda'
):
    """
    Sample bimodal pairs (x_1, y_1) with optional guidance.

    Baseline (guidance_method='none'):
        dx_t/dt = u_t^x(x_t, t)
        dy_t/dt = u_t^y(y_t, t)
        → independent pairs

    Guided (guidance_method='grad_log_ratio'):
        dx_t/dt = u_t^x(x_t, t) + γ * ∂_x log r̂(x_t, y_t)
        dy_t/dt = u_t^y(y_t, t) + γ * ∂_y log r̂(x_t, y_t)
        → coherent pairs

    Args:
        fm_x: FlowMatchingModel for modality x
        fm_y: FlowMatchingModel for modality y
        ratio_estimator: RatioEstimator (None = baseline)
        guidance_method: 'none' or 'grad_log_ratio'
        guidance_strength: γ (guidance strength)
        num_samples: Number of pairs to generate
        num_steps: Number of ODE integration steps
        device: 'cuda' or 'cpu'

    Returns:
        samples_x: [num_samples, 1, 28, 28]
        samples_y: [num_samples, 1, 28, 28]
    """
    fm_x.eval()
    fm_y.eval()
    if ratio_estimator is not None:
        ratio_estimator.eval()

    # Initialize from prior N(0,I)
    x_t = torch.randn(num_samples, 1, 28, 28, device=device)
    y_t = torch.randn(num_samples, 1, 28, 28, device=device)

    dt = 1.0 / num_steps

    for step in tqdm(range(num_steps), desc="Sampling"):
        t = step * dt
        t_batch = torch.full((num_samples,), t, device=device)

        with torch.no_grad():
            # Velocity of independent flow
            v_x = fm_x(x_t, t_batch)
            v_y = fm_y(y_t, t_batch)

        # Guidance (if activated)
        if guidance_method == 'grad_log_ratio' and ratio_estimator is not None:
            # Guidance warmup: only after t > 0.3 for stability
            if t > 0.3:
                with torch.enable_grad():
                    x_t_grad = x_t.clone().requires_grad_(True)
                    y_t_grad = y_t.clone().requires_grad_(True)

                    # Log-ratio (terminal, not time-dependent)
                    log_r = ratio_estimator.log_ratio(x_t_grad, y_t_grad)
                    log_r_sum = log_r.sum()

                    # Gradients
                    grad_x, grad_y = torch.autograd.grad(
                        log_r_sum, [x_t_grad, y_t_grad]
                    )

                # Gradient clipping for stability
                grad_norm = torch.sqrt(grad_x.pow(2).sum() + grad_y.pow(2).sum())
                if grad_norm > 1.0:  # clip if norm > 1
                    grad_x = grad_x / grad_norm
                    grad_y = grad_y / grad_norm

                # Add guidance
                v_x = v_x + guidance_strength * grad_x.detach()
                v_y = v_y + guidance_strength * grad_y.detach()

        # Euler step
        x_t = x_t + v_x * dt
        y_t = y_t + v_y * dt

    return x_t, y_t
