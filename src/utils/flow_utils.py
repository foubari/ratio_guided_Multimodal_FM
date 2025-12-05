"""
Utilities for Conditional Flow Matching (CFM).
"""
import torch
import torch.nn.functional as F
from tqdm import tqdm
import math


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

    def __init__(self, sigma=0.0):
        """
        Args:
            sigma: noise level (0 = deterministic OT path)
        """
        self.sigma = sigma

    def compute_mu_t(self, x_0, x_1, t):
        """Mean of the path: mu_t = (1-t)*x_0 + t*x_1"""
        t = t.view(-1, 1, 1, 1)
        return (1 - t) * x_0 + t * x_1

    def compute_sigma_t(self, t):
        """Standard deviation at time t (for noise-free path, this is just sigma)"""
        return self.sigma

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


def train_flow_matching_epoch(model, dataloader, optimizer, schedule, device, modality='x'):
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
        modality: 'x' or 'y' - which modality to train on

    Returns:
        avg_loss: float
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in tqdm(dataloader, desc=f"Training FM_{modality}"):
        # Get data for the specified modality
        x_1 = batch[modality].to(device)

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


def log_gaussian_density(x, mu, sigma):
    """
    Compute log probability under isotropic Gaussian N(mu, sigma^2 * I).
    
    Args:
        x: [B, ...] - samples
        mu: [B, ...] - means
        sigma: scalar - standard deviation
        
    Returns:
        log_prob: [B] - log probabilities
    """
    dim = x[0].numel()
    diff = (x - mu).view(x.shape[0], -1)  # [B, D]
    log_prob = -0.5 * (diff ** 2).sum(dim=1) / (sigma ** 2 + 1e-8)
    log_prob = log_prob - 0.5 * dim * math.log(2 * math.pi) - dim * math.log(sigma + 1e-8)
    return log_prob


def sample_bimodal_guided(
    fm_x,
    fm_y,
    ratio_estimator=None,
    guidance_method='none',
    guidance_strength=0.0,
    num_samples=16,
    num_steps=100,
    device='cuda',
    mc_batch_size=64,  # Number of MC samples per step
):
    """
    Sample bimodal pairs (x_1, y_1) with optional guidance.

    Baseline (guidance_method='none'):
        Independent sampling from FM_x and FM_y.

    Guided (guidance_method='mc_feng'):
        Monte Carlo guidance following Feng et al. Algorithm 2.
        Uses GENERATED samples from the flows (not dataset!) to estimate the guidance term.

    Args:
        fm_x: FlowMatchingModel for modality x
        fm_y: FlowMatchingModel for modality y
        ratio_estimator: RatioEstimator (required for guidance)
        guidance_method: 'none' or 'mc_feng'
        guidance_strength: γ (guidance strength)
        num_samples: Number of pairs to generate
        num_steps: Number of ODE integration steps
        device: 'cuda' or 'cpu'
        mc_batch_size: Number of MC samples for guidance estimation

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
    eps = 1e-3  # Small epsilon to avoid numerical issues
    
    # Pre-load MC samples if using guidance
    mc_x1_samples = None
    mc_y1_samples = None
    mc_ratios = None
    
    if guidance_method == 'mc_feng' and ratio_estimator is not None:
        print(f"  Generating {mc_batch_size} independent MC samples from flows...")
        
        # Generate x samples from FM_x
        mc_x1_samples = torch.randn(mc_batch_size, 1, 28, 28, device=device)
        for s in range(num_steps):
            t_mc = s * dt
            t_mc_batch = torch.full((mc_batch_size,), t_mc, device=device)
            with torch.no_grad():
                v = fm_x(mc_x1_samples, t_mc_batch)
            mc_x1_samples = mc_x1_samples + v * dt
        
        # Generate y samples from FM_y
        mc_y1_samples = torch.randn(mc_batch_size, 1, 28, 28, device=device)
        for s in range(num_steps):
            t_mc = s * dt
            t_mc_batch = torch.full((mc_batch_size,), t_mc, device=device)
            with torch.no_grad():
                v = fm_y(mc_y1_samples, t_mc_batch)
            mc_y1_samples = mc_y1_samples + v * dt
        
        print(f"  Generated MC samples: x shape={mc_x1_samples.shape}, y shape={mc_y1_samples.shape}")
        
        # Pre-compute ratios r_1(x_1^(i), y_1^(i)) for all MC samples
        with torch.no_grad():
            log_r = ratio_estimator.log_ratio(mc_x1_samples, mc_y1_samples)  # [N_mc]
            mc_ratios = log_r.exp()  # r_1 = exp(log_ratio) = q(x,y) / p_ind(x,y)
            print(f"  MC ratios: min={mc_ratios.min():.4f}, max={mc_ratios.max():.4f}, mean={mc_ratios.mean():.4f}")
    
    # Diagnostics
    guidance_printed = False
    
    for step in tqdm(range(num_steps), desc="Sampling"):
        t = step * dt
        t_batch = torch.full((num_samples,), t, device=device)

        with torch.no_grad():
            # Velocity of independent flow
            v_x = fm_x(x_t, t_batch)
            v_y = fm_y(y_t, t_batch)

        # MC Guidance following Feng et al. Algorithm 2
        if guidance_method == 'mc_feng' and mc_x1_samples is not None and t > eps:
            with torch.no_grad():
                N_mc = mc_x1_samples.shape[0]
                
                # Path parameters for rectified flow: x_t = (1-t)*x_0 + t*x_1
                # Since x_0 ~ N(0,I), we have: x_t | x_1 ~ N(t*x_1, (1-t)^2 * I)
                sigma_t = (1 - t + eps)
                
                # For each current sample (x_t[b], y_t[b]) and each MC sample (x_1^(i), y_1^(i)):
                # Compute p_t(x_t[b] | x_1^(i)) * p_t(y_t[b] | y_1^(i))
                
                # Expand for broadcasting: [B, 1, ...] vs [1, N_mc, ...]
                x_t_exp = x_t.unsqueeze(1)  # [B, 1, 1, 28, 28]
                y_t_exp = y_t.unsqueeze(1)  # [B, 1, 1, 28, 28]
                mc_x1_exp = mc_x1_samples.unsqueeze(0)  # [1, N_mc, 1, 28, 28]
                mc_y1_exp = mc_y1_samples.unsqueeze(0)  # [1, N_mc, 1, 28, 28]
                
                # Mean of conditional: mu_t = t * x_1
                mu_x = t * mc_x1_exp  # [1, N_mc, 1, 28, 28]
                mu_y = t * mc_y1_exp  # [1, N_mc, 1, 28, 28]
                
                # Log probability: log p_t(x_t | x_1^(i))
                # [B, N_mc]
                diff_x = (x_t_exp - mu_x).view(num_samples, N_mc, -1)  # [B, N_mc, D]
                diff_y = (y_t_exp - mu_y).view(num_samples, N_mc, -1)  # [B, N_mc, D]
                D = diff_x.shape[-1]
                
                log_p_x = -0.5 * (diff_x ** 2).sum(dim=-1) / (sigma_t ** 2)  # [B, N_mc]
                log_p_y = -0.5 * (diff_y ** 2).sum(dim=-1) / (sigma_t ** 2)  # [B, N_mc]
                log_p_joint = log_p_x + log_p_y  # [B, N_mc]
                
                # Normalize for numerical stability
                log_p_max = log_p_joint.max(dim=1, keepdim=True)[0]  # [B, 1]
                p_joint = (log_p_joint - log_p_max).exp()  # [B, N_mc]
                
                # Compute tilde_p_t = (1/N) * sum_i p_t^(i)
                p_bar = p_joint.mean(dim=1, keepdim=True) + 1e-10  # [B, 1]
                
                # Compute tilde_Z_t = (1/N) * sum_i exp(-J_i) * p_t^(i)
                #                   = (1/N) * sum_i r_1(x_1^(i), y_1^(i)) * p_t^(i)
                # [N_mc] -> [1, N_mc]
                mc_ratios_exp = mc_ratios.unsqueeze(0)  # [1, N_mc]
                Z_bar = (mc_ratios_exp * p_joint).mean(dim=1, keepdim=True) + 1e-10  # [B, 1]
                
                # Weights: w_i = (r_1^(i) / Z_bar) * (p_t^(i) / p_bar)
                weights = (mc_ratios_exp / Z_bar) * (p_joint / p_bar)  # [B, N_mc]
                weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-10)  # Normalize
                
                # Conditional velocity: v_{t|x_1^(i)}(x_t | x_1^(i)) = x_1^(i) - x_0
                # For rectified flow with x_0 recovered: x_0 = (x_t - t*x_1) / (1-t)
                # So v = x_1 - x_0 = x_1 - (x_t - t*x_1)/(1-t) = (x_1 - x_t)/(1-t) + t*x_1/(1-t)
                #      = (x_1*(1-t+t) - x_t) / (1-t) = (x_1 - x_t) / (1-t)
                # But actually for guidance, we use v_{t|x1} = (x_1 - x_t) / (1 - t) which is the direction to x_1
                
                # Guidance is the weighted sum of conditional velocities minus baseline
                # g_t = sum_i w_i * v_{t|x1^(i)} - v_t
                # But actually following Feng more closely:
                # g_t = sum_i w_i * v_{t|z^(i)} where z^(i) = (x_1^(i), y_1^(i))
                
                # v_{t|x1}(x_t) for x: direction toward x_1^(i)
                v_cond_x = (mc_x1_exp.squeeze(0) - x_t.unsqueeze(1)) / (1 - t + eps)  # [B, N_mc, 1, 28, 28]
                v_cond_y = (mc_y1_exp.squeeze(0) - y_t.unsqueeze(1)) / (1 - t + eps)  # [B, N_mc, 1, 28, 28]
                
                # Weighted sum of conditional velocities
                # weights: [B, N_mc] -> [B, N_mc, 1, 1, 1]
                weights_exp = weights.view(num_samples, N_mc, 1, 1, 1)
                
                g_x = (weights_exp * v_cond_x).sum(dim=1)  # [B, 1, 28, 28]
                g_y = (weights_exp * v_cond_y).sum(dim=1)  # [B, 1, 28, 28]
                
                # The guidance is g_t (the reweighted velocity), not added to v_t
                # Following Feng: v_guided = v_ind + guidance_strength * (g_t - v_ind)
                # Or equivalently: v_guided = (1 - gamma) * v_ind + gamma * g_t for gamma in [0,1]
                # But with gamma as strength: v_guided = v_ind + gamma * (g_t - v_ind)
                
                # Diagnostics
                if not guidance_printed and step == int(0.3 * num_steps):
                    g_x_norm = g_x.view(num_samples, -1).norm(dim=1).mean().item()
                    g_y_norm = g_y.view(num_samples, -1).norm(dim=1).mean().item()
                    v_x_norm = v_x.view(num_samples, -1).norm(dim=1).mean().item()
                    v_y_norm = v_y.view(num_samples, -1).norm(dim=1).mean().item()
                    w_max = weights.max().item()
                    w_min = weights.min().item()
                    
                    print(f"\n[MC Guidance Diagnostics at t={t:.2f}]")
                    print(f"  sigma_t={sigma_t:.4f}")
                    print(f"  ||v_x||={v_x_norm:.4f}, ||v_y||={v_y_norm:.4f}")
                    print(f"  ||g_x||={g_x_norm:.4f}, ||g_y||={g_y_norm:.4f}")
                    print(f"  weights: min={w_min:.6f}, max={w_max:.6f}")
                    print(f"  Z_bar: {Z_bar.mean().item():.4f}")
                    guidance_printed = True
                
                # Apply guidance: interpolate between independent and guided
                # gamma = 0: pure independent
                # gamma = 1: pure MC guidance
                v_x = (1 - guidance_strength) * v_x + guidance_strength * g_x
                v_y = (1 - guidance_strength) * v_y + guidance_strength * g_y

        # Euler step
        x_t = x_t + v_x * dt
        y_t = y_t + v_y * dt

    return x_t, y_t
