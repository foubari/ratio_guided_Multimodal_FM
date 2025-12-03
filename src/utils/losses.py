"""
Density ratio estimation losses.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DensityRatioLoss(nn.Module):
    """Abstract base class for ratio estimation losses."""

    def forward(self, scores_real, scores_fake):
        """
        Compute loss from scores.

        Args:
            scores_real: [B_real] - T_θ(x,y) for real pairs
            scores_fake: [B_fake] - T_θ(x,y') for fake pairs

        Returns:
            loss: scalar tensor
            metrics: dict (for logging)
        """
        raise NotImplementedError


class DiscriminatorLoss(DensityRatioLoss):
    """
    Binary classification loss (logistic regression).

    Formula:
        D(z) = sigmoid(T(z))
        Loss = -E_q[log D] - E_p[log(1-D)]

    At optimum: D(z) / (1-D(z)) = q(z) / p(z) = r(z)

    Therefore: log_ratio(z) = log(D) - log(1-D) = T(z) - log(1+exp(T))
    """

    def forward(self, scores_real, scores_fake):
        """
        Args:
            scores_real: [B_real] - logits for real pairs
            scores_fake: [B_fake] - logits for fake pairs

        Returns:
            loss: scalar
            metrics: dict with 'loss', 'acc_real', 'acc_fake'
        """
        # BCE with logits (numerically stable)
        loss_real = F.binary_cross_entropy_with_logits(
            scores_real, torch.ones_like(scores_real)
        )
        loss_fake = F.binary_cross_entropy_with_logits(
            scores_fake, torch.zeros_like(scores_fake)
        )
        loss = loss_real + loss_fake

        # Metrics
        with torch.no_grad():
            acc_real = (scores_real > 0).float().mean().item()
            acc_fake = (scores_fake < 0).float().mean().item()

        return loss, {
            'loss': loss.item(),
            'acc_real': acc_real,
            'acc_fake': acc_fake
        }


class RuLSIFLoss(DensityRatioLoss):
    """
    Relative unconstrained Least-Squares Importance Fitting.

    Formula:
        w(z) = softplus(T(z))  # ratio estimate
        mixture: p_α = α*q + (1-α)*p

        Loss = 0.5 * E_{p_α}[w²] - E_q[w] + λ*(E_{p_α}[w] - 1)²

    Args:
        alpha: Mixture ratio (default=0.2)
        lambda_penalty: Constraint penalty (default=0.1)

    log_ratio = log(softplus(T))
    """

    def __init__(self, alpha=0.2, lambda_penalty=0.1):
        super().__init__()
        self.alpha = alpha
        self.lambda_penalty = lambda_penalty

    def forward(self, scores_real, scores_fake):
        """
        Args:
            scores_real: [B_real] - T(x,y) for real pairs
            scores_fake: [B_fake] - T(x,y') for fake pairs

        Returns:
            loss: scalar
            metrics: dict
        """
        # Compute weights
        w_real = F.softplus(scores_real)
        w_fake = F.softplus(scores_fake)

        # Mixture samples: combine real and fake
        # Approximation: treat concatenated samples as mixture
        w_mixture = torch.cat([w_real, w_fake])

        # Loss components
        # 1. 0.5 * E_{p_α}[w²]
        sq_term = 0.5 * w_mixture.pow(2).mean()

        # 2. - E_q[w]
        linear_term = -w_real.mean()

        # 3. λ*(E_{p_α}[w] - 1)²
        constraint = self.lambda_penalty * (w_mixture.mean() - 1.0).pow(2)

        loss = sq_term + linear_term + constraint

        # Metrics
        with torch.no_grad():
            mean_w_real = w_real.mean().item()
            mean_w_fake = w_fake.mean().item()

        return loss, {
            'loss': loss.item(),
            'mean_w_real': mean_w_real,
            'mean_w_fake': mean_w_fake,
            'constraint_term': constraint.item()
        }


def get_ratio_loss(loss_type='disc', **kwargs):
    """
    Factory function for ratio losses.

    Args:
        loss_type: 'disc', 'rulsif', ...
        **kwargs: Loss-specific hyperparameters

    Returns:
        DensityRatioLoss instance
    """
    if loss_type == 'disc':
        return DiscriminatorLoss()
    elif loss_type == 'rulsif':
        return RuLSIFLoss(
            alpha=kwargs.get('alpha', 0.2),
            lambda_penalty=kwargs.get('lambda_penalty', 0.1)
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
