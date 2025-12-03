"""
Trainer for ratio estimator.
"""
import torch
import numpy as np
from tqdm import tqdm


class RatioTrainer:
    """
    Trainer for ratio estimator.

    Args:
        model: RatioEstimator
        loss_fn: DensityRatioLoss
        optimizer: torch.optim optimizer
        device: 'cuda' or 'cpu'
    """

    def __init__(self, model, loss_fn, optimizer, device='cuda'):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device

    def train_step(self, batch):
        """
        One training step.

        Args:
            batch: dict with {'x': [B,1,28,28], 'y': [B,1,28,28], 'is_real': [B]}

        Returns:
            metrics: dict
        """
        x = batch['x'].to(self.device)
        y = batch['y'].to(self.device)
        is_real = batch['is_real'].to(self.device)

        # Forward
        scores = self.model(x, y)

        # Separate real/fake
        scores_real = scores[is_real == 1]
        scores_fake = scores[is_real == 0]

        # Loss
        loss, metrics = self.loss_fn(scores_real, scores_fake)

        # Backprop
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        return metrics

    def train_epoch(self, dataloader):
        """
        Train one complete epoch.

        Args:
            dataloader: DataLoader for ratio training

        Returns:
            avg_metrics: dict with averaged metrics
        """
        self.model.train()
        metrics_list = []

        for batch in tqdm(dataloader, desc="Training ratio"):
            metrics = self.train_step(batch)
            metrics_list.append(metrics)

        # Average metrics
        avg_metrics = {
            k: np.mean([m[k] for m in metrics_list])
            for k in metrics_list[0].keys()
        }

        return avg_metrics

    def evaluate(self, dataloader):
        """
        Evaluate on validation/test set.

        Args:
            dataloader: DataLoader

        Returns:
            avg_metrics: dict
        """
        self.model.eval()
        metrics_list = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                x = batch['x'].to(self.device)
                y = batch['y'].to(self.device)
                is_real = batch['is_real'].to(self.device)

                # Forward
                scores = self.model(x, y)

                # Separate real/fake
                scores_real = scores[is_real == 1]
                scores_fake = scores[is_real == 0]

                # Loss
                _, metrics = self.loss_fn(scores_real, scores_fake)
                metrics_list.append(metrics)

        # Average metrics
        avg_metrics = {
            k: np.mean([m[k] for m in metrics_list])
            for k in metrics_list[0].keys()
        }

        return avg_metrics
