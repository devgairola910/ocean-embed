"""Baseline models for scientific comparison against OceanEmbed.

Includes:
1. ClimatologyBaseline: Predicts 0 anomaly everywhere (climatology baseline).
2. DirectRegressionBaseline: Standard ConvNet architecture without self-supervised pretraining.
"""

from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class ClimatologyBaseline:
    """Naive baseline predicting mean climatology (zero anomaly)."""

    def __call__(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Produce zero anomaly and default 70m thermocline depth.

        Args:
            x: Input tensor of shape (B, T_lag, C, H, W)

        Returns:
            Tuple of:
              - pred_depth: (B, 15, H, W) filled with zeros
              - pred_thermo: (B, H, W) filled with 70.0 meters
        """
        B, _, _, H, W = x.shape
        device = x.device
        pred_depth = torch.zeros(B, 15, H, W, device=device, dtype=torch.float32)
        pred_thermo = torch.full((B, H, W), 70.0, device=device, dtype=torch.float32)
        return pred_depth, pred_thermo


class DirectRegressionBaseline(nn.Module):
    """Direct convolutional regression model trained end-to-end without self-supervised pretraining."""

    def __init__(
        self,
        in_channels: int = 35,  # T_lag * C = 5 * 7
        num_depths: int = 15,
        hidden_dim: int = 128
    ) -> None:
        """Initialize Direct CNN regression baseline."""
        super().__init__()
        self.in_channels = in_channels
        self.num_depths = num_depths

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )

        self.temp_head = nn.Conv2d(hidden_dim, num_depths, kernel_size=1)
        self.thermo_head = nn.Sequential(
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, T_lag, C, H, W)

        Returns:
            Tuple of:
              - pred_depth: (B, 15, H, W)
              - pred_thermo: (B, H, W)
        """
        B, T, C, H, W = x.shape
        x_flat = x.view(B, T * C, H, W)
        feat = self.encoder(x_flat)

        pred_depth = self.temp_head(feat)
        pred_thermo = self.thermo_head(feat).squeeze(1)
        return pred_depth, pred_thermo
