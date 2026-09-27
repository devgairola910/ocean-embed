"""Physics-Aware Differentiable Loss and Oceanographic Stability Constraints.

Follows architecture.md & design.md:
- L_total = w_temp * L_temp + w_thermo * L_thermo + w_stability * L_stability
- L_stability penalizes gravitational instability / non-monotonic density & temperature inversions.
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def compute_stability_penalty(
    pred_temp: torch.Tensor,
    land_mask: Optional[torch.Tensor] = None,
    allowed_tolerance: float = 0.05
) -> torch.Tensor:
    """Compute differentiable hinge penalty for non-monotonic temperature profiles.

    In the ocean, deeper layers are denser and colder than upper layers.
    A predicted deeper layer warmer than its shallower layer (T[k+1] > T[k])
    indicates an unphysical temperature inversion unless compensated by salinity.

    Args:
        pred_temp: Predicted temperature array of shape (B, D, H, W)
        land_mask: Boolean mask of shape (H, W) or (B, H, W) where True is ocean
        allowed_tolerance: Small inversion threshold in °C allowed before penalty kicks in.

    Returns:
        Scalar differentiable loss tensor.
    """
    # Vertical difference between adjacent depth levels: T(z_{k+1}) - T(z_k)
    # Expected to be <= 0 in stable stratification
    delta_T = pred_temp[:, 1:, :, :] - pred_temp[:, :-1, :, :]  # (B, D-1, H, W)

    # Inversions occur when delta_T > allowed_tolerance
    inversion_magnitude = F.relu(delta_T - allowed_tolerance) ** 2  # (B, D-1, H, W)

    if land_mask is not None:
        if land_mask.ndim == 2:
            mask = land_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        elif land_mask.ndim == 3:
            mask = land_mask.unsqueeze(1)               # (B, 1, H, W)
        else:
            mask = land_mask
        inversion_magnitude = inversion_magnitude * mask.float()
        total_pixels = mask.float().sum() * delta_T.shape[1] + 1e-6
        return inversion_magnitude.sum() / total_pixels

    return inversion_magnitude.mean()


def verify_profile_monotonicity(profiles: np.ndarray, tolerance: float = 0.1) -> float:
    """Calculate the percentage of physically valid (monotonic) profiles in a test set.

    Args:
        profiles: Temperature profiles array of shape (N, D) in °C
        tolerance: Inversion tolerance in °C

    Returns:
        Validity percentage (0.0 to 100.0%)
    """
    if profiles.ndim == 1:
        profiles = profiles.reshape(1, -1)
    # Check if delta T <= tolerance across all depth intervals
    diffs = np.diff(profiles, axis=1)  # (N, D-1)
    is_valid = np.all(diffs <= tolerance, axis=1)
    return float(np.mean(is_valid) * 100.0)


class OceanPhysicsLoss(nn.Module):
    """Combined multi-objective loss function with physics stability constraints."""

    def __init__(
        self,
        w_temp: float = 1.0,
        w_thermo: float = 0.2,
        w_stability: float = 0.1,
        loss_type: str = "mse"
    ) -> None:
        """Initialize loss function.

        Args:
            w_temp: Weight for 15-depth temperature anomaly loss (default: 1.0)
            w_thermo: Weight for auxiliary thermocline depth loss (default: 0.2)
            w_stability: Weight for differentiable physical stability penalty (default: 0.1)
            loss_type: 'mse' or 'l1'
        """
        super().__init__()
        self.w_temp = w_temp
        self.w_thermo = w_thermo
        self.w_stability = w_stability
        self.loss_type = loss_type

    def forward(
        self,
        pred_temp: torch.Tensor,             # (B, 15, H, W)
        target_temp: torch.Tensor,           # (B, 15, H, W)
        pred_thermo: torch.Tensor,           # (B, H, W)
        target_thermo: torch.Tensor,         # (B, H, W)
        land_mask: torch.Tensor              # (H, W) or (B, H, W)
    ) -> Dict[str, torch.Tensor]:
        """Compute all loss components and combined scalar loss.

        Returns:
            Dict containing:
              - 'loss_total': Combined weighted scalar loss
              - 'loss_temp': Temperature reconstruction loss
              - 'loss_thermo': Thermocline depth loss
              - 'loss_stability': Physical stability penalty
        """
        if land_mask.ndim == 2:
            mask_4d = land_mask.unsqueeze(0).unsqueeze(0).float()  # (1, 1, H, W)
            mask_3d = land_mask.unsqueeze(0).float()               # (1, H, W)
        else:
            mask_4d = land_mask.unsqueeze(1).float()
            mask_3d = land_mask.float()

        # 1. Temperature anomaly loss
        if self.loss_type == "mse":
            temp_diff = (pred_temp - target_temp) ** 2
        else:
            temp_diff = torch.abs(pred_temp - target_temp)
        masked_temp_diff = temp_diff * mask_4d
        loss_temp = masked_temp_diff.sum() / (mask_4d.sum() * pred_temp.shape[1] + 1e-6)

        # 2. Thermocline depth loss
        thermo_diff = (pred_thermo - target_thermo) ** 2
        masked_thermo_diff = thermo_diff * mask_3d
        loss_thermo = masked_thermo_diff.sum() / (mask_3d.sum() + 1e-6)

        # 3. Physics Stability Penalty
        loss_stability = compute_stability_penalty(pred_temp, land_mask=land_mask)

        # Combined Total Loss
        loss_total = (
            self.w_temp * loss_temp +
            self.w_thermo * loss_thermo +
            self.w_stability * loss_stability
        )

        return {
            "loss_total": loss_total,
            "loss_temp": loss_temp,
            "loss_thermo": loss_thermo,
            "loss_stability": loss_stability
        }
