"""Unit tests for physics-aware stability loss and monotonicity verification."""

import pytest
import numpy as np
import torch
from src.models.physics_loss import OceanPhysicsLoss, compute_stability_penalty, verify_profile_monotonicity


def test_stability_penalty_zero_for_monotonic_profile():
    # Stable profile: temperature decreases strictly with depth
    # Depth levels: 15 (e.g. 28°C down to 4°C)
    stable_profile = torch.linspace(28.0, 4.0, 15).view(1, 15, 1, 1).repeat(2, 1, 10, 10)
    loss = compute_stability_penalty(stable_profile)
    assert loss.item() == 0.0


def test_stability_penalty_positive_for_inversions():
    # Unstable profile: shallow water is 10°C, deep water is 25°C (inverted!)
    unstable_profile = torch.linspace(10.0, 25.0, 15).view(1, 15, 1, 1).repeat(2, 1, 10, 10)
    unstable_profile.requires_grad_(True)
    loss = compute_stability_penalty(unstable_profile)

    assert loss.item() > 0.0
    # Test differentiability / backward pass
    loss.backward()
    assert unstable_profile.grad is not None
    assert torch.any(unstable_profile.grad != 0.0)


def test_combined_physics_loss_gradient_flow():
    loss_fn = OceanPhysicsLoss(w_temp=1.0, w_thermo=0.2, w_stability=0.1)

    pred_temp = torch.randn(2, 15, 16, 16, requires_grad=True)
    target_temp = torch.randn(2, 15, 16, 16)
    pred_thermo = torch.randn(2, 16, 16, requires_grad=True)
    target_thermo = torch.randn(2, 16, 16)
    land_mask = torch.ones(16, 16, dtype=torch.bool)

    losses = loss_fn(pred_temp, target_temp, pred_thermo, target_thermo, land_mask)
    assert "loss_total" in losses
    assert "loss_temp" in losses
    assert "loss_thermo" in losses
    assert "loss_stability" in losses

    losses["loss_total"].backward()
    assert pred_temp.grad is not None
    assert pred_thermo.grad is not None


def test_monotonicity_verifier():
    # 1. Monotonic profile (valid)
    valid_prof = np.linspace(28.0, 4.0, 15).reshape(1, 15)
    assert verify_profile_monotonicity(valid_prof) == 100.0

    # 2. Inverted profile (invalid)
    invalid_prof = np.linspace(4.0, 28.0, 15).reshape(1, 15)
    assert verify_profile_monotonicity(invalid_prof) == 0.0
