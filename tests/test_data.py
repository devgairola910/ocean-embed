"""Unit tests for grid, climatology, synthesizer, and dataset shape contracts."""

import pytest
import numpy as np
import torch
from src.data.grid import OceanGrid, STANDARD_DEPTH_LEVELS
from src.data.climatology import ClimatologyComputer
from src.data.synthetic import PhysicalOceanSynthesizer
from src.data.dataset import OceanEmbedDataset, create_dataloaders


def test_grid_initialization():
    grid = OceanGrid(lat_min=0.0, lat_max=10.0, lon_min=50.0, lon_max=70.0, resolution=0.5)
    assert grid.H == 21
    assert grid.W == 41
    assert grid.D == 15
    assert grid.land_mask.shape == (21, 41)


def test_climatology_and_anomaly_inversion():
    grid = OceanGrid(lat_min=0.0, lat_max=5.0, lon_min=60.0, lon_max=70.0, resolution=1.0)
    synth = PhysicalOceanSynthesizer(grid, seed=42)

    surf_list, depth_list = [], []
    doys = np.array([10, 50, 100, 150, 200])
    for d in doys:
        s, dep, _ = synth.generate_day(2020, int(d))
        surf_list.append(s)
        depth_list.append(dep)

    surf_arr = np.stack(surf_list, axis=0)
    depth_arr = np.stack(depth_list, axis=0)

    clim = ClimatologyComputer(num_days=366)
    clim.fit(surf_arr, depth_arr, doys)

    assert clim.is_fitted
    # Test anomaly computation and reconstruction
    test_doy = 100
    raw_depth = depth_arr[2]
    anom_depth = clim.compute_depth_anomaly(raw_depth, test_doy)
    reconstructed_depth = clim.reconstruct_absolute_temperature(anom_depth, test_doy)

    # Reconstructed should match original within numerical precision
    np.testing.assert_allclose(reconstructed_depth, raw_depth, rtol=1e-4, atol=1e-4)


def test_dataset_shape_contract():
    grid = OceanGrid(lat_min=0.0, lat_max=5.0, lon_min=60.0, lon_max=70.0, resolution=1.0)
    train_loader, val_loader, test_loader, _, _ = create_dataloaders(
        train_years=[2019, 2020],
        val_years=[2021],
        test_years=[2022],
        batch_size=2,
        lag_days=5,
        days_step=30,
        grid=grid
    )

    batch = next(iter(train_loader))
    assert "surface_input" in batch
    assert "target_depth" in batch
    assert "target_thermocline" in batch
    assert "land_mask" in batch

    # Shape contracts
    # surface_input: (B, T_lag=5, C=7, H, W)
    assert batch["surface_input"].shape == (2, 5, 7, grid.H, grid.W)
    # target_depth: (B, D=15, H, W)
    assert batch["target_depth"].shape == (2, 15, grid.H, grid.W)
    # target_thermocline: (B, H, W)
    assert batch["target_thermocline"].shape == (2, grid.H, grid.W)
    # land_mask: (B, H, W)
    assert batch["land_mask"].shape == (2, grid.H, grid.W)
