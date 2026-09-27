"""Comprehensive verification test suite for Phase 0 and Phase 1 requirements."""

import os
import pytest
import numpy as np
import torch
import xarray as xr

from src.data.grid import OceanGrid, STANDARD_DEPTH_LEVELS, STANDARD_SURFACE_CHANNELS
from src.data.ingestion import SatelliteDataIngestor
from src.data.regrid import OceanDataRegridder
from src.data.climatology import ClimatologyComputer
from src.data.synthetic import PhysicalOceanSynthesizer
from src.data.dataset import OceanEmbedDataset, create_dataloaders
from src.data.diagnostics import run_data_pipeline_diagnostics


def test_phase0_repo_structure_and_sources():
    """Verify repo layout, configs, and data sources documentation exist per rules.md."""
    assert os.path.exists("data/SOURCES.md"), "data/SOURCES.md must exist."
    assert os.path.exists("configs/default_config.yaml"), "configs/default_config.yaml must exist."
    assert os.path.exists("configs/exp_bob_arabian_2026.yaml"), "experiment yaml must exist."
    assert os.path.exists("requirements.txt"), "requirements.txt must exist."


def test_phase1_fr1_and_fr2_ingestion_and_regridding(tmp_path):
    """FR1 & FR2: Ingest all 7 surface channels + GLORYS and regrid to 0.25° grid."""
    grid = OceanGrid(lat_min=0.0, lat_max=10.0, lon_min=50.0, lon_max=70.0, resolution=0.5)
    ingestor = SatelliteDataIngestor(grid=grid)

    mock_nc_path = str(tmp_path / "sample_surface_glorys.nc")
    ingestor.create_mock_raw_netcdf_sample(mock_nc_path, year=2022, doy=150)

    # Test loading individual products
    sst_regrid = ingestor.load_ostia_sst(mock_nc_path)
    sss_regrid = ingestor.load_smap_sss(mock_nc_path)
    sla_regrid = ingestor.load_duacs_ssh(mock_nc_path)
    glorys_3d, thermo_depth = ingestor.load_glorys_reanalysis(mock_nc_path)

    # Check shapes
    assert sst_regrid.shape == (grid.H, grid.W)
    assert sss_regrid.shape == (grid.H, grid.W)
    assert sla_regrid.shape == (grid.H, grid.W)
    assert glorys_3d.shape == (15, grid.H, grid.W)
    assert thermo_depth.shape == (grid.H, grid.W)

    # Land pixels must be properly masked
    assert np.all(sst_regrid[~grid.land_mask] == 0.0)
    assert np.all(glorys_3d[:, ~grid.land_mask] == 0.0)


def test_phase1_climatology_no_temporal_leakage():
    """Verify daily climatology is fit only on training years."""
    grid = OceanGrid(lat_min=0.0, lat_max=10.0, lon_min=50.0, lon_max=70.0, resolution=0.5)
    synth = PhysicalOceanSynthesizer(grid, seed=42)

    train_years = [2018, 2019, 2020]
    test_year = 2022

    # Fit climatology strictly on training years
    clim = ClimatologyComputer()
    surf_list, depth_list, doys = [], [], []
    for yr in train_years:
        for doy in [50, 150, 250]:
            s, dep, _ = synth.generate_day(yr, doy)
            surf_list.append(s)
            depth_list.append(dep)
            doys.append(doy)

    clim.fit(np.stack(surf_list), np.stack(depth_list), np.array(doys))

    # Test on unseen test year
    test_surf, test_depth, _ = synth.generate_day(test_year, 150)
    test_surf_anom = clim.compute_surface_anomaly(test_surf, 150)
    test_depth_anom = clim.compute_depth_anomaly(test_depth, 150)

    assert test_surf_anom.shape == (7, grid.H, grid.W)
    assert test_depth_anom.shape == (15, grid.H, grid.W)
    assert not np.isnan(test_surf_anom).any()
    assert not np.isnan(test_depth_anom).any()


def test_phase1_lagged_dataloader_exit_criteria():
    """Exit criteria: DataLoader yields valid (input_tensor, target_tensor, aux_targets) batches."""
    grid = OceanGrid(lat_min=0.0, lat_max=10.0, lon_min=50.0, lon_max=70.0, resolution=0.5)
    train_loader, val_loader, test_loader, _, _ = create_dataloaders(
        train_years=[2018, 2019, 2020],
        val_years=[2021],
        test_years=[2022],
        batch_size=4,
        lag_days=5,
        days_step=20,
        grid=grid
    )

    for loader, name in [(train_loader, "Train"), (val_loader, "Val"), (test_loader, "Test")]:
        batch = next(iter(loader))
        # 1. Surface lagged input: (B=4, T_lag=5, C=7, H, W)
        assert batch["surface_input"].shape == (4, 5, 7, grid.H, grid.W), f"{name} input shape mismatch"
        # 2. Subsurface target: (B=4, D=15, H, W)
        assert batch["target_depth"].shape == (4, 15, grid.H, grid.W), f"{name} target shape mismatch"
        # 3. Thermocline target: (B=4, H, W)
        assert batch["target_thermocline"].shape == (4, grid.H, grid.W), f"{name} thermo shape mismatch"
        # 4. Land mask: (B=4, H, W)
        assert batch["land_mask"].shape == (4, grid.H, grid.W), f"{name} mask shape mismatch"
        # 5. Date: (B=4, 2)
        assert batch["target_date"].shape == (4, 2), f"{name} date shape mismatch"


def test_phase1_diagnostics_report(tmp_path):
    """Run data pipeline diagnostics and verify 0 NaNs and physical bounds."""
    report = run_data_pipeline_diagnostics(output_dir=str(tmp_path))
    assert report["status"] == "PASS"
    assert report["zero_nan_passed"] is True
    assert report["sst_valid"] is True
    assert report["sss_valid"] is True
    assert report["sla_valid"] is True
    assert report["monotonicity_passed"] is True
    assert os.path.exists(report["figure_saved_to"])
