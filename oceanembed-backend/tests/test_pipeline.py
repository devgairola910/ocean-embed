import os
import numpy as np
import xarray as xr
import pytest

from app.ingestion.fetch import LocalFileSatelliteDataSource
from app.ingestion.regrid import regrid_surface_fields
from app.core.inference import predict_subsurface
from app.ingestion.pipeline import run_ingestion_pipeline


def test_fetch_synthetic_data(tmp_path):
    source = LocalFileSatelliteDataSource(raw_data_dir=str(tmp_path))
    ds = source.fetch_surface_fields("2025-05-15")

    assert isinstance(ds, xr.Dataset)
    assert "sst" in ds.data_vars
    assert "sss" in ds.data_vars
    assert "sla" in ds.data_vars
    assert "u" in ds.data_vars
    assert "v" in ds.data_vars


def test_regrid_surface_fields(tmp_path):
    source = LocalFileSatelliteDataSource(raw_data_dir=str(tmp_path))
    ds = source.fetch_surface_fields("2025-05-15")

    norm_stats = {
        "sst": {"mean": 28.0, "std": 1.5},
        "sss": {"mean": 34.5, "std": 0.5},
    }

    tensor, ds_interp = regrid_surface_fields(ds, normalization_stats=norm_stats)

    assert tensor.ndim == 4
    assert tensor.shape[0] == 1
    assert tensor.shape[1] == 5


def test_predict_subsurface_fallback():
    dummy_input = np.random.rand(1, 5, 10, 10).astype(np.float32)
    lats = [10.0, 10.25, 10.5]
    lons = [65.0, 65.25, 65.5]

    output_3d = predict_subsurface(dummy_input, lats, lons, "2025-05-15")

    assert output_3d.shape == (15, 3, 3)


def test_run_ingestion_pipeline(tmp_path):
    out_dir = tmp_path / "zarr_output"
    zarr_path = run_ingestion_pipeline(
        date_str="2025-05-15",
        data_source=LocalFileSatelliteDataSource(raw_data_dir=str(tmp_path)),
        output_dir=str(out_dir),
    )

    assert os.path.exists(zarr_path)

    ds_zarr = xr.open_zarr(zarr_path)
    assert "temp_depth_0m" in ds_zarr.data_vars
    assert "temp_depth_450m" in ds_zarr.data_vars
    assert len(ds_zarr.data_vars) == 15
