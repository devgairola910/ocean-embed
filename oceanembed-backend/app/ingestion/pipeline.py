import argparse
import os
import numpy as np
import xarray as xr

from app.config import get_settings
from app.core.constants import (
    DEPTH_LEVELS_M,
    GRID_LAT_MIN, GRID_LAT_MAX,
    GRID_LON_MIN, GRID_LON_MAX,
    GRID_RES,
)
from app.core.inference import predict_subsurface
from app.ingestion.fetch import LocalFileSatelliteDataSource, SatelliteDataSource
from app.ingestion.regrid import regrid_surface_fields


def run_ingestion_pipeline(
    date_str: str = "2025-05-15",
    data_source: SatelliteDataSource = None,
    output_dir: str = None,
) -> str:
    settings = get_settings()

    if data_source is None:
        data_source = LocalFileSatelliteDataSource()

    if output_dir is None:
        output_dir = settings.DATA_OUTPUT_PATH

    os.makedirs(output_dir, exist_ok=True)

    raw_ds = data_source.fetch_surface_fields(date_str)

    target_lats = [round(lat, 2) for lat in np.arange(GRID_LAT_MIN, GRID_LAT_MAX + 0.001, GRID_RES).tolist()]
    target_lons = [round(lon, 2) for lon in np.arange(GRID_LON_MIN, GRID_LON_MAX + 0.001, GRID_RES).tolist()]

    batch_tensor, _ = regrid_surface_fields(raw_ds, target_lats, target_lons)

    subsurface_3d = predict_subsurface(batch_tensor, target_lats, target_lons, date_str)

    data_vars = {}
    for i, d in enumerate(DEPTH_LEVELS_M):
        var_name = f"temp_depth_{int(d)}m"
        data_vars[var_name] = (["lat", "lon"], subsurface_3d[i])

    ds_out = xr.Dataset(
        data_vars=data_vars,
        coords={
            "lat": target_lats,
            "lon": target_lons,
        },
        attrs={
            "date": date_str,
            "depth_levels_m": list(DEPTH_LEVELS_M),
            "mode": settings.MODE,
        },
    )

    zarr_path = os.path.join(output_dir, f"oceanembed_{date_str}.zarr")
    ds_out.to_zarr(zarr_path, mode="w")

    print(f"Ingestion pipeline complete for date {date_str}. Written to {zarr_path}")
    return zarr_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run OceanEmbed Ingestion Pipeline")
    parser.add_argument("--date", type=str, default="2025-05-15", help="Target date YYYY-MM-DD")
    args = parser.parse_args()

    run_ingestion_pipeline(args.date)
