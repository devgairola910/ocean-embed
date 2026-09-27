from typing import Dict, List, Optional, Tuple
import numpy as np
import xarray as xr

from app.core.constants import (
    GRID_LAT_MIN, GRID_LAT_MAX,
    GRID_LON_MIN, GRID_LON_MAX,
    GRID_RES,
)
from app.mock.generator import is_land


def regrid_surface_fields(
    ds: xr.Dataset,
    target_lats: Optional[List[float]] = None,
    target_lons: Optional[List[float]] = None,
    normalization_stats: Optional[Dict[str, Dict[str, float]]] = None,
) -> Tuple[np.ndarray, xr.Dataset]:
    """Regrid, mask, and normalize surface variables.

    Returns:
        batch_tensor: np.ndarray of shape (1, 5, H, W) for ML model input.
        ds_interp: xarray.Dataset containing regridded ocean fields.
    """
    if target_lats is None:
        target_lats = np.arange(GRID_LAT_MIN, GRID_LAT_MAX + 0.001, GRID_RES).tolist()
    if target_lons is None:
        target_lons = np.arange(GRID_LON_MIN, GRID_LON_MAX + 0.001, GRID_RES).tolist()

    ds_interp = ds.interp(lat=target_lats, lon=target_lons, method="linear")

    vars_list = ["sst", "sss", "sla", "u", "v"]
    channels = []

    H = len(target_lats)
    W = len(target_lons)

    land_mask = np.zeros((H, W), dtype=bool)
    for i, lat in enumerate(target_lats):
        for j, lon in enumerate(target_lons):
            land_mask[i, j] = is_land(lat, lon)

    for var in vars_list:
        if var in ds_interp:
            arr = ds_interp[var].values.copy()
        else:
            arr = np.zeros((H, W), dtype=np.float32)

        arr[land_mask] = np.nan

        if normalization_stats and var in normalization_stats:
            mean = normalization_stats[var].get("mean", 0.0)
            std = normalization_stats[var].get("std", 1.0)
            if std != 0:
                arr = (arr - mean) / std

        channels.append(arr)

    stacked = np.stack(channels, axis=0)
    batch_tensor = np.expand_dims(stacked, axis=0)

    return batch_tensor, ds_interp
