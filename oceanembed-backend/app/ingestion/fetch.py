from abc import ABC, abstractmethod
import os
import glob
from typing import Optional
import numpy as np
import xarray as xr

from app.core.constants import (
    GRID_LAT_MIN, GRID_LAT_MAX,
    GRID_LON_MIN, GRID_LON_MAX,
    GRID_RES
)


class SatelliteDataSource(ABC):
    @abstractmethod
    def fetch_surface_fields(self, date_str: str) -> xr.Dataset:
        """Fetch surface satellite fields for target date. Returns xarray.Dataset with sst, sss, sla, u, v."""
        pass


class LocalFileSatelliteDataSource(SatelliteDataSource):
    def __init__(self, raw_data_dir: str = "data/raw"):
        self.raw_data_dir = raw_data_dir

    def fetch_surface_fields(self, date_str: str) -> xr.Dataset:
        pattern = os.path.join(self.raw_data_dir, f"*{date_str}*.nc")
        matches = glob.glob(pattern)
        if not matches:
            pattern = os.path.join(self.raw_data_dir, "*.nc")
            matches = glob.glob(pattern)

        if matches:
            return xr.open_dataset(matches[0])

        lats = np.arange(GRID_LAT_MIN, GRID_LAT_MAX + 0.001, GRID_RES)
        lons = np.arange(GRID_LON_MIN, GRID_LON_MAX + 0.001, GRID_RES)

        mesh_lon, mesh_lat = np.meshgrid(lons, lats)

        sst = 28.0 + 2.0 * np.sin(mesh_lat * 0.1)
        sss = 34.5 + 0.5 * np.cos(mesh_lon * 0.05)
        sla = 0.05 * np.sin(mesh_lat * 0.2 + mesh_lon * 0.2)
        u = 0.1 * np.cos(mesh_lat * 0.1)
        v = -0.05 * np.sin(mesh_lon * 0.1)

        ds = xr.Dataset(
            data_vars={
                "sst": (["lat", "lon"], sst),
                "sss": (["lat", "lon"], sss),
                "sla": (["lat", "lon"], sla),
                "u": (["lat", "lon"], u),
                "v": (["lat", "lon"], v),
            },
            coords={
                "lat": lats,
                "lon": lons,
            },
            attrs={"description": "Synthetic surface fields", "date": date_str}
        )
        return ds


class CopernicusSatelliteDataSource(SatelliteDataSource):
    def fetch_surface_fields(self, date_str: str) -> xr.Dataset:
        local_fallback = LocalFileSatelliteDataSource()
        return local_fallback.fetch_surface_fields(date_str)
