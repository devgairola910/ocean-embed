"""Data regridding, harmonization, and file ingestion utilities for OceanEmbed."""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
from src.data.grid import OceanGrid, STANDARD_DEPTH_LEVELS, STANDARD_SURFACE_CHANNELS


class OceanDataRegridder:
    """Harmonizes raw heterogeneous satellite and reanalysis fields onto the standard 0.25° grid."""

    def __init__(self, target_grid: Optional[OceanGrid] = None) -> None:
        """Initialize regridder with target grid.

        Args:
            target_grid: OceanGrid instance defining target resolution and boundaries.
        """
        self.grid = target_grid or OceanGrid()

    def regrid_2d_field(
        self,
        field: np.ndarray,
        src_lats: np.ndarray,
        src_lons: np.ndarray,
        method: str = "linear",
        fill_value: float = np.nan
    ) -> np.ndarray:
        """Regrid a 2D spatial slice (H_src, W_src) to target grid (H_tgt, W_tgt).

        Args:
            field: 2D array of shape (H_src, W_src)
            src_lats: Source 1D latitude coordinates (strictly monotonic)
            src_lons: Source 1D longitude coordinates (strictly monotonic)
            method: 'linear' or 'nearest'
            fill_value: Value for out-of-bounds coordinates.

        Returns:
            Regridded 2D array of shape (H_tgt, W_tgt)
        """
        # Ensure coordinates are sorted ascending
        lat_order = np.argsort(src_lats)
        lon_order = np.argsort(src_lons)
        sorted_lats = src_lats[lat_order]
        sorted_lons = src_lons[lon_order]
        sorted_field = field[lat_order, :][:, lon_order]

        interpolator = RegularGridInterpolator(
            (sorted_lats, sorted_lons),
            sorted_field,
            method=method,
            bounds_error=False,
            fill_value=fill_value
        )

        query_points = np.stack([self.grid.lat_grid.ravel(), self.grid.lon_grid.ravel()], axis=-1)
        regridded = interpolator(query_points).reshape(self.grid.H, self.grid.W)

        # Apply land-sea mask
        regridded[~self.grid.land_mask] = 0.0
        return regridded.astype(np.float32)

    def regrid_3d_profile(
        self,
        field_3d: np.ndarray,
        src_depths: np.ndarray,
        src_lats: np.ndarray,
        src_lons: np.ndarray
    ) -> np.ndarray:
        """Regrid 3D volume (D_src, H_src, W_src) to target (15, H_tgt, W_tgt).

        Args:
            field_3d: 3D array of shape (D_src, H_src, W_src)
            src_depths: Source 1D depth levels in meters.
            src_lats: Source 1D latitudes.
            src_lons: Source 1D longitudes.

        Returns:
            Regridded 3D array of shape (15, H_tgt, W_tgt)
        """
        D_tgt = self.grid.D
        target_3d = np.zeros((D_tgt, self.grid.H, self.grid.W), dtype=np.float32)

        # 1. Regrid each source depth slice spatially to (H_tgt, W_tgt)
        D_src = len(src_depths)
        temp_regridded_src = np.zeros((D_src, self.grid.H, self.grid.W), dtype=np.float32)
        for d in range(D_src):
            temp_regridded_src[d] = self.regrid_2d_field(field_3d[d], src_lats, src_lons)

        # 2. Interpolate vertically at each ocean grid cell to standard 15 depths
        for i in range(self.grid.H):
            for j in range(self.grid.W):
                if not self.grid.land_mask[i, j]:
                    continue
                profile_src = temp_regridded_src[:, i, j]
                target_3d[:, i, j] = np.interp(
                    self.grid.depth_levels,
                    src_depths,
                    profile_src,
                    left=profile_src[0],
                    right=profile_src[-1]
                )

        return target_3d
