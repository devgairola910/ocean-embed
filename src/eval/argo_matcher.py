"""Spatio-temporal matching of model grid predictions with independent Argo float profiles."""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from src.data.grid import OceanGrid, STANDARD_DEPTH_LEVELS


class ArgoMatcher:
    """Matches 2D/3D model predictions with independent in-situ Argo float profiles."""

    def __init__(self, grid: Optional[OceanGrid] = None) -> None:
        """Initialize matcher with OceanGrid."""
        self.grid = grid or OceanGrid()
        self.depths = np.array(self.grid.depth_levels, dtype=np.float32)

    def match_single_profile(
        self,
        pred_temp_3d: np.ndarray,      # (15, H, W) in absolute °C
        argo_profile: Dict[str, Union[float, list, str]]
    ) -> Optional[Dict[str, Union[np.ndarray, float, str]]]:
        """Match a single Argo profile against the model's 3D temperature field.

        Args:
            pred_temp_3d: Array of shape (15, H, W) in absolute °C
            argo_profile: Dict containing 'lat', 'lon', 'depths', 'temperature', 'basin'

        Returns:
            Dict containing matched 'pred', 'actual', 'depths', 'lat', 'lon', 'basin'
            or None if location falls on land.
        """
        lat = float(argo_profile["lat"])
        lon = float(argo_profile["lon"])
        lat_idx, lon_idx = self.grid.find_nearest_indices(lat, lon)

        if not self.grid.land_mask[lat_idx, lon_idx]:
            return None

        model_profile = pred_temp_3d[:, lat_idx, lon_idx]  # (15,)

        # Extract and vertically interpolate Argo profile to model's 15 depths if necessary
        argo_depths = np.array(argo_profile["depths"], dtype=np.float32)
        argo_temps = np.array(argo_profile["temperature"], dtype=np.float32)

        if np.array_equal(argo_depths, self.depths):
            argo_interp = argo_temps
        else:
            argo_interp = np.interp(self.depths, argo_depths, argo_temps, left=argo_temps[0], right=argo_temps[-1])

        basin = argo_profile.get("basin", "arabian_sea" if lon < 77.5 else "bay_of_bengal")

        return {
            "float_id": argo_profile.get("float_id", "ARGO"),
            "lat": lat,
            "lon": lon,
            "basin": basin,
            "depths": self.depths,
            "pred_temperature": model_profile,
            "argo_temperature": argo_interp
        }

    def match_batch(
        self,
        pred_temp_3d: np.ndarray,
        argo_profiles: List[Dict[str, Union[float, list, str]]]
    ) -> List[Dict[str, Union[np.ndarray, float, str]]]:
        """Match multiple Argo profiles for a given date."""
        matched = []
        for prof in argo_profiles:
            res = self.match_single_profile(pred_temp_3d, prof)
            if res is not None:
                matched.append(res)
        return matched
