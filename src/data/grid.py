"""Grid geometry and spatial domain definitions for OceanEmbed.

Focus domain: North Indian Ocean (Arabian Sea, Bay of Bengal, Equatorial IO).
Grid resolution: 0.25° regular lat/lon grid.
Standard depth levels: 15 levels from surface (0 m) to 1000 m.
"""

from typing import List, Tuple, Optional
import numpy as np

# Standard depth levels in meters defined in PRD & Architecture
STANDARD_DEPTH_LEVELS: List[float] = [
    0.0, 10.0, 20.0, 30.0, 50.0, 75.0, 100.0, 125.0, 150.0, 200.0,
    250.0, 300.0, 400.0, 600.0, 1000.0
]

# Standard 7 surface input channels
STANDARD_SURFACE_CHANNELS: List[str] = [
    "sst_anomaly",   # Sea surface temperature anomaly (°C)
    "sss_anomaly",   # Sea surface salinity anomaly (psu)
    "ssh_anomaly",   # Sea surface height / SLA anomaly (m)
    "u_current",     # Surface zonal ocean current (m/s)
    "v_current",     # Surface meridional ocean current (m/s)
    "wind_u",        # Surface zonal wind at 10m (m/s)
    "wind_v"         # Surface meridional wind at 10m (m/s)
]


class OceanGrid:
    """Manages the 2D spatial coordinate grid, sub-basins, and land-sea masks."""

    def __init__(
        self,
        lat_min: float = 0.0,
        lat_max: float = 25.0,
        lon_min: float = 40.0,
        lon_max: float = 100.0,
        resolution: float = 0.25,
        depth_levels: Optional[List[float]] = None
    ) -> None:
        """Initialize coordinate vectors and dimensions.

        Args:
            lat_min: Southern boundary in degrees North.
            lat_max: Northern boundary in degrees North.
            lon_min: Western boundary in degrees East.
            lon_max: Eastern boundary in degrees East.
            resolution: Grid spacing in degrees.
            depth_levels: Custom depth levels or defaults to 15 standard depths.
        """
        self.lat_min = lat_min
        self.lat_max = lat_max
        self.lon_min = lon_min
        self.lon_max = lon_max
        self.resolution = resolution
        self.depth_levels = depth_levels or STANDARD_DEPTH_LEVELS

        self.lats = np.arange(lat_min, lat_max + 1e-6, resolution)
        self.lons = np.arange(lon_min, lon_max + 1e-6, resolution)
        self.H = len(self.lats)
        self.W = len(self.lons)
        self.D = len(self.depth_levels)

        self.lon_grid, self.lat_grid = np.meshgrid(self.lons, self.lats)
        self.land_mask = self._generate_land_sea_mask()

    def _generate_land_sea_mask(self) -> np.ndarray:
        """Generate a realistic land-sea mask (1 = ocean, 0 = land) for North Indian Ocean.

        Returns:
            Boolean ndarray of shape (H, W) where True is ocean and False is land.
        """
        ocean = np.ones((self.H, self.W), dtype=bool)
        lats = self.lat_grid
        lons = self.lon_grid

        # 1. Indian Subcontinent (approx triangle between 8°N and 25°N, 68°E and 88°E)
        # Western Ghats / West Coast approx line
        for i in range(self.H):
            for j in range(self.W):
                lat = lats[i, j]
                lon = lons[i, j]

                # Main Indian landmass
                if lat >= 8.0:
                    west_bound = 68.0 + (lat - 8.0) * 0.35
                    east_bound = 88.0 - (lat - 8.0) * 0.2
                    if lat > 20.0:
                        west_bound = 68.0
                        east_bound = 90.0
                    if west_bound <= lon <= east_bound and lat <= 25.0:
                        ocean[i, j] = False

                # Arabian Peninsula & Horn of Africa (west of 60E above 12N, west of 50E)
                if lon <= 55.0 and lat >= 12.0:
                    ocean[i, j] = False
                if lon <= 45.0:
                    ocean[i, j] = False
                if lat >= 22.0 and lon <= 62.0:  # Oman/UAE/Iran coast
                    ocean[i, j] = False
                if lat >= 24.0 and lon <= 68.0:  # Pakistan/Iran
                    ocean[i, j] = False

                # Myanmar / Southeast Asia (East of 92E above 14N, East of 98E)
                if lon >= 94.0 and lat >= 16.0:
                    ocean[i, j] = False
                if lon >= 98.0 and lat >= 8.0:  # Malay Peninsula / Thailand
                    ocean[i, j] = False

        return ocean

    def get_basin_mask(self, basin: str) -> np.ndarray:
        """Get sub-region boolean mask for ocean pixels.

        Args:
            basin: 'arabian_sea', 'bay_of_bengal', or 'all'

        Returns:
            Boolean ndarray of shape (H, W)
        """
        ocean = self.land_mask.copy()
        lats = self.lat_grid
        lons = self.lon_grid

        if basin == "arabian_sea":
            in_basin = (lons >= 45.0) & (lons <= 77.5) & (lats >= 4.0) & (lats <= 25.0)
            return ocean & in_basin
        elif basin == "bay_of_bengal":
            in_basin = (lons >= 79.5) & (lons <= 98.0) & (lats >= 4.0) & (lats <= 23.0)
            return ocean & in_basin
        elif basin == "equatorial":
            in_basin = (lats >= 0.0) & (lats <= 5.0)
            return ocean & in_basin
        elif basin == "all":
            return ocean
        else:
            raise ValueError(f"Unknown basin '{basin}'. Choose 'arabian_sea', 'bay_of_bengal', 'equatorial', or 'all'.")

    def find_nearest_indices(self, lat: float, lon: float) -> Tuple[int, int]:
        """Find the nearest (lat_idx, lon_idx) on the grid for given coordinates.

        Args:
            lat: Target latitude in degrees North.
            lon: Target longitude in degrees East.

        Returns:
            Tuple of (lat_index, lon_index)
        """
        lat_idx = int(np.argmin(np.abs(self.lats - lat)))
        lon_idx = int(np.argmin(np.abs(self.lons - lon)))
        return lat_idx, lon_idx
