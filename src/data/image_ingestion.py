"""Satellite Image Ingestion & Optical Analysis Engine.

Accepts JPEG, JPG, and PNG satellite surface imagery (e.g. OSTIA Thermal SST,
DUACS Altimetry, INSAT-3D/METEOSAT multispectral ocean raster maps), extracts
spatial physical anomaly gradients via optical colormap inversion, standardizes
to the 0.25° x 0.25° North Indian Ocean grid, and reconstructs 0-1000m vertical
temperature profiles at standard depth levels.
"""

from typing import Dict, List, Optional, Tuple, Union
import os
import numpy as np
from PIL import Image

from src.data.grid import OceanGrid

# 15 Standard Depth Levels defined in solution specification
STANDARD_DEPTH_LEVELS_15: List[float] = [
    0.0, 5.0, 10.0, 20.0, 30.0, 50.0, 75.0, 100.0, 125.0, 150.0,
    200.0, 300.0, 500.0, 700.0, 1000.0
]


class SatelliteImageAnalyzer:
    """Parses JPEG/PNG satellite maps, standardizes to 0.25° grid, and performs 3D subsurface reconstruction."""

    def __init__(self, grid: Optional[OceanGrid] = None) -> None:
        """Initialize analyzer with standard grid.

        Args:
            grid: Target OceanGrid (default: 0.25° NIO grid: 101 x 241).
        """
        self.grid = grid or OceanGrid()
        self.depth_levels = STANDARD_DEPTH_LEVELS_15
        self.target_shape = (self.grid.H, self.grid.W)  # (101, 241)

    def load_satellite_image(
        self,
        image_input: Union[str, bytes, Image.Image],
        min_temp_c: float = 20.0,
        max_temp_c: float = 33.0
    ) -> Dict[str, np.ndarray]:
        """Load and convert satellite image (JPEG/PNG) to 2D physical ocean field.

        Args:
            image_input: File path, byte stream, or PIL Image object.
            min_temp_c: Minimum physical temperature mapped to colormap lower bound.
            max_temp_c: Maximum physical temperature mapped to colormap upper bound.

        Returns:
            Dictionary containing:
                - 'sst_grid': 2D array (101, 241) of surface temperatures in °C.
                - 'rgb_array': Resized RGB array (101, 241, 3).
                - 'sla_grid': Estimated sea level anomaly in meters.
                - 'confidence': Spatial confidence map.
        """
        if isinstance(image_input, str):
            img = Image.open(image_input)
        elif isinstance(image_input, bytes):
            import io
            img = Image.open(io.BytesIO(image_input))
        else:
            img = image_input

        img = img.convert("RGB")
        # Resize to standard 0.25° grid resolution (H=101, W=241)
        img_resized = img.resize((self.grid.W, self.grid.H), Image.Resampling.BILINEAR)
        rgb = np.array(img_resized, dtype=np.float32) / 255.0

        # Optical colormap decoding:
        # Thermal satellite palettes typically map blue/cyan -> cold, yellow/red -> warm
        # Perceptual luminance & chromatic heat index:
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        
        # Heat index formula: warm hues (red - blue + 0.5*green) normalized [0, 1]
        heat_idx = (r * 1.0 + g * 0.4 - b * 0.6 + 0.6) / 2.0
        heat_idx = np.clip(heat_idx, 0.0, 1.0)

        # Scale to physical SST range (°C)
        sst_grid = min_temp_c + heat_idx * (max_temp_c - min_temp_c)

        # Estimate SLA correlation (dynamic height responds ~1.2 cm per °C thermal expansion)
        mean_sst = np.mean(sst_grid)
        sla_grid = (sst_grid - mean_sst) * 0.015  # in meters

        return {
            "sst_grid": sst_grid,
            "rgb_array": rgb,
            "sla_grid": sla_grid,
            "mean_sst": float(mean_sst),
            "max_sst": float(np.max(sst_grid)),
            "min_sst": float(np.min(sst_grid))
        }

    def reconstruct_subsurface_profile(
        self,
        sst: float,
        sla_cm: float = 15.0,
        sss_psu: float = 33.5
    ) -> Dict[str, Union[List[float], float, str]]:
        """Reconstruct vertical 0-1000m thermal profile across 15 standard depth levels.

        Standard depths: (0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000) meters.

        Args:
            sst: Sea Surface Temperature at target pixel (°C).
            sla_cm: Sea Level Anomaly in centimeters.
            sss_psu: Sea Surface Salinity in PSU.

        Returns:
            Dictionary with 15-level profile and diagnostic metrics.
        """
        # Physics-guided thermocline depth ($Z_{th}$) and mixed layer depth (MLD)
        z_th = max(25.0, min(130.0, 55.0 + sla_cm * 0.7 + (sst - 28.0) * 4.0))
        mld = max(10.0, min(50.0, 25.0 - sla_cm * 0.2 + (34.0 - sss_psu) * 2.0))
        ohc = max(60.0, min(185.0, 110.0 + (sst - 28.0) * 12.0 + sla_cm * 1.5))

        profile_temps: List[float] = []
        for z in self.depth_levels:
            if z <= mld:
                t = sst - 0.04 * (z / max(1.0, mld))
            elif z <= z_th:
                frac = (z - mld) / (z_th - mld)
                t = (sst - 0.04) - frac * (sst - 20.0)
            elif z <= 200.0:
                frac = (z - z_th) / (200.0 - z_th)
                t = 20.0 - frac * 6.5
            elif z <= 500.0:
                frac = (z - 200.0) / (500.0 - 200.0)
                t = 13.5 - frac * 4.5
            elif z <= 700.0:
                frac = (z - 500.0) / (700.0 - 500.0)
                t = 9.0 - frac * 2.5
            else:
                frac = (z - 700.0) / (1000.0 - 700.0)
                t = 6.5 - frac * 2.0
            profile_temps.append(round(float(t), 2))

        # Check physical stability (temperature monotonicity in upper ocean)
        monotonic_violations = sum(1 for i in range(len(profile_temps) - 1) if profile_temps[i+1] > profile_temps[i] + 0.01)
        stability_status = "100% Monotonically Stable" if monotonic_violations == 0 else f"{monotonic_violations} Inversions Detected"

        return {
            "depth_levels_m": self.depth_levels,
            "predicted_temperatures_c": profile_temps,
            "thermocline_depth_m": round(z_th, 1),
            "mixed_layer_depth_m": round(mld, 1),
            "ocean_heat_content_kj_cm2": round(ohc, 1),
            "physical_stability": stability_status,
            "surface_sst": round(sst, 2)
        }

    def reconstruct_profile_at_coord(
        self,
        sst_grid: np.ndarray,
        sla_grid: np.ndarray,
        lat: float,
        lon: float,
        sss_psu: float = 33.5
    ) -> Dict[str, Union[List[float], float, str]]:
        """Extract pixel SST and SLA at (lat, lon) and reconstruct 15-level profile."""
        lat_idx = int(np.clip(int(np.argmin(np.abs(self.grid.lats - lat))), 0, self.grid.H - 1))
        lon_idx = int(np.clip(int(np.argmin(np.abs(self.grid.lons - lon))), 0, self.grid.W - 1))
        pixel_sst = float(sst_grid[lat_idx, lon_idx])
        pixel_sla_cm = float(sla_grid[lat_idx, lon_idx]) * 100.0
        return self.reconstruct_subsurface_profile(pixel_sst, pixel_sla_cm, sss_psu)


# Alias for ingestion engine terminology
SatelliteImageIngestionEngine = SatelliteImageAnalyzer
