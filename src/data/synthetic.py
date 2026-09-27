"""Physical ocean data generator and realistic emulator for North Indian Ocean.

Implements physically-grounded geophysical dynamics:
- Monsoonal wind regimes (SW & NE Monsoons)
- Mixed layer, thermocline steepness, and deep ocean stratification (0-1000m)
- Mesoscale eddy coupling (SLA-thermocline height linkage)
- Realistic synthetic GLORYS and independent Argo profiles for offline training/demo.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from src.data.grid import OceanGrid, STANDARD_DEPTH_LEVELS, STANDARD_SURFACE_CHANNELS


class PhysicalOceanSynthesizer:
    """Generates physically consistent multi-year ocean surface and subsurface datasets."""

    def __init__(self, grid: Optional[OceanGrid] = None, seed: int = 42) -> None:
        """Initialize synthesizer with spatial grid.

        Args:
            grid: OceanGrid instance or default NIO grid.
            seed: Random seed for reproducibility.
        """
        self.grid = grid or OceanGrid()
        self.rng = np.random.default_rng(seed)
        self.depths = np.array(self.grid.depth_levels, dtype=np.float32)

    def _compute_background_profile(self, sst: float, lat: float, lon: float) -> np.ndarray:
        """Compute physically stable 1D temperature profile (0-1000m) from surface state.

        Uses a two-layer thermocline model with exponential deep decay:
        T(z) = T_deep + (SST - T_deep) / (1 + exp((z - Z_th) / H_th))

        Args:
            sst: Surface temperature (°C).
            lat: Latitude (°N).
            lon: Longitude (°E).

        Returns:
            1D array of shape (D,) with temperatures strictly non-increasing with depth.
        """
        t_deep = 4.5  # Typical temperature at 1000m in NIO
        # Thermocline depth varies with latitude (deeper in central Arabian Sea, shallower near equator)
        z_th = 60.0 + 20.0 * np.sin(np.radians(lat * 3.6)) + 10.0 * np.cos(np.radians(lon * 2.0))
        h_th = 35.0  # Thermocline transition thickness scale (m)

        profile = t_deep + (sst - t_deep) / (1.0 + np.exp((self.depths - z_th) / h_th))
        return profile

    def generate_day(
        self,
        year: int,
        day_of_year: int,
        add_eddy_field: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate a single day of 7-channel surface fields and 15-depth temperature fields.

        Args:
            year: Calendar year (e.g. 2020)
            day_of_year: Day of year (1 to 365)
            add_eddy_field: Whether to inject coherent mesoscale eddy anomalies.

        Returns:
            Tuple of:
              - surface_fields: ndarray of shape (7, H, W)
              - depth_temperature: ndarray of shape (15, H, W) in °C
              - thermocline_depth: ndarray of shape (H, W) in meters
        """
        H, W = self.grid.H, self.grid.W
        D = self.grid.D
        lats = self.grid.lat_grid
        lons = self.grid.lon_grid
        ocean_mask = self.grid.land_mask

        # 1. Seasonal Monsoonal Cycle (DOY 150-250 is SW Monsoon, DOY 320-50 is NE Monsoon)
        monsoon_phase = 2.0 * np.pi * (day_of_year - 15) / 365.0
        sw_monsoon_intensity = np.clip(np.sin(monsoon_phase), -1.0, 1.0)

        # Baseline Climatological SST (27°C - 31°C)
        sst_base = 28.5 + 1.8 * np.sin(monsoon_phase - 0.5) - 0.15 * lats
        # SW Monsoon upwelling cooling off Oman / Somalia / SW India
        upwelling = np.zeros_like(sst_base)
        upwelling += 2.5 * np.exp(-((lons - 55.0)**2 + (lats - 14.0)**2) / 35.0) * max(0.0, sw_monsoon_intensity)
        upwelling += 1.8 * np.exp(-((lons - 75.0)**2 + (lats - 9.0)**2) / 20.0) * max(0.0, sw_monsoon_intensity)
        sst = sst_base - upwelling

        # Baseline SSS (Salinity: 32 psu in northern Bay of Bengal to 36.5 psu in Arabian Sea)
        sss = 35.0 + 1.2 * (lons < 77.0) - 2.8 * (lons > 85.0) * (lats > 12.0)
        sss += 0.3 * np.sin(monsoon_phase)

        # Baseline SSH (Sea Surface Height in meters, -0.4 to +0.4 m)
        ssh = 0.15 * np.sin(monsoon_phase) + 0.08 * np.sin(np.radians(lons * 3.0))

        # Surface Winds (m/s)
        wind_u = 7.0 * sw_monsoon_intensity + 1.5 * np.sin(np.radians(lats * 5.0))
        wind_v = 6.0 * max(0.0, sw_monsoon_intensity) - 3.0 * max(0.0, -sw_monsoon_intensity)

        # Surface Currents (m/s) - Geostrophic + Ekman driven
        u_curr = 0.35 * sw_monsoon_intensity + 0.1 * np.cos(np.radians(lats * 4.0))
        v_curr = 0.25 * max(0.0, sw_monsoon_intensity) + 0.05 * np.sin(np.radians(lons * 3.0))

        # 2. Interannual Anomaly & Coherent Mesoscale Eddies
        # Interannual shift (e.g. Indian Ocean Dipole / ENSO teleconnection)
        iod_phase = np.sin((year - 2015) * 1.3)
        sst += 0.4 * iod_phase * (1.0 if lons.mean() > 80.0 else -1.0)

        # Add coherent mesoscale vortices
        temp_subsurface_anom = np.zeros((D, H, W), dtype=np.float32)
        if add_eddy_field:
            num_eddies = 10
            lat_span = self.grid.lat_max - self.grid.lat_min
            lon_span = self.grid.lon_max - self.grid.lon_min
            lat_pad = min(2.0, lat_span * 0.2)
            lon_pad = min(3.0, lon_span * 0.2)
            lat_low, lat_high = self.grid.lat_min + lat_pad, self.grid.lat_max - lat_pad
            lon_low, lon_high = self.grid.lon_min + lon_pad, self.grid.lon_max - lon_pad
            if lat_high <= lat_low:
                lat_low, lat_high = self.grid.lat_min, self.grid.lat_max
            if lon_high <= lon_low:
                lon_low, lon_high = self.grid.lon_min, self.grid.lon_max

            for _ in range(num_eddies):
                eddy_lat = self.rng.uniform(lat_low, lat_high)
                eddy_lon = self.rng.uniform(lon_low, lon_high)
                polarity = self.rng.choice([-1.0, 1.0])  # Cyclonic (-1) or Anticyclonic (+1)
                radius = max(0.8, min(3.0, lat_span * 0.2))  # degrees radius
                amp_ssh = polarity * self.rng.uniform(0.08, 0.25)  # meters SLA
                amp_sst = polarity * self.rng.uniform(0.4, 1.2)   # °C SST

                dist_sq = (lats - eddy_lat)**2 + (lons - eddy_lon)**2
                eddy_mask = np.exp(-dist_sq / (2.0 * radius**2))

                ssh += amp_ssh * eddy_mask
                sst += amp_sst * eddy_mask

                # Eddy effect on subsurface temperature: strongest at thermocline depth (50-200m)
                for d_idx, z in enumerate(self.depths):
                    # Vertical eddy envelope peaking at 100m
                    z_weight = (z / 100.0) * np.exp(1.0 - z / 100.0)
                    temp_subsurface_anom[d_idx] += (polarity * 2.2 * z_weight) * eddy_mask.astype(np.float32)

        # Add small high-frequency noise to surface channels
        sst += self.rng.normal(0.0, 0.08, size=(H, W))
        sss += self.rng.normal(0.0, 0.05, size=(H, W))
        ssh += self.rng.normal(0.0, 0.015, size=(H, W))
        u_curr += self.rng.normal(0.0, 0.03, size=(H, W))
        v_curr += self.rng.normal(0.0, 0.03, size=(H, W))
        wind_u += self.rng.normal(0.0, 0.3, size=(H, W))
        wind_v += self.rng.normal(0.0, 0.3, size=(H, W))

        # 3. Vectorized Subsurface Temperature Field (15 depths)
        t_deep = 4.5
        z_th = 60.0 + 20.0 * np.sin(np.radians(lats * 3.6)) + 10.0 * np.cos(np.radians(lons * 2.0))  # (H, W)
        h_th = 35.0

        depths_3d = self.depths[:, np.newaxis, np.newaxis]  # (D, 1, 1)
        z_th_3d = z_th[np.newaxis, :, :]                    # (1, H, W)
        sst_3d = sst[np.newaxis, :, :]                      # (1, H, W)

        bg_prof = t_deep + (sst_3d - t_deep) / (1.0 + np.exp((depths_3d - z_th_3d) / h_th))  # (D, H, W)
        depth_temperature = (bg_prof + temp_subsurface_anom).astype(np.float32)

        # Enforce physical monotonicity across depths
        for d in range(1, D):
            inv_mask = depth_temperature[d] > depth_temperature[d - 1]
            depth_temperature[d, inv_mask] = depth_temperature[d - 1, inv_mask] - 0.01

        # Compute thermocline depth = depth where vertical gradient max(-dT/dz) occurs
        dT = -(depth_temperature[1:] - depth_temperature[:-1])               # (D-1, H, W)
        dz = (self.depths[1:] - self.depths[:-1])[:, np.newaxis, np.newaxis] # (D-1, 1, 1)
        grad = dT / dz                                                       # (D-1, H, W)
        max_grad_idx = np.argmax(grad, axis=0)                               # (H, W)
        mid_depths = 0.5 * (self.depths[1:] + self.depths[:-1])
        thermocline_depth = mid_depths[max_grad_idx].astype(np.float32)      # (H, W)

        # Stack 7 surface channels
        surface_fields = np.stack([sst, sss, ssh, u_curr, v_curr, wind_u, wind_v], axis=0).astype(np.float32)

        # Zero-out land pixels
        surface_fields[:, ~ocean_mask] = 0.0
        depth_temperature[:, ~ocean_mask] = 0.0
        thermocline_depth[~ocean_mask] = 0.0

        return surface_fields, depth_temperature, thermocline_depth

    def generate_argo_profiles(
        self,
        year: int,
        day_of_year: int,
        num_floats: int = 12
    ) -> List[Dict[str, float]]:
        """Generate independent Argo float profiles for validation.

        Args:
            year: Calendar year
            day_of_year: Day of year
            num_floats: Number of float profiles to generate in NIO domain.

        Returns:
            List of dicts representing Argo profile measurements.
        """
        _, full_depth_temp, _ = self.generate_day(year, day_of_year, add_eddy_field=True)
        argo_list = []

        for f_idx in range(num_floats):
            # Select random ocean location
            lat_span = self.grid.lat_max - self.grid.lat_min
            lon_span = self.grid.lon_max - self.grid.lon_min
            lat_min_b = self.grid.lat_min + (0.5 if lat_span > 2.0 else 0.0)
            lat_max_b = self.grid.lat_max - (0.5 if lat_span > 2.0 else 0.0)
            lon_min_b = self.grid.lon_min + (0.5 if lon_span > 2.0 else 0.0)
            lon_max_b = self.grid.lon_max - (0.5 if lon_span > 2.0 else 0.0)

            lat_i, lon_j = 0, 0
            for _ in range(50):
                lat = float(self.rng.uniform(lat_min_b, lat_max_b))
                lon = float(self.rng.uniform(lon_min_b, lon_max_b))
                lat_i, lon_j = self.grid.find_nearest_indices(lat, lon)
                if self.grid.land_mask[lat_i, lon_j]:
                    break

            # Extract ground truth profile and add in-situ sensor noise
            true_prof = full_depth_temp[:, lat_i, lon_j].copy()
            noise = self.rng.normal(0.0, 0.05, size=len(true_prof))
            measured_prof = true_prof + noise

            # Determine basin
            if lon < 77.5:
                basin = "arabian_sea"
            elif lon > 79.5:
                basin = "bay_of_bengal"
            else:
                basin = "equatorial"

            argo_list.append({
                "float_id": f"ARGO_290{1000 + f_idx}",
                "year": year,
                "doy": day_of_year,
                "lat": round(lat, 2),
                "lon": round(lon, 2),
                "basin": basin,
                "depths": self.depths.tolist(),
                "temperature": measured_prof.tolist()
            })

        return argo_list
