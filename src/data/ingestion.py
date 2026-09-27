"""Data ingestion routines for Copernicus Marine, NASA PODAAC, and Argo GDAC.

Provides automated ingestion utilities and NetCDF readers for:
- OSTIA SST (Copernicus Marine)
- SMAP SSS (NASA PODAAC)
- DUACS Sea Surface Height / SLA (Copernicus Marine)
- OSCAR Surface Currents (NASA PODAAC)
- CCMP Surface Winds (NASA / RSS)
- GLORYS12V1 Subsurface Reanalysis (Copernicus Marine)
- Argo Float Profiles (Argo GDAC / argopy)
"""

from typing import Dict, List, Optional, Tuple, Union
import os
import numpy as np
import pandas as pd
import xarray as xr

from src.data.grid import OceanGrid, STANDARD_DEPTH_LEVELS, STANDARD_SURFACE_CHANNELS
from src.data.regrid import OceanDataRegridder


class SatelliteDataIngestor:
    """Ingests, parses, and formats raw satellite and reanalysis NetCDF datasets."""

    def __init__(self, raw_dir: str = "data/raw", interim_dir: str = "data/interim", grid: Optional[OceanGrid] = None) -> None:
        """Initialize data ingestor.

        Args:
            raw_dir: Directory where raw downloaded NetCDF files are stored.
            interim_dir: Directory for regridded intermediate files.
            grid: Target OceanGrid (default: 0.25° NIO grid).
        """
        self.raw_dir = raw_dir
        self.interim_dir = interim_dir
        self.grid = grid or OceanGrid()
        self.regridder = OceanDataRegridder(self.grid)

        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.interim_dir, exist_ok=True)

    def load_ostia_sst(self, file_path: str) -> np.ndarray:
        """Load and regrid OSTIA SST NetCDF file to 0.25° grid.

        Args:
            file_path: Path to OSTIA NetCDF file.

        Returns:
            Regridded 2D array (H, W) in °C.
        """
        ds = xr.open_dataset(file_path)
        # OSTIA standard variable name is 'analysed_sst' in Kelvin
        var_name = "analysed_sst" if "analysed_sst" in ds else "sst"
        sst_k = ds[var_name].values.squeeze()
        # Convert Kelvin to Celsius if necessary
        sst_c = sst_k - 273.15 if np.nanmean(sst_k) > 200.0 else sst_k

        lats = ds["lat"].values if "lat" in ds else ds["latitude"].values
        lons = ds["lon"].values if "lon" in ds else ds["longitude"].values
        ds.close()

        return self.regridder.regrid_2d_field(sst_c, lats, lons)

    def load_smap_sss(self, file_path: str) -> np.ndarray:
        """Load and regrid SMAP SSS NetCDF file to 0.25° grid.

        Args:
            file_path: Path to SMAP NetCDF file.

        Returns:
            Regridded 2D array (H, W) in psu.
        """
        ds = xr.open_dataset(file_path)
        var_name = "sss" if "sss" in ds else "smap_sss"
        sss = ds[var_name].values.squeeze()
        lats = ds["lat"].values if "lat" in ds else ds["latitude"].values
        lons = ds["lon"].values if "lon" in ds else ds["longitude"].values
        ds.close()

        return self.regridder.regrid_2d_field(sss, lats, lons)

    def load_duacs_ssh(self, file_path: str) -> np.ndarray:
        """Load and regrid DUACS Sea Level Anomaly (SLA) NetCDF file.

        Args:
            file_path: Path to DUACS NetCDF file.

        Returns:
            Regridded 2D array (H, W) in meters.
        """
        ds = xr.open_dataset(file_path)
        var_name = "sla" if "sla" in ds else "adt"
        sla = ds[var_name].values.squeeze()
        lats = ds["lat"].values if "lat" in ds else ds["latitude"].values
        lons = ds["lon"].values if "lon" in ds else ds["longitude"].values
        ds.close()

        return self.regridder.regrid_2d_field(sla, lats, lons)

    def load_glorys_reanalysis(self, file_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load and regrid GLORYS12V1 3D reanalysis temperature and thermocline depth.

        Args:
            file_path: Path to GLORYS NetCDF file.

        Returns:
            Tuple of:
              - temp_15d: (15, H, W) in °C
              - thermocline_depth: (H, W) in meters
        """
        ds = xr.open_dataset(file_path)
        var_name = "thetao" if "thetao" in ds else "temp"
        temp_3d = ds[var_name].values.squeeze()  # (depth, lat, lon)
        depths = ds["depth"].values
        lats = ds["lat"].values if "lat" in ds else ds["latitude"].values
        lons = ds["lon"].values if "lon" in ds else ds["longitude"].values
        ds.close()

        # Regrid spatially and vertically to 15 standard depth levels
        regridded_3d = self.regridder.regrid_3d_profile(temp_3d, depths, lats, lons)

        # Compute thermocline depth = max vertical gradient depth
        dT = -(regridded_3d[1:] - regridded_3d[:-1])
        dz = (np.array(self.grid.depth_levels[1:]) - np.array(self.grid.depth_levels[:-1]))[:, np.newaxis, np.newaxis]
        grad = dT / dz
        max_idx = np.argmax(grad, axis=0)
        mid_depths = 0.5 * (np.array(self.grid.depth_levels[1:]) + np.array(self.grid.depth_levels[:-1]))
        thermo_depth = mid_depths[max_idx].astype(np.float32)
        thermo_depth[~self.grid.land_mask] = 0.0

        return regridded_3d, thermo_depth

    def create_mock_raw_netcdf_sample(self, output_path: str, year: int = 2022, doy: int = 180) -> str:
        """Create a standard self-describing NetCDF file containing all 7 surface channels and GLORYS depth levels.

        Useful for standalone testing, offline reproduction, and validation without external network credentials.

        Args:
            output_path: Target NetCDF file path.
            year: Calendar year.
            doy: Day of year.

        Returns:
            Saved file path.
        """
        from src.data.synthetic import PhysicalOceanSynthesizer
        synth = PhysicalOceanSynthesizer(self.grid, seed=year + doy)
        surf, depth, thermo = synth.generate_day(year, doy, add_eddy_field=True)

        ds = xr.Dataset(
            data_vars={
                "sst": (["lat", "lon"], surf[0], {"units": "degC", "long_name": "Sea Surface Temperature"}),
                "sss": (["lat", "lon"], surf[1], {"units": "psu", "long_name": "Sea Surface Salinity"}),
                "sla": (["lat", "lon"], surf[2], {"units": "m", "long_name": "Sea Level Anomaly"}),
                "u_curr": (["lat", "lon"], surf[3], {"units": "m/s", "long_name": "Zonal Surface Current"}),
                "v_curr": (["lat", "lon"], surf[4], {"units": "m/s", "long_name": "Meridional Surface Current"}),
                "wind_u": (["lat", "lon"], surf[5], {"units": "m/s", "long_name": "Zonal 10m Wind"}),
                "wind_v": (["lat", "lon"], surf[6], {"units": "m/s", "long_name": "Meridional 10m Wind"}),
                "thetao": (["depth", "lat", "lon"], depth, {"units": "degC", "long_name": "Potential Temperature"}),
                "thermocline_depth": (["lat", "lon"], thermo, {"units": "m", "long_name": "Thermocline Depth"}),
                "land_mask": (["lat", "lon"], self.grid.land_mask.astype(np.int32), {"long_name": "Land Sea Mask (1=Ocean)"})
            },
            coords={
                "lat": (["lat"], self.grid.lats, {"units": "degrees_north"}),
                "lon": (["lon"], self.grid.lons, {"units": "degrees_east"}),
                "depth": (["depth"], self.grid.depth_levels, {"units": "m"}),
                "time": [pd.to_datetime(f"{year}-01-01") + pd.Timedelta(days=doy - 1)]
            },
            attrs={
                "title": "OceanEmbed Harmonized 0.25° Satellite & Reanalysis Daily Product",
                "institution": "MoES / INCOIS / Copernicus Marine / NASA PODAAC",
                "spatial_domain": "North Indian Ocean (0-25N, 40-100E)",
                "grid_resolution": "0.25 deg"
            }
        )

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        ds.to_netcdf(output_path)
        ds.close()
        return output_path
