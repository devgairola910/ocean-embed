"""Climatology computation and anomaly transforms for OceanEmbed.

Follows rules.md: Climatology is strictly computed from training period years only
to avoid temporal leakage into validation/test splits.
"""

from typing import Dict, Optional, Tuple, Union
import numpy as np


class ClimatologyComputer:
    """Computes and applies daily climatology and standardization for surface and depth fields."""

    def __init__(self, num_days: int = 366) -> None:
        """Initialize container for daily climatological means and standard deviations."""
        self.num_days = num_days
        self.surface_mean: Optional[np.ndarray] = None  # (366, C, H, W) or (C, H, W)
        self.surface_std: Optional[np.ndarray] = None   # (C, 1, 1) or (C, H, W)
        self.depth_mean: Optional[np.ndarray] = None    # (366, D, H, W) or (D, H, W)
        self.depth_std: Optional[np.ndarray] = None     # (D, 1, 1) or (D, H, W)
        self.is_fitted: bool = False

    def fit(
        self,
        surface_series: np.ndarray,
        depth_series: np.ndarray,
        doys: np.ndarray,
        smooth_window: int = 31
    ) -> "ClimatologyComputer":
        """Compute daily climatology from training time series.

        Args:
            surface_series: Array of shape (N_samples, C, H, W)
            depth_series: Array of shape (N_samples, D, H, W)
            doys: Day of year integer array (1 to 366) of shape (N_samples,)
            smooth_window: Rolling window size in days for harmonic/smooth climatology.

        Returns:
            Self (fitted instance)
        """
        N, C, H, W = surface_series.shape
        _, D, _, _ = depth_series.shape

        self.surface_mean = np.zeros((self.num_days, C, H, W), dtype=np.float32)
        self.depth_mean = np.zeros((self.num_days, D, H, W), dtype=np.float32)

        # Compute raw DOY averages
        for d in range(1, self.num_days + 1):
            mask = (doys == d)
            if np.any(mask):
                self.surface_mean[d - 1] = np.mean(surface_series[mask], axis=0)
                self.depth_mean[d - 1] = np.mean(depth_series[mask], axis=0)
            else:
                # If a specific DOY is missing, initialize with overall mean
                self.surface_mean[d - 1] = np.mean(surface_series, axis=0)
                self.depth_mean[d - 1] = np.mean(depth_series, axis=0)

        # Smooth across DOY cyclically with rolling average
        pad = smooth_window // 2
        for arr in [self.surface_mean, self.depth_mean]:
            padded = np.pad(arr, ((pad, pad), (0, 0), (0, 0), (0, 0)), mode="wrap")
            for i in range(self.num_days):
                arr[i] = np.mean(padded[i : i + smooth_window], axis=0)

        # Compute overall standardization statistics for anomaly fields
        # Compute anomalies on training series
        surf_anoms = []
        depth_anoms = []
        for i in range(N):
            doy_idx = int(doys[i] - 1) % self.num_days
            surf_anoms.append(surface_series[i] - self.surface_mean[doy_idx])
            depth_anoms.append(depth_series[i] - self.depth_mean[doy_idx])

        surf_anoms_arr = np.stack(surf_anoms, axis=0)
        depth_anoms_arr = np.stack(depth_anoms, axis=0)

        # Channel-wise standard deviation (per channel/depth level across all spatial/temporal samples)
        self.surface_std = np.std(surf_anoms_arr, axis=(0, 2, 3), keepdims=True) + 1e-6
        self.depth_std = np.std(depth_anoms_arr, axis=(0, 2, 3), keepdims=True) + 1e-6

        self.is_fitted = True
        return self

    def compute_surface_anomaly(self, raw_surface: np.ndarray, doy: int) -> np.ndarray:
        """Convert raw surface fields to standardized anomaly tensor.

        Args:
            raw_surface: Array of shape (C, H, W) or (B, C, H, W)
            doy: Day of year (1-366)

        Returns:
            Standardized anomaly array matching input shape.
        """
        doy_idx = int(doy - 1) % self.num_days
        mean = self.surface_mean[doy_idx]
        anomaly = raw_surface - mean
        std = self.surface_std.squeeze(0) if raw_surface.ndim == 3 else self.surface_std
        return anomaly / std

    def compute_depth_anomaly(self, raw_depth: np.ndarray, doy: int) -> np.ndarray:
        """Convert raw subsurface temperature to standardized anomaly tensor.

        Args:
            raw_depth: Array of shape (D, H, W) or (B, D, H, W)
            doy: Day of year (1-366)

        Returns:
            Standardized anomaly array matching input shape.
        """
        doy_idx = int(doy - 1) % self.num_days
        mean = self.depth_mean[doy_idx]
        anomaly = raw_depth - mean
        std = self.depth_std.squeeze(0) if raw_depth.ndim == 3 else self.depth_std
        return anomaly / std

    def reconstruct_absolute_temperature(self, temp_anomaly: np.ndarray, doy: int) -> np.ndarray:
        """Convert predicted standardized temperature anomaly back to absolute temperature in °C.

        Args:
            temp_anomaly: Array of shape (D, H, W) or (D,) or (B, D, H, W)
            doy: Day of year (1-366)

        Returns:
            Absolute temperature array in °C.
        """
        doy_idx = int(doy - 1) % self.num_days
        if temp_anomaly.ndim == 1:
            # Single profile (D,)
            mean_1d = np.mean(self.depth_mean[doy_idx], axis=(1, 2))
            std_1d = self.depth_std.squeeze()
            return temp_anomaly * std_1d + mean_1d
        elif temp_anomaly.ndim == 3:
            # (D, H, W)
            mean = self.depth_mean[doy_idx]
            std = self.depth_std.squeeze(0)
            return temp_anomaly * std + mean
        else:
            # (B, D, H, W)
            mean = self.depth_mean[doy_idx]
            std = self.depth_std
            return temp_anomaly * std + mean
