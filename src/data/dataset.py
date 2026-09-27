"""PyTorch Dataset and DataLoader implementations for OceanEmbed.

Implements sliding-window lag stacking:
  Input tensor:  (B, T_lag, C, H, W) where C=7 surface anomaly channels, T_lag=5 days
  Target tensor: (B, D, H, W) where D=15 subsurface temperature anomaly levels
  Aux target:    (B, H, W) scalar thermocline depth in meters
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from src.data.grid import OceanGrid, STANDARD_DEPTH_LEVELS
from src.data.climatology import ClimatologyComputer
from src.data.synthetic import PhysicalOceanSynthesizer


class OceanEmbedDataset(Dataset):
    """PyTorch Dataset yielding lagged surface anomaly stacks and subsurface target profiles."""

    def __init__(
        self,
        surface_series: np.ndarray,
        depth_series: np.ndarray,
        thermocline_series: np.ndarray,
        dates: List[Tuple[int, int]],  # List of (year, doy)
        climatology: ClimatologyComputer,
        lag_days: int = 5,
        grid: Optional[OceanGrid] = None
    ) -> None:
        """Initialize dataset.

        Args:
            surface_series: Raw surface fields of shape (N_timesteps, 7, H, W)
            depth_series: Raw subsurface temperature fields of shape (N_timesteps, 15, H, W)
            thermocline_series: Thermocline depth maps of shape (N_timesteps, H, W)
            dates: List of (year, day_of_year) tuples for each timestep in sequence.
            climatology: Fitted ClimatologyComputer instance.
            lag_days: Number of past days to stack (default: 5).
            grid: OceanGrid instance.
        """
        super().__init__()
        self.surface_series = surface_series
        self.depth_series = depth_series
        self.thermocline_series = thermocline_series
        self.dates = dates
        self.climatology = climatology
        self.lag_days = lag_days
        self.grid = grid or OceanGrid()

        assert len(surface_series) == len(depth_series) == len(dates), "Length mismatch in dataset series."
        assert len(surface_series) >= lag_days, f"Series length ({len(surface_series)}) must exceed lag_days ({lag_days})."

        # Number of valid sliding windows
        self.num_samples = len(surface_series) - lag_days + 1

    def __len__(self) -> int:
        """Return total number of lagged sample windows."""
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Fetch single lagged batch item.

        Returns:
            Dict containing:
              - 'surface_input': Tensor of shape (T_lag, C, H, W) in standardized anomaly space
              - 'target_depth': Tensor of shape (D, H, W) in standardized temperature anomaly space
              - 'target_thermocline': Tensor of shape (H, W) in meters
              - 'land_mask': Bool Tensor of shape (H, W) where True=Ocean
              - 'target_date': Tensor of shape (2,) containing [year, doy]
        """
        # Window spans indices [idx, idx + lag_days)
        window_start = idx
        window_end = idx + self.lag_days
        target_idx = window_end - 1  # Predict subsurface state for today (last day in window)

        target_year, target_doy = self.dates[target_idx]

        # Compute standardized anomalies for each day in the lag window
        lag_anomalies = []
        for t in range(window_start, window_end):
            _, doy = self.dates[t]
            raw_surf = self.surface_series[t]
            anom_surf = self.climatology.compute_surface_anomaly(raw_surf, doy)
            lag_anomalies.append(anom_surf)

        # Stack into (T_lag, C, H, W)
        input_stack = np.stack(lag_anomalies, axis=0).astype(np.float32)

        # Target anomaly for the current day
        raw_depth = self.depth_series[target_idx]
        target_depth_anom = self.climatology.compute_depth_anomaly(raw_depth, target_doy).astype(np.float32)

        target_thermo = self.thermocline_series[target_idx].astype(np.float32)
        ocean_mask = self.grid.land_mask.astype(bool)

        return {
            "surface_input": torch.from_numpy(input_stack),          # (T_lag, C, H, W)
            "target_depth": torch.from_numpy(target_depth_anom),      # (D, H, W)
            "target_thermocline": torch.from_numpy(target_thermo),    # (H, W)
            "land_mask": torch.from_numpy(ocean_mask),                # (H, W)
            "target_date": torch.tensor([target_year, target_doy], dtype=torch.long)
        }


def generate_dataset_split(
    years: List[int],
    grid: OceanGrid,
    synthesizer: PhysicalOceanSynthesizer,
    days_step: int = 5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Tuple[int, int]]]:
    """Generate daily time-series sequence for specified calendar years.

    Args:
        years: List of years in split
        grid: OceanGrid
        synthesizer: PhysicalOceanSynthesizer
        days_step: Step in days (e.g. 1 for full daily, 5 for fast training subset)

    Returns:
        Tuple of (surface_series, depth_series, thermocline_series, dates)
    """
    surface_list = []
    depth_list = []
    thermo_list = []
    dates_list = []

    for yr in years:
        for doy in range(1, 366, days_step):
            surf, depth, thermo = synthesizer.generate_day(yr, doy, add_eddy_field=True)
            surface_list.append(surf)
            depth_list.append(depth)
            thermo_list.append(thermo)
            dates_list.append((yr, doy))

    return (
        np.stack(surface_list, axis=0),
        np.stack(depth_list, axis=0),
        np.stack(thermo_list, axis=0),
        dates_list
    )


def create_dataloaders(
    train_years: List[int] = [2016, 2017, 2018, 2019, 2020],
    val_years: List[int] = [2021],
    test_years: List[int] = [2022],
    batch_size: int = 8,
    lag_days: int = 5,
    days_step: int = 5,
    grid: Optional[OceanGrid] = None,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader, ClimatologyComputer, OceanGrid]:
    """Factory creating train, val, and test DataLoaders strictly partitioned by year.

    Args:
        train_years: Whole calendar years for training
        val_years: Whole calendar years for validation
        test_years: Whole calendar years for testing
        batch_size: Batch size
        lag_days: Lag window size
        days_step: Sampling interval in days
        grid: Grid geometry
        seed: Random seed

    Returns:
        Tuple of (train_loader, val_loader, test_loader, fitted_climatology, grid)
    """
    g = grid or OceanGrid()
    synth = PhysicalOceanSynthesizer(g, seed=seed)

    # 1. Generate multi-year sequences
    train_surf, train_depth, train_thermo, train_dates = generate_dataset_split(train_years, g, synth, days_step)
    val_surf, val_depth, val_thermo, val_dates = generate_dataset_split(val_years, g, synth, days_step)
    test_surf, test_depth, test_thermo, test_dates = generate_dataset_split(test_years, g, synth, days_step)

    # 2. Fit Climatology strictly on train split
    train_doys = np.array([doy for _, doy in train_dates])
    clim = ClimatologyComputer(num_days=366)
    clim.fit(train_surf, train_depth, train_doys)

    # 3. Instantiate datasets
    train_ds = OceanEmbedDataset(train_surf, train_depth, train_thermo, train_dates, clim, lag_days, g)
    val_ds = OceanEmbedDataset(val_surf, val_depth, val_thermo, val_dates, clim, lag_days, g)
    test_ds = OceanEmbedDataset(test_surf, test_depth, test_thermo, test_dates, clim, lag_days, g)

    # 4. Create DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, clim, g
