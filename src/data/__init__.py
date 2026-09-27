"""OceanEmbed data pipeline: Ingestion, regridding, climatology, and dataset classes."""

try:
    from src.data.dataset import OceanEmbedDataset, create_dataloaders
except ImportError:
    OceanEmbedDataset, create_dataloaders = None, None

try:
    from src.data.climatology import ClimatologyComputer
except ImportError:
    ClimatologyComputer = None

from src.data.grid import OceanGrid

__all__ = ["OceanGrid"]
if OceanEmbedDataset is not None:
    __all__.extend(["OceanEmbedDataset", "create_dataloaders"])
if ClimatologyComputer is not None:
    __all__.append("ClimatologyComputer")

