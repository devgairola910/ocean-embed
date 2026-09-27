from typing import List
from pydantic import BaseModel, Field


class DomainBounds(BaseModel):
    lat_min: float = Field(0.0, examples=[0.0])
    lat_max: float = Field(25.0, examples=[25.0])
    lon_min: float = Field(45.0, examples=[45.0])
    lon_max: float = Field(100.0, examples=[100.0])
    resolution_deg: float = Field(0.25, examples=[0.25])


class ValidDateRange(BaseModel):
    start: str = Field("2024-01-01", examples=["2024-01-01"])
    end: str = Field("2026-06-30", examples=["2026-06-30"])


class MetadataResponse(BaseModel):
    domain_bounds: DomainBounds
    depth_levels_m: List[float] = Field(
        ...,
        examples=[[0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 250, 300, 400, 450]]
    )
    valid_date_range: ValidDateRange
    model_version: str = Field("0.1.0-mock", examples=["0.1.0-mock"])
    mode: str = Field("mock", examples=["mock"])
