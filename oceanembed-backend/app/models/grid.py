from typing import List, Optional
from pydantic import BaseModel, Field


class GridResponse(BaseModel):
    date: str = Field(..., examples=["2025-05-15"])
    depth_m: float = Field(..., examples=[0.0])
    lats: List[float] = Field(..., description="Latitude axis coordinates")
    lons: List[float] = Field(..., description="Longitude axis coordinates")
    temperature_c: List[List[Optional[float]]] = Field(
        ...,
        description="2D array (lats x lons) of temperatures in °C (null for land cells)"
    )
    mode: str = Field("mock", examples=["mock"])
