from typing import List
from pydantic import BaseModel, Field


class LocationPoint(BaseModel):
    lat: float = Field(..., examples=[12.5])
    lon: float = Field(..., examples=[80.0])


class ProfilePoint(BaseModel):
    depth_m: float = Field(..., examples=[0.0])
    temperature_c: float = Field(..., examples=[28.5])
    uncertainty_c: float = Field(..., examples=[0.2])


class SurfaceInputs(BaseModel):
    sst: float = Field(..., description="Sea Surface Temperature (°C)", examples=[28.5])
    sss: float = Field(..., description="Sea Surface Salinity (PSU)", examples=[34.5])
    sla: float = Field(..., description="Sea Level Anomaly (m)", examples=[0.02])
    u: float = Field(..., description="Zonal Current Velocity (m/s)", examples=[0.1])
    v: float = Field(..., description="Meridional Current Velocity (m/s)", examples=[-0.05])


class PredictResponse(BaseModel):
    requested_location: LocationPoint
    nearest_grid_point: LocationPoint
    date: str = Field(..., examples=["2025-05-15"])
    profile: List[ProfilePoint]
    surface_inputs_used: SurfaceInputs
    mode: str = Field("mock", examples=["mock"])
