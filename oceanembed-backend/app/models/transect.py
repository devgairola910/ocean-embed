from typing import List, Optional
from pydantic import BaseModel, Field
from app.models.predict import LocationPoint, ProfilePoint


class TransectPoint(BaseModel):
    lat: float = Field(..., examples=[12.5])
    lon: float = Field(..., examples=[80.0])
    distance_km: float = Field(..., description="Cumulative distance from start point (km)", examples=[0.0])
    is_ocean: bool = Field(..., description="Whether point is in ocean or land", examples=[True])
    profile: Optional[List[ProfilePoint]] = Field(None, description="Depth profile (null if land)")


class TransectResponse(BaseModel):
    date: str = Field(..., examples=["2025-05-15"])
    n_points: int = Field(..., examples=[50])
    start: LocationPoint
    end: LocationPoint
    points: List[TransectPoint]
    mode: str = Field("mock", examples=["mock"])
