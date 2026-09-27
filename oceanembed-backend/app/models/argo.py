from typing import List
from pydantic import BaseModel, Field
from app.models.predict import LocationPoint, ProfilePoint


class DepthValidation(BaseModel):
    depth_m: float = Field(..., examples=[0.0])
    rmse_c: float = Field(..., examples=[0.25])
    correlation: float = Field(..., examples=[0.98])


class OverallValidation(BaseModel):
    rmse_c: float = Field(..., examples=[0.42])
    correlation: float = Field(..., examples=[0.96])


class ArgoValidationResponse(BaseModel):
    date_from: str = Field(..., examples=["2024-01-01"])
    date_to: str = Field(..., examples=["2026-06-30"])
    n_matchups: int = Field(..., examples=[1450])
    overall: OverallValidation
    by_depth: List[DepthValidation]
    mode: str = Field("mock", examples=["mock"])


class ArgoFloatProfile(BaseModel):
    float_id: str = Field(..., examples=["ARGO_2901482"])
    lat: float = Field(..., examples=[12.6])
    lon: float = Field(..., examples=[80.1])
    date: str = Field(..., examples=["2025-05-14"])
    distance_km: float = Field(..., examples=[15.2])
    profile: List[ProfilePoint]


class ArgoProfilesResponse(BaseModel):
    requested_location: LocationPoint
    radius_km: float = Field(..., examples=[100.0])
    date: str = Field(..., examples=["2025-05-15"])
    profiles: List[ArgoFloatProfile]
    mode: str = Field("mock", examples=["mock"])
