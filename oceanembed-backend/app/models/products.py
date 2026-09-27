from pydantic import BaseModel, Field
from app.models.predict import LocationPoint


class DepthRange(BaseModel):
    min: float = Field(0.0, examples=[0.0])
    max: float = Field(450.0, examples=[450.0])


class OHCProductResponse(BaseModel):
    location: LocationPoint
    date: str = Field(..., examples=["2025-05-15"])
    ohc_value: float = Field(..., description="Ocean Heat Content value", examples=[3.68])
    unit: str = Field("GJ/m^2", examples=["GJ/m^2"])
    ohc_joules_m2: float = Field(..., description="Ocean Heat Content in Joules per square meter", examples=[3680000000.0])
    reference_temperature_c: float = Field(0.0, examples=[0.0])
    depth_range_m: DepthRange
    mode: str = Field("mock", examples=["mock"])


class MarineHeatwaveResponse(BaseModel):
    location: LocationPoint
    date: str = Field(..., examples=["2025-05-15"])
    is_heatwave: bool = Field(..., examples=[False])
    sst_c: float = Field(..., examples=[28.4])
    climatological_threshold_c: float = Field(29.5, examples=[29.5])
    severity_category: str = Field(..., description="None, Moderate, Strong, Severe, or Extreme", examples=["None"])
    mode: str = Field("mock", examples=["mock"])


class CycloneRiskResponse(BaseModel):
    location: LocationPoint
    date: str = Field(..., examples=["2025-05-15"])
    tchp_score_kj_cm2: float = Field(..., description="Tropical Cyclone Heat Potential in kJ/cm^2", examples=[62.4])
    isotherm_26c_depth_m: float = Field(..., description="Depth of 26°C isotherm in meters", examples=[65.0])
    risk_category: str = Field(..., description="Low, Moderate, or High", examples=["Moderate"])
    mode: str = Field("mock", examples=["mock"])
