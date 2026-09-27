from pydantic import BaseModel, Field


class ErrorDetail(BaseModel):
    code: str = Field(..., description="Error code identifier", examples=["OUT_OF_DOMAIN"])
    message: str = Field(..., description="Human readable error message", examples=["Location is outside domain or on land."])


class ErrorResponse(BaseModel):
    error: ErrorDetail
