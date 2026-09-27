from typing import Optional
from pydantic import BaseModel, Field


class IngestTriggerResponse(BaseModel):
    job_id: str = Field(..., examples=["JOB_a1b2c3d4"])
    status: str = Field("running", examples=["running"])
    started_at: str
    mode: str = Field("mock", examples=["mock"])


class IngestStatusResponse(BaseModel):
    job_id: str = Field(..., examples=["JOB_a1b2c3d4"])
    status: str = Field(..., examples=["succeeded"])
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    error_message: Optional[str] = None
    mode: str = Field("mock", examples=["mock"])
