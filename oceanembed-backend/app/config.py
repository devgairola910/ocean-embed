import os
from functools import lru_cache
from typing import List, Union
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    MODE: str = "mock"
    CORS_ORIGINS: Union[List[str], str] = "http://localhost:3000,http://localhost:8501,http://localhost:5173"
    API_V1_PREFIX: str = "/api/v1"
    MODEL_PATH: str = "models_store"
    DATA_OUTPUT_PATH: str = "/tmp/data/outputs" if os.environ.get("VERCEL") else "data/outputs"
    DATABASE_URL: str = "sqlite:////tmp/oceanembed.db" if os.environ.get("VERCEL") else "sqlite:///./oceanembed.db"
    INGESTION_INTERVAL_HOURS: int = 24
    NORMALIZATION_STATS_PATH: str = "models_store/normalization_stats.json"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        if isinstance(v, str):
            if not v:
                return []
            return [i.strip() for i in v.split(",") if i.strip()]
        return v


@lru_cache()
def get_settings() -> Settings:
    return Settings()
