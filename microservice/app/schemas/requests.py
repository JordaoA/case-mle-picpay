"""
schema/requests.py
------------------
Pydantic models for all API request payloads and response bodies.
Centralizing schemas here keeps routers clean and enables easy reuse.
"""
from pydantic import BaseModel, Field
from typing import Optional

class LoadModelRequest(BaseModel):
    model: str = Field(
        ...,
        description="spaCy model name to download and register.",
        examples=["en_core_web_sm", "en_core_web_md", "en_core_web_lg"],
    )


class ModelInfo(BaseModel):
    name: str
    version: str = "—"
    stage: str = "—"
    loaded: bool = False              # True if currently in memory cache
    run_id: Optional[str] = None


class LoadModelResponse(BaseModel):
    model: str
    status: str  # "downloaded" | "already_available"
    message: str


class ModelInfo(BaseModel):
    name: str
    status: str  # "loaded" | "available"


class ListModelsResponse(BaseModel):
    models: list[ModelInfo]