"""
schema/requests.py
------------------
Pydantic models for all API request payloads and response bodies.
Centralizing schemas here keeps routers clean and enables easy reuse.
"""
from pydantic import BaseModel, Field


class LoadModelRequest(BaseModel):
    model: str = Field(
        ...,
        description="spaCy model name to download and register.",
        examples=["en_core_web_sm", "en_core_web_md", "en_core_web_lg"],
    )


class LoadModelResponse(BaseModel):
    model: str
    status: str  # "downloaded" | "already_available"
    message: str


class ModelInfo(BaseModel):
    name: str
    status: str  # "loaded" | "available"


class ListModelsResponse(BaseModel):
    models: list[ModelInfo]