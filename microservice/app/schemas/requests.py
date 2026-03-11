"""
schema/requests.py
------------------
Pydantic models for all API request payloads and response bodies.
Centralizing schemas here keeps routers clean and enables easy reuse.
"""
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


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
    loaded: bool = False
    run_id: Optional[str] = None


class LoadModelResponse(BaseModel):
    model: str
    status: str
    message: str
    mlflow_version: Optional[str] = None
    mlflow_stage: Optional[str] = None
    mlflow_run_id: Optional[str] = None


class ModelInfo(BaseModel):
    name: str
    status: str = "—"
    version: str = "—"
    stage: str = "—"
    loaded: bool = False 
    run_id: Optional[str] = None


class ListModelsResponse(BaseModel):
    models: list[ModelInfo]


class EntityResult(BaseModel):
    label: str = Field(description="NER entity label (e.g. PERSON, MONEY, DATE).")
    text: str = Field(description="Extracted entity text.")
    start: int = Field(description="Character start offset in the input text.")
    end: int = Field(description="Character end offset in the input text.")


class PredictionRecord(BaseModel):
    id: str
    input_text: str
    output: list[EntityResult]
    model: str
    timestamp: datetime


class ListPredictionsResponse(BaseModel):
    total: int
    predictions: list[PredictionRecord]


class PredictResponse(BaseModel):
    entities: list[EntityResult]
    record_id: str


class PredictRequest(BaseModel):
    text: str = Field(
        ...,
        description="Text (in English) to run NER inference on.",
        examples=["Can you send $45 to Michael on June 3?"],
    )
    model: str = Field(
        ...,
        description="spaCy model name to use for inference.",
        examples=["en_core_web_sm"],
    )
    

class HealthResponse(BaseModel):
    status: str
    loaded_models: list[str]
    total_predictions: int
    history_backend: str
    mongodb_connected: bool 
    

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None