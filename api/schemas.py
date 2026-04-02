from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"


class IngestResponse(BaseModel):
    indexed_chunks: int


class UploadResponse(BaseModel):
    filename: str
    indexed_chunks: int


class AskRequest(BaseModel):
    query: str = Field(min_length=1, description="User question to answer")


class SourceResponse(BaseModel):
    filename: str
    source_path: str
    chunk_index: int | str
    score: float
    text: str


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceResponse]
