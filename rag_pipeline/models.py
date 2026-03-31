from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Document:
    """Raw document loaded from disk before chunking."""

    doc_id: str
    source_path: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Chunk:
    """Normalized chunk persisted in the vector index sidecar metadata."""

    chunk_id: str
    doc_id: str
    text: str
    source_path: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RetrievedChunk:
    """Chunk returned by retrieval with a similarity score."""

    chunk: Chunk
    score: float


@dataclass(slots=True)
class AnswerResult:
    """Final RAG response returned to callers and the CLI."""

    answer: str
    sources: list[RetrievedChunk]

