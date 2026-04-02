from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import (
    AskRequest,
    AskResponse,
    HealthResponse,
    IngestResponse,
    SourceResponse,
    UploadResponse,
)
from rag_pipeline.config import get_settings
from rag_pipeline.logging_config import configure_logging
from rag_pipeline.pipeline import RagPipeline

logger = logging.getLogger(__name__)
SUPPORTED_UPLOAD_EXTENSIONS = {".txt", ".md", ".pdf"}

settings = get_settings()
configure_logging(settings.log_level)

app = FastAPI(
    title="Local RAG Pipeline API",
    version="0.1.0",
    description="Thin FastAPI wrapper around the local-first RAG pipeline.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def build_pipeline() -> RagPipeline:
    return RagPipeline(settings)


def _validate_upload_filename(filename: str | None) -> Path:
    if not filename:
        raise HTTPException(status_code=400, detail="Uploaded file is missing a name.")

    safe_name = Path(filename).name
    suffix = Path(safe_name).suffix.lower()
    if suffix not in SUPPORTED_UPLOAD_EXTENSIONS:
        allowed = ", ".join(sorted(SUPPORTED_UPLOAD_EXTENSIONS))
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed extensions: {allowed}.",
        )
    return settings.documents_dir / safe_name


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    return HealthResponse()


@app.post("/ingest", response_model=IngestResponse)
def ingest_documents() -> IngestResponse:
    try:
        indexed_chunks = build_pipeline().ingest()
    except Exception as exc:
        logger.exception("Ingest request failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return IngestResponse(indexed_chunks=indexed_chunks)


@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)) -> UploadResponse:
    target_path = _validate_upload_filename(file.filename)

    try:
        content = await file.read()
        target_path.write_bytes(content)
        indexed_chunks = build_pipeline().ingest()
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Upload request failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        await file.close()

    return UploadResponse(filename=target_path.name, indexed_chunks=indexed_chunks)


@app.post("/ask", response_model=AskResponse)
def ask_question(payload: AskRequest) -> AskResponse:
    try:
        result = build_pipeline().answer(payload.query)
    except Exception as exc:
        logger.exception("Ask request failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    sources = [
        SourceResponse(
            filename=item.chunk.metadata.get("filename", item.chunk.source_path),
            source_path=item.chunk.source_path,
            chunk_index=item.chunk.metadata.get("chunk_index", "?"),
            score=item.score,
            text=item.chunk.text,
        )
        for item in result.sources
    ]
    return AskResponse(answer=result.answer, sources=sources)
