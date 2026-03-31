from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


@dataclass(slots=True)
class Settings:
    """Centralized runtime configuration loaded from environment variables."""

    documents_dir: Path = Path(os.getenv("RAG_DOCUMENTS_DIR", "data/documents"))
    index_dir: Path = Path(os.getenv("RAG_INDEX_DIR", "data/index"))
    embedding_provider: str = os.getenv("RAG_EMBEDDING_PROVIDER", "openai")
    vector_store_backend: str = os.getenv("RAG_VECTOR_STORE_BACKEND", "faiss")
    llm_provider: str = os.getenv("RAG_LLM_PROVIDER", "openai")
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    embedding_model: str = os.getenv("RAG_EMBEDDING_MODEL", "text-embedding-3-small")
    chat_model: str = os.getenv("RAG_CHAT_MODEL", "gpt-4o-mini")
    chunk_size: int = int(os.getenv("RAG_CHUNK_SIZE", "900"))
    chunk_overlap: int = int(os.getenv("RAG_CHUNK_OVERLAP", "150"))
    retrieval_k: int = int(os.getenv("RAG_RETRIEVAL_K", "4"))
    log_level: str = os.getenv("RAG_LOG_LEVEL", "INFO")
    enable_ocr: bool = os.getenv("RAG_ENABLE_OCR", "false").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    tesseract_cmd: str | None = os.getenv("TESSERACT_CMD")

    def ensure_directories(self) -> None:
        self.documents_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)


def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_directories()
    return settings
