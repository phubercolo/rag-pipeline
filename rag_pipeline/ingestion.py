from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from pypdf import PdfReader

from rag_pipeline.models import Document

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


def load_documents(documents_dir: Path) -> list[Document]:
    """Load supported files from a folder into normalized Document objects."""

    documents: list[Document] = []

    for path in sorted(documents_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        try:
            text = extract_text(path)
            if not text.strip():
                logger.warning("Skipping empty document: %s", path)
                continue

            doc_hash = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()
            documents.append(
                Document(
                    doc_id=doc_hash,
                    source_path=str(path),
                    content=normalize_whitespace(text),
                    metadata={"extension": path.suffix.lower(), "filename": path.name},
                )
            )
        except Exception as exc:
            logger.exception("Failed to ingest %s: %s", path, exc)

    logger.info("Loaded %s documents from %s", len(documents), documents_dir)
    return documents


def extract_text(path: Path) -> str:
    suffix = path.suffix.lower()

    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="ignore")
    if suffix == ".pdf":
        return extract_pdf_text(path)

    raise ValueError(f"Unsupported file type: {path.suffix}")


def extract_pdf_text(path: Path) -> str:
    """Extract text page by page and ignore missing-text pages gracefully."""

    reader = PdfReader(str(path))
    pages: list[str] = []

    for page_number, page in enumerate(reader.pages, start=1):
        try:
            page_text = page.extract_text() or ""
            if page_text.strip():
                pages.append(page_text)
            else:
                logger.debug("No extractable text on page %s of %s", page_number, path)
        except Exception as exc:
            logger.warning("Failed to read page %s of %s: %s", page_number, path, exc)

    return "\n".join(pages)


def normalize_whitespace(text: str) -> str:
    lines = [line.strip() for line in text.replace("\r\n", "\n").splitlines()]
    collapsed = "\n".join(line for line in lines if line)
    return collapsed.strip()

