from __future__ import annotations

import hashlib
from typing import Iterable

from rag_pipeline.models import Chunk, Document


def _split_text_recursively(
    text: str, chunk_size: int, separators: list[str]
) -> list[str]:
    if len(text) <= chunk_size:
        return [text]
    if not separators:
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    separator = separators[0]
    if separator:
        parts = text.split(separator)
        if len(parts) == 1:
            return _split_text_recursively(text, chunk_size, separators[1:])

        pieces: list[str] = []
        current = ""
        for part in parts:
            candidate = part if not current else f"{current}{separator}{part}"
            if len(candidate) <= chunk_size:
                current = candidate
                continue

            if current:
                pieces.append(current)
            if len(part) <= chunk_size:
                current = part
            else:
                pieces.extend(_split_text_recursively(part, chunk_size, separators[1:]))
                current = ""
        if current:
            pieces.append(current)
        return pieces

    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]


def _add_overlap(chunks: list[str], chunk_overlap: int) -> list[str]:
    if chunk_overlap <= 0 or len(chunks) < 2:
        return [chunk.strip() for chunk in chunks if chunk.strip()]

    overlapped: list[str] = []
    for index, chunk in enumerate(chunks):
        chunk = chunk.strip()
        if not chunk:
            continue
        if index == 0:
            overlapped.append(chunk)
            continue

        previous = overlapped[-1]
        overlap = previous[-chunk_overlap:] if chunk_overlap < len(previous) else previous
        merged = f"{overlap}{chunk}"
        overlapped.append(merged[-(len(chunk) + min(len(overlap), chunk_overlap)) :].strip())
    return overlapped


def split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    separators = ["\n\n", "\n", ". ", " ", ""]
    base_chunks = _split_text_recursively(text, chunk_size, separators)
    return _add_overlap(base_chunks, chunk_overlap)


def chunk_documents(
    documents: Iterable[Document], chunk_size: int, chunk_overlap: int
) -> list[Chunk]:
    """Split documents into overlapping chunks while preserving source metadata."""

    chunks: list[Chunk] = []
    for document in documents:
        split_texts = split_text(document.content, chunk_size, chunk_overlap)
        for index, text in enumerate(split_texts):
            chunk_hash = hashlib.sha1(
                f"{document.doc_id}:{index}:{text}".encode("utf-8")
            ).hexdigest()
            chunks.append(
                Chunk(
                    chunk_id=chunk_hash,
                    doc_id=document.doc_id,
                    text=text,
                    source_path=document.source_path,
                    metadata={**document.metadata, "chunk_index": index},
                )
            )
    return chunks
