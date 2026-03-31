from __future__ import annotations

import hashlib
from typing import Iterable

from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_pipeline.models import Chunk, Document


def chunk_documents(
    documents: Iterable[Document], chunk_size: int, chunk_overlap: int
) -> list[Chunk]:
    """Split documents into overlapping chunks while preserving source metadata."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[Chunk] = []
    for document in documents:
        split_texts = splitter.split_text(document.content)
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

