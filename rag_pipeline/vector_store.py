from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Protocol

import faiss
import numpy as np

from rag_pipeline.models import Chunk, RetrievedChunk

logger = logging.getLogger(__name__)


class VectorStore(Protocol):
    def save(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        ...

    def search(self, query_embedding: list[float], k: int) -> list[RetrievedChunk]:
        ...

    def exists(self) -> bool:
        ...


class FaissVectorStore:
    """Local vector store with a FAISS index and JSON metadata sidecar."""

    def __init__(self, index_dir: Path) -> None:
        self.index_dir = index_dir
        self.index_path = index_dir / "chunks.index"
        self.metadata_path = index_dir / "chunks.json"

    def save(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            raise ValueError("No chunks available to store.")
        if len(chunks) != len(embeddings):
            raise ValueError("Chunk and embedding counts must match.")

        matrix = np.array(embeddings, dtype="float32")
        faiss.normalize_L2(matrix)

        index = faiss.IndexFlatIP(matrix.shape[1])
        index.add(matrix)
        faiss.write_index(index, str(self.index_path))

        payload = [
            {
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "text": chunk.text,
                "source_path": chunk.source_path,
                "metadata": chunk.metadata,
            }
            for chunk in chunks
        ]
        self.metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("Saved %s chunks to %s", len(chunks), self.index_dir)

    def search(self, query_embedding: list[float], k: int) -> list[RetrievedChunk]:
        if not self.exists():
            raise FileNotFoundError(
                "Vector index not found. Run the ingest command first."
            )

        index = faiss.read_index(str(self.index_path))
        metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))

        query_matrix = np.array([query_embedding], dtype="float32")
        faiss.normalize_L2(query_matrix)

        scores, indices = index.search(query_matrix, k)
        results: list[RetrievedChunk] = []

        for score, idx in zip(scores[0], indices[0], strict=False):
            if idx < 0:
                continue
            item = metadata[idx]
            chunk = Chunk(
                chunk_id=item["chunk_id"],
                doc_id=item["doc_id"],
                text=item["text"],
                source_path=item["source_path"],
                metadata=item.get("metadata", {}),
            )
            results.append(RetrievedChunk(chunk=chunk, score=float(score)))

        return results

    def exists(self) -> bool:
        return self.index_path.exists() and self.metadata_path.exists()

