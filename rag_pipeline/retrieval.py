from __future__ import annotations

from rag_pipeline.embeddings import EmbeddingProvider
from rag_pipeline.models import RetrievedChunk
from rag_pipeline.vector_store import VectorStore


class Retriever:
    """Coordinates query embedding and vector similarity search."""

    def __init__(self, embedding_provider: EmbeddingProvider, vector_store: VectorStore):
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store

    def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        query_embedding = self.embedding_provider.embed_query(query)
        return self.vector_store.search(query_embedding=query_embedding, k=k)

