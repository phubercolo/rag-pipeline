from __future__ import annotations

import logging

from rag_pipeline.chunking import chunk_documents
from rag_pipeline.config import Settings
from rag_pipeline.embeddings import OpenAIEmbeddingProvider
from rag_pipeline.generation import OpenAIGenerator
from rag_pipeline.ingestion import load_documents
from rag_pipeline.models import AnswerResult
from rag_pipeline.retrieval import Retriever
from rag_pipeline.vector_store import FaissVectorStore

logger = logging.getLogger(__name__)


class RagPipeline:
    """Application service that exposes ingestion and question-answering flows."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

        if settings.embedding_provider != "openai":
            raise ValueError(
                f"Unsupported embedding provider: {settings.embedding_provider}"
            )
        if settings.vector_store_backend != "faiss":
            raise ValueError(
                f"Unsupported vector store backend: {settings.vector_store_backend}"
            )
        if settings.llm_provider != "openai":
            raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")

        self.embedding_provider = OpenAIEmbeddingProvider(
            api_key=settings.openai_api_key or "",
            model=settings.embedding_model,
        )
        self.vector_store = FaissVectorStore(index_dir=settings.index_dir)
        self.generator = OpenAIGenerator(
            api_key=settings.openai_api_key or "",
            model=settings.chat_model,
        )
        self.retriever = Retriever(self.embedding_provider, self.vector_store)

    def ingest(self) -> int:
        documents = load_documents(self.settings.documents_dir)
        if not documents:
            logger.warning("No supported documents found in %s", self.settings.documents_dir)
            return 0

        chunks = chunk_documents(
            documents,
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )
        embeddings = self.embedding_provider.embed_texts([chunk.text for chunk in chunks])
        self.vector_store.save(chunks, embeddings)
        logger.info("Ingestion complete: %s chunks indexed", len(chunks))
        return len(chunks)

    def answer(self, query: str) -> AnswerResult:
        retrieved = self.retriever.retrieve(query, k=self.settings.retrieval_k)
        if not retrieved:
            return AnswerResult(
                answer="I could not find relevant context in the indexed documents.",
                sources=[],
            )

        answer = self.generator.generate(query=query, retrieved_chunks=retrieved)
        return AnswerResult(answer=answer, sources=retrieved)

