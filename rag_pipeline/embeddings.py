from __future__ import annotations

import logging
from typing import Protocol

from openai import OpenAI

logger = logging.getLogger(__name__)


class EmbeddingProvider(Protocol):
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        ...

    def embed_query(self, text: str) -> list[float]:
        ...


class OpenAIEmbeddingProvider:
    """Embedding adapter isolated behind a small interface for future swaps."""

    def __init__(self, api_key: str, model: str) -> None:
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAI embeddings.")
        self._client = OpenAI(api_key=api_key)
        self._model = model

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        logger.info("Creating embeddings for %s chunks", len(texts))
        response = self._client.embeddings.create(model=self._model, input=texts)
        return [item.embedding for item in response.data]

    def embed_query(self, text: str) -> list[float]:
        response = self._client.embeddings.create(model=self._model, input=[text])
        return response.data[0].embedding

