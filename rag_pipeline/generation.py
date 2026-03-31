from __future__ import annotations

import logging
from typing import Protocol

from openai import OpenAI

from rag_pipeline.models import RetrievedChunk

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are a helpful assistant for question answering over local documents.
Use only the retrieved context when answering.
If the answer is not supported by the context, say you do not have enough information.
Be concise and cite the source file names in your answer when helpful."""


class Generator(Protocol):
    def generate(self, query: str, retrieved_chunks: list[RetrievedChunk]) -> str:
        ...


class OpenAIGenerator:
    """Simple generation adapter that can be replaced later by Bedrock or another LLM."""

    def __init__(self, api_key: str, model: str) -> None:
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAI generation.")
        self._client = OpenAI(api_key=api_key)
        self._model = model

    def generate(self, query: str, retrieved_chunks: list[RetrievedChunk]) -> str:
        context = build_context_block(retrieved_chunks)
        prompt = f"""Question:
{query}

Retrieved context:
{context}

Answer using the retrieved context only."""

        response = self._client.chat.completions.create(
            model=self._model,
            temperature=0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content or ""


def build_context_block(retrieved_chunks: list[RetrievedChunk]) -> str:
    parts: list[str] = []
    for index, item in enumerate(retrieved_chunks, start=1):
        source_name = item.chunk.metadata.get("filename", item.chunk.source_path)
        parts.append(
            f"[Source {index}] file={source_name} score={item.score:.4f}\n{item.chunk.text}"
        )
    return "\n\n".join(parts)

