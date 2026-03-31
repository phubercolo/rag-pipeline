# Small Local-First RAG Pipeline

This project is a compact retrieval-augmented generation (RAG) pipeline in Python. It loads local documents, chunks them, creates embeddings, stores them in a local FAISS index, retrieves the most relevant chunks for a question, and sends the retrieved context to an LLM to generate a grounded answer.

The code is intentionally split into small modules so you can swap parts later, including replacing the local vector store with an AWS-backed option such as OpenSearch or Bedrock Knowledge Bases.

## Folder Structure

```text
.
|-- .env.example
|-- README.md
|-- main.py
|-- requirements.txt
|-- data/
|   |-- documents/
|   `-- index/
`-- rag_pipeline/
    |-- __init__.py
    |-- chunking.py
    |-- cli.py
    |-- config.py
    |-- embeddings.py
    |-- generation.py
    |-- ingestion.py
    |-- logging_config.py
    |-- models.py
    |-- pipeline.py
    |-- retrieval.py
    `-- vector_store.py
```

## Setup

1. Create and activate a virtual environment.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

2. Install dependencies.

```powershell
pip install -r requirements.txt
```

3. Copy `.env.example` to `.env` and add your `OPENAI_API_KEY`.

4. Put your `.txt`, `.md`, and `.pdf` files into `data/documents/`.

## Run

Build the local index:

```powershell
python main.py ingest
```

Ask one question:

```powershell
python main.py ask "What does the documentation say about deployment?"
```

Start an interactive session:

```powershell
python main.py chat
```

## Architecture

- `ingestion.py` loads files from disk and extracts text from text, markdown, and PDF documents.
- `chunking.py` uses overlapping recursive character splitting to preserve context across chunk boundaries.
- `embeddings.py` wraps the embedding provider behind a small interface so another provider can be added later.
- `vector_store.py` stores vectors locally in FAISS and persists chunk metadata in JSON for source reconstruction.
- `retrieval.py` embeds a query and performs similarity search.
- `generation.py` builds the prompt from retrieved chunks and asks the LLM to answer using only that context.
- `pipeline.py` composes the services into two flows: ingest and answer.
- `cli.py` exposes a simple command-line interface for local use in VS Code or a terminal.

## Environment Variables

- `OPENAI_API_KEY`: API key for embeddings and generation.
- `RAG_DOCUMENTS_DIR`: Folder containing source documents.
- `RAG_INDEX_DIR`: Folder where the FAISS index and chunk metadata are stored.
- `RAG_EMBEDDING_PROVIDER`: Embedding provider name. Currently `openai`.
- `RAG_VECTOR_STORE_BACKEND`: Vector store backend. Currently `faiss`.
- `RAG_LLM_PROVIDER`: LLM provider name. Currently `openai`.
- `RAG_EMBEDDING_MODEL`: Embedding model name.
- `RAG_CHAT_MODEL`: Chat model name.
- `RAG_CHUNK_SIZE`: Chunk size in characters.
- `RAG_CHUNK_OVERLAP`: Chunk overlap in characters.
- `RAG_RETRIEVAL_K`: Number of chunks to retrieve.
- `RAG_LOG_LEVEL`: Logging level.

## End-to-End Flow

1. The `ingest` command scans `data/documents/` for supported files.
2. Extracted text is normalized and split into overlapping chunks.
3. Each chunk is embedded and stored in a local FAISS index, with metadata written to disk alongside it.
4. The `ask` or `chat` command embeds the user query and retrieves the top matching chunks.
5. Those chunks are inserted into the generation prompt.
6. The LLM returns an answer grounded in the retrieved context, and the CLI prints both the answer and the source chunks used.

## AWS-Friendly Extension Points

- Replace `FaissVectorStore` with an implementation that writes to Amazon OpenSearch Serverless or another managed vector database.
- Replace `OpenAIGenerator` and `OpenAIEmbeddingProvider` with Amazon Bedrock adapters while keeping the rest of the pipeline unchanged.

## Next Steps

- Add a reranking stage after retrieval to improve answer quality when documents are long or noisy.
- Add metadata filters, such as filename or document type constraints, before generation.

