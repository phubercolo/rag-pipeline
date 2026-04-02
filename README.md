# RAG Pipeline

Supports local document ingestion for `.txt`, `.md`, and `.pdf` files, with optional OCR for scanned or image-based PDFs.

This repository contains a retrieval-augmented generation (RAG) pipeline in Python. It ingests local documents, chunks content, creates embeddings, stores them in a FAISS index, retrieves the most relevant context for a question, and sends that context to an LLM to generate grounded answers.

The system is organized into modular components so you can evolve the architecture over time, including replacing the local vector store with an AWS-backed option such as OpenSearch or Bedrock Knowledge Bases.
It supports text, markdown, and PDF files. For scanned or image-based PDFs, you can enable OCR with Tesseract.

## Folder Structure

```text
.
|-- api/
|   |-- __init__.py
|   |-- main.py
|   `-- schemas.py
|-- .env.example
|-- frontend/
|   |-- index.html
|   |-- package.json
|   |-- tsconfig.app.json
|   |-- tsconfig.json
|   |-- tsconfig.node.json
|   |-- vite.config.ts
|   `-- src/
|       |-- api/
|       |   |-- client.ts
|       |   `-- rag.ts
|       |-- types/
|       |   `-- api.ts
|       |-- App.tsx
|       |-- main.tsx
|       `-- styles.css
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

If you want the React frontend too, install its dependencies:

```powershell
cd frontend
npm install
cd ..
```

On Windows PowerShell, if `npm` is blocked by execution policy, either reopen the terminal after setting:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

or use:

```powershell
npm.cmd install
```

3. Optional but recommended for scanned PDFs: install Tesseract OCR.

On Windows, install Tesseract and either add it to `PATH` or set `TESSERACT_CMD` in `.env`.

4. Copy `.env.example` to `.env` and add your `OPENAI_API_KEY`.

5. Put your `.txt`, `.md`, and `.pdf` files into `data/documents/`.

6. If you want OCR for scanned PDFs, set this in `.env`:

```env
RAG_ENABLE_OCR=true
```

## Run

Build the local index:

```powershell
python main.py ingest
```

If OCR is enabled, scanned PDFs without an embedded text layer will be sent through Tesseract during ingestion.

Ask one question:

```powershell
python main.py ask "What does the documentation say about deployment?"
```

Start an interactive session:

```powershell
python main.py chat
```

Run the FastAPI backend:

```powershell
uvicorn api.main:app --reload
```

Run the React frontend:

```powershell
cd frontend
npm run dev
```

If PowerShell blocks `npm`, use:

```powershell
npm.cmd run dev
```

The frontend defaults to `http://127.0.0.1:8000` for the API. To point it somewhere else, create `frontend/.env` with:

```env
VITE_API_BASE_URL=http://127.0.0.1:8000
```

From the browser UI, you can:
- Upload a `.txt`, `.md`, or `.pdf` file directly into `data/documents/`
- Automatically trigger ingest right after upload
- Re-run ingest for all documents already on disk
- Ask questions against the current local index

## Architecture

- `ingestion.py` loads files from disk and extracts text from text, markdown, and PDF documents.
- `chunking.py` uses overlapping recursive character splitting to preserve context across chunk boundaries.
- `embeddings.py` wraps the embedding provider behind a clean interface so another provider can be added later.
- `vector_store.py` stores vectors locally in FAISS and persists chunk metadata in JSON for source reconstruction.
- `retrieval.py` embeds a query and performs similarity search.
- `generation.py` builds the prompt from retrieved chunks and asks the LLM to answer using only that context.
- `pipeline.py` composes the services into two flows: ingest and answer.
- `cli.py` exposes a command-line interface for local use in VS Code or a terminal.
- `api/main.py` exposes `health`, `ingest`, `upload`, and `ask` endpoints for the UI.
- `frontend/` contains a React + TypeScript app that calls the FastAPI layer.

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
- `RAG_ENABLE_OCR`: Enables OCR fallback for scanned PDFs when embedded text is missing.
- `TESSERACT_CMD`: Optional full path to the Tesseract executable if it is not on `PATH`.

## End-to-End Flow

1. The `ingest` command scans `data/documents/` for supported files.
2. The browser upload flow saves a selected file into `data/documents/` and then runs the same ingest pipeline.
3. Extracted text is normalized and split into overlapping chunks. If a PDF has no embedded text and OCR is enabled, the pipeline tries Tesseract OCR.
4. Each chunk is embedded and stored in a local FAISS index, with metadata written to disk alongside it.
5. The `ask` or `chat` command embeds the user query and retrieves the top matching chunks.
6. Those chunks are inserted into the generation prompt.
7. The LLM returns an answer grounded in the retrieved context, and the CLI or frontend prints both the answer and the source chunks used.

## PDF Notes

- Text-based PDFs work with `pypdf`.
- Scanned PDFs usually need OCR.
- When a PDF has no extractable text, the log now says so explicitly instead of calling the file "empty."
- To enable OCR, set `RAG_ENABLE_OCR=true` in `.env`.
- If Tesseract is not on `PATH`, set `TESSERACT_CMD` to the full path of `tesseract.exe`.

## AWS-Friendly Extension Points

- Replace `FaissVectorStore` with an implementation that writes to Amazon OpenSearch Serverless or another managed vector database.
- Replace `OpenAIGenerator` and `OpenAIEmbeddingProvider` with Amazon Bedrock adapters while keeping the rest of the pipeline unchanged.

## Next Steps

- Add a reranking stage after retrieval to improve answer quality when documents are long or noisy.
- Add metadata filters, such as filename or document type constraints, before generation.
