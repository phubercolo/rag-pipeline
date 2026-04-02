import { ChangeEvent, FormEvent, useState } from "react";

import { askQuestion, ingestDocuments, uploadDocument } from "./api/rag";
import type { AskResponse } from "./types/api";

const initialAnswer: AskResponse = {
  answer: "",
  sources: [],
};

export default function App() {
  const [query, setQuery] = useState("");
  const [answerResult, setAnswerResult] = useState<AskResponse>(initialAnswer);
  const [statusMessage, setStatusMessage] = useState(
    "API idle. Ingest documents or ask a question.",
  );
  const [errorMessage, setErrorMessage] = useState("");
  const [isAsking, setIsAsking] = useState(false);
  const [isIngesting, setIsIngesting] = useState(false);
  const [isUploading, setIsUploading] = useState(false);

  async function handleIngest() {
    setIsIngesting(true);
    setErrorMessage("");
    setStatusMessage("Ingesting documents into the local vector store...");

    try {
      const response = await ingestDocuments();
      setStatusMessage(`Indexed ${response.indexed_chunks} chunks.`);
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "Ingest request failed.";
      setErrorMessage(message);
      setStatusMessage("Ingest failed.");
    } finally {
      setIsIngesting(false);
    }
  }

  async function handleAsk(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!query.trim()) {
      setErrorMessage("Enter a question before submitting.");
      return;
    }

    setIsAsking(true);
    setErrorMessage("");
    setStatusMessage("Retrieving context and generating an answer...");

    try {
      const response = await askQuestion(query.trim());
      setAnswerResult(response);
      setStatusMessage(
        response.sources.length
          ? `Answered using ${response.sources.length} source chunk(s).`
          : "No relevant indexed context was found.",
      );
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "Ask request failed.";
      setErrorMessage(message);
      setStatusMessage("Question failed.");
    } finally {
      setIsAsking(false);
    }
  }

  async function handleUpload(event: ChangeEvent<HTMLInputElement>) {
    const input = event.currentTarget;
    const file = input.files?.[0];
    if (!file) {
      return;
    }

    setIsUploading(true);
    setErrorMessage("");
    setStatusMessage(`Uploading ${file.name} and ingesting documents...`);

    try {
      const response = await uploadDocument(file);
      setStatusMessage(
        `Uploaded ${response.filename} and indexed ${response.indexed_chunks} chunks.`,
      );
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "Upload request failed.";
      setErrorMessage(message);
      setStatusMessage("Upload failed.");
    } finally {
      input.value = "";
      setIsUploading(false);
    }
  }

  return (
    <main className="app-shell">
      <section className="hero">
        <p className="eyebrow">Local-First RAG</p>
        <h1>Ask questions and manage ingestion from one screen.</h1>
        <p className="hero-copy">
          This frontend talks to the FastAPI wrapper for your existing Python
          pipeline. Use it to index local documents, then query the vector store.
        </p>
      </section>

      <section className="panel action-panel">
        <div>
          <h2>Document Ingestion</h2>
          <p>Upload a file to the documents folder or re-run ingest for everything already on disk.</p>
        </div>
        <div className="action-group">
          <label className="upload-button">
            {isUploading ? "Uploading..." : "Upload and Ingest"}
            <input
              accept=".txt,.md,.pdf"
              className="hidden-input"
              disabled={isUploading}
              onChange={handleUpload}
              type="file"
            />
          </label>
          <button
            className="primary-button"
            onClick={handleIngest}
            disabled={isIngesting}
            type="button"
          >
            {isIngesting ? "Ingesting..." : "Ingest Documents"}
          </button>
        </div>
      </section>

      <section className="panel">
        <h2>Ask a Question</h2>
        <form className="question-form" onSubmit={handleAsk}>
          <label className="field-label" htmlFor="query">
            Question
          </label>
          <textarea
            id="query"
            className="question-input"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="What does the document say about warranty coverage?"
            rows={5}
          />
          <button
            className="primary-button"
            disabled={isAsking}
            type="submit"
          >
            {isAsking ? "Asking..." : "Submit Question"}
          </button>
        </form>
      </section>

      <section className="panel status-panel">
        <h2>Status</h2>
        <p>{statusMessage}</p>
        {errorMessage ? <p className="error-text">{errorMessage}</p> : null}
      </section>

      <section className="panel">
        <h2>Answer</h2>
        <div className="answer-card">
          {answerResult.answer || "No answer yet."}
        </div>
      </section>

      <section className="panel">
        <h2>Sources</h2>
        {answerResult.sources.length ? (
          <div className="source-list">
            {answerResult.sources.map((source) => (
              <article
                className="source-card"
                key={`${source.source_path}-${source.chunk_index}`}
              >
                <div className="source-meta">
                  <strong>{source.filename}</strong>
                  <span>chunk {source.chunk_index}</span>
                  <span>score {source.score.toFixed(4)}</span>
                </div>
                <p>{source.text}</p>
              </article>
            ))}
          </div>
        ) : (
          <p>No sources yet.</p>
        )}
      </section>
    </main>
  );
}
