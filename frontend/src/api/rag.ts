import { apiRequest } from "./client";
import type { AskResponse, IngestResponse, UploadResponse } from "../types/api";

export function ingestDocuments(): Promise<IngestResponse> {
  return apiRequest<IngestResponse>("/ingest", { method: "POST" });
}

export function askQuestion(query: string): Promise<AskResponse> {
  return apiRequest<AskResponse>("/ask", {
    method: "POST",
    body: { query },
  });
}

export function uploadDocument(file: File): Promise<UploadResponse> {
  const formData = new FormData();
  formData.append("file", file);

  return apiRequest<UploadResponse>("/upload", {
    method: "POST",
    formData,
  });
}
