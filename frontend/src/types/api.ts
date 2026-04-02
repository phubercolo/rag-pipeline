export type Source = {
  filename: string;
  source_path: string;
  chunk_index: number | string;
  score: number;
  text: string;
};

export type AskResponse = {
  answer: string;
  sources: Source[];
};

export type IngestResponse = {
  indexed_chunks: number;
};

export type UploadResponse = {
  filename: string;
  indexed_chunks: number;
};
