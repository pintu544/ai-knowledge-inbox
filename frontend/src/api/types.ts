// Types mirroring the backend Pydantic contract (see backend/app/models.py).

export type SourceType = "note" | "url";

export interface IngestRequest {
  source_type: SourceType;
  content: string;
  title?: string | null;
}

export interface ItemSummary {
  id: string;
  source_type: SourceType;
  title: string;
  source_url: string | null;
  created_at: string;
  chunk_count: number;
  char_count: number;
  preview: string;
}

export interface IngestResponse {
  item: ItemSummary;
  chunk_count: number;
}

export interface ItemListResponse {
  items: ItemSummary[];
  count: number;
}

export interface SourceSnippet {
  citation: number;
  item_id: string;
  source_type: SourceType;
  title: string;
  source_url: string | null;
  chunk_index: number;
  score: number;
  snippet: string;
}

export interface QueryRequest {
  question: string;
  top_k?: number | null;
}

export interface QueryResponse {
  question: string;
  answer: string;
  sources: SourceSnippet[];
  model: string;
  retrieved_chunk_count: number;
}

export interface ModelStatus {
  configured: string;
  in_use: string;
  validated: boolean;
  detail: string | null;
}

export interface HealthResponse {
  status: string;
  chat_model: ModelStatus;
  embedding_model: string;
}

// The backend's single error envelope:
//   { "error": { "code": "...", "message": "...", "details": {...}? } }
export interface ApiErrorDetailField {
  field: string;
  message: string;
}

export interface ApiErrorBody {
  error: {
    code: string;
    message: string;
    details?: {
      fields?: ApiErrorDetailField[];
      [key: string]: unknown;
    };
  };
}
