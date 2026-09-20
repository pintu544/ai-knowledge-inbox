// Thin fetch wrapper around the backend. Every non-2xx response is normalised
// into an ApiError so components have a single error shape to render.

import type {
  ApiErrorBody,
  ApiErrorDetailField,
  HealthResponse,
  IngestRequest,
  IngestResponse,
  ItemListResponse,
  QueryRequest,
  QueryResponse,
} from "./types";

// In dev, vite proxies /api -> the FastAPI server. Override with VITE_API_BASE
// for other setups (e.g. a deployed backend).
const API_BASE = import.meta.env.VITE_API_BASE ?? "/api";

export class ApiError extends Error {
  readonly status: number;
  readonly code: string;
  readonly fields: ApiErrorDetailField[];

  constructor(status: number, code: string, message: string, fields: ApiErrorDetailField[] = []) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
    this.fields = fields;
  }
}

function isApiErrorBody(value: unknown): value is ApiErrorBody {
  return (
    typeof value === "object" &&
    value !== null &&
    "error" in value &&
    typeof (value as ApiErrorBody).error?.code === "string"
  );
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE}${path}`, {
      ...init,
      headers: {
        "Content-Type": "application/json",
        ...(init?.headers ?? {}),
      },
    });
  } catch {
    throw new ApiError(0, "network_error", "Could not reach the server. Is the backend running?");
  }

  if (response.ok) {
    if (response.status === 204) return undefined as T;
    return (await response.json()) as T;
  }

  // Try to parse the backend's error envelope; fall back to a generic message.
  let body: unknown;
  try {
    body = await response.json();
  } catch {
    body = null;
  }

  if (isApiErrorBody(body)) {
    const { code, message, details } = body.error;
    throw new ApiError(response.status, code, message, details?.fields ?? []);
  }

  throw new ApiError(response.status, "http_error", `Request failed (${response.status}).`);
}

export const api = {
  health(): Promise<HealthResponse> {
    return request<HealthResponse>("/health");
  },

  listItems(): Promise<ItemListResponse> {
    return request<ItemListResponse>("/items");
  },

  ingest(payload: IngestRequest): Promise<IngestResponse> {
    return request<IngestResponse>("/ingest", {
      method: "POST",
      body: JSON.stringify(payload),
    });
  },

  query(payload: QueryRequest): Promise<QueryResponse> {
    return request<QueryResponse>("/query", {
      method: "POST",
      body: JSON.stringify(payload),
    });
  },
};
