import { config } from "../config";
import type {
  Collection,
  DocumentItem,
  DocumentPage,
  Figure,
  Me,
  Plan,
  SearchRequest,
  SearchResponse,
  UploadResponse,
} from "./types";

/** Error returned by the API, with the server's message and optional machine-readable code. */
export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
    public code?: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

export type TokenProvider = () => Promise<string | null>;

let tokenProvider: TokenProvider = async () => null;

/** Registered by the auth context so every request carries a fresh Firebase ID token. */
export function setTokenProvider(provider: TokenProvider) {
  tokenProvider = provider;
}

function errorFromBody(status: number, body: unknown): ApiError {
  const detail = (body as { detail?: unknown } | null)?.detail;
  if (typeof detail === "string") return new ApiError(status, detail);
  if (detail && typeof detail === "object" && "message" in detail) {
    const d = detail as { message: string; code?: string };
    return new ApiError(status, d.message, d.code);
  }
  if (Array.isArray(detail) && detail[0]?.msg) return new ApiError(status, String(detail[0].msg));
  return new ApiError(status, `Request failed (${status})`);
}

export async function request<T>(path: string, init: RequestInit & { json?: unknown } = {}): Promise<T> {
  const headers = new Headers(init.headers);
  const token = await tokenProvider();
  if (token) headers.set("Authorization", `Bearer ${token}`);
  let body = init.body;
  if (init.json !== undefined) {
    headers.set("Content-Type", "application/json");
    body = JSON.stringify(init.json);
  }

  let response: Response;
  try {
    response = await fetch(`${config.apiBaseUrl}/api${path}`, { ...init, headers, body });
  } catch {
    throw new ApiError(0, "Can't reach the server. Check your connection and try again.");
  }
  if (response.status === 204) return undefined as T;
  const data = await response.json().catch(() => null);
  if (!response.ok) throw errorFromBody(response.status, data);
  return data as T;
}

export const api = {
  plans: () => request<Plan[]>("/plans"),
  me: () => request<Me>("/me"),
  updateMe: (display_name: string) => request<Me>("/me", { method: "PATCH", json: { display_name } }),
  deleteMe: () => request<void>("/me", { method: "DELETE" }),

  collections: () => request<Collection[]>("/collections"),
  createCollection: (name: string) => request<Collection>("/collections", { method: "POST", json: { name } }),
  renameCollection: (id: string, name: string) =>
    request<Collection>(`/collections/${id}`, { method: "PATCH", json: { name } }),
  deleteCollection: (id: string) => request<void>(`/collections/${id}`, { method: "DELETE" }),

  documents: (params: { collection?: string; q?: string; limit?: number; offset?: number } = {}) => {
    const query = new URLSearchParams();
    for (const [k, v] of Object.entries(params)) if (v !== undefined && v !== "") query.set(k, String(v));
    return request<DocumentPage>(`/documents?${query}`);
  },
  startUpload: (body: { filename: string; size_bytes: number; sha256: string; collection_id?: string | null }) =>
    request<UploadResponse>("/documents/uploads", { method: "POST", json: body }),
  completeUpload: (id: string) => request<DocumentItem>(`/documents/${id}/complete`, { method: "POST" }),
  updateDocument: (id: string, body: { title?: string | null; collection_id?: string | null }) =>
    request<DocumentItem>(`/documents/${id}`, { method: "PATCH", json: body }),
  deleteDocument: (id: string) => request<void>(`/documents/${id}`, { method: "DELETE" }),
  retryDocument: (id: string) => request<DocumentItem>(`/documents/${id}/retry`, { method: "POST" }),
  downloadUrl: (id: string) => request<{ url: string }>(`/documents/${id}/download`),
  figures: (id: string) => request<Figure[]>(`/documents/${id}/figures`),

  search: (body: SearchRequest) => request<SearchResponse>("/search", { method: "POST", json: body }),

  checkout: () => request<{ url: string }>("/billing/checkout", { method: "POST" }),
  billingPortal: () => request<{ url: string }>("/billing/portal", { method: "POST" }),
};
