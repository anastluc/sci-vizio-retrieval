// Mirrors sci_vizio_saas/schemas.py.

export interface Plan {
  id: "free" | "premium";
  name: string;
  price_monthly_usd: number;
  max_documents: number;
  max_file_mb: number;
  max_figures_per_document: number;
}

export interface Me {
  id: string;
  email: string | null;
  display_name: string | null;
  plan: Plan;
  usage: { documents: number; figures: number; storage_bytes: number };
  subscription_status: string | null;
  current_period_end: string | null;
  billing_enabled: boolean;
  created_at: string;
}

export interface Collection {
  id: string;
  name: string;
  description: string | null;
  document_count: number;
  created_at: string;
}

export type DocumentStatus = "uploading" | "queued" | "processing" | "ready" | "failed";

export interface DocumentItem {
  id: string;
  filename: string;
  title: string | null;
  collection_id: string | null;
  status: DocumentStatus;
  error: string | null;
  size_bytes: number;
  page_count: number | null;
  figure_count: number;
  skipped_figure_count: number;
  created_at: string;
  processed_at: string | null;
}

export interface DocumentPage {
  items: DocumentItem[];
  total: number;
}

export interface UploadResponse {
  document: DocumentItem;
  upload: { url: string; method: string; headers: Record<string, string> };
}

export interface Figure {
  id: string;
  document_id: string;
  page: number;
  figure_type: string;
  image_type: string | null;
  title: string | null;
  description: string | null;
  analysis: Record<string, unknown>;
  width: number;
  height: number;
  image_url: string;
}

export interface SearchHit extends Figure {
  score: number | null;
  document: { id: string; filename: string; title: string | null };
  collection: { id: string; name: string } | null;
}

export interface FacetValue {
  value: string;
  label: string;
  count: number;
}

export interface SearchRequest {
  query?: string;
  collections?: string[];
  documents?: string[];
  figure_types?: string[];
  include_decorative?: boolean;
  limit?: number;
  offset?: number;
}

export interface SearchResponse {
  total: number;
  results: SearchHit[];
  facets: { figure_types: FacetValue[]; collections: FacetValue[]; documents: FacetValue[] };
}

export const UNFILED = "unfiled";
