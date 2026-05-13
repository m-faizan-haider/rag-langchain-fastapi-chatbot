// lib/api.ts — typed fetch wrapper for the RAG backend

export const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000';

export interface Source {
  filename: string;
  page:     number | null;
  score:    number | null;
}

export interface FactVerification {
  fact:           string;
  tag_fname:      string;
  tag_page:       string | null;
  similarity:     number;
  verbatim_match: boolean;
  matched_preview: string | null;
}

export interface QueryResponse {
  question:   string;
  answer:     string;
  sources:    Source[];
  elapsed_s:  number;
  session_id: string | null;
  cache_hit:  boolean | null;
  extracted_facts_preview: string | null;
  verification: FactVerification[] | null;
}

export interface QueryRequest {
  question:   string;
  session_id?: string | null;
  debug?:     boolean;
  top_k?:     number | null;
}

// ── Non-streaming query ─────────────────────────────────────────────────────
export async function queryRAG(req: QueryRequest): Promise<QueryResponse> {
  const res = await fetch(`${API_BASE}/query`, {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify(req),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? `Request failed: ${res.status}`);
  }
  return res.json();
}

// ── Health check ────────────────────────────────────────────────────────────
export async function checkHealth(): Promise<{ status: string; embedding_model: string }> {
  const res = await fetch(`${API_BASE}/health`);
  return res.json();
}
