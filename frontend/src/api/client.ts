import type {
  ClaimOut, EvaluationList, Evidence, Frame, HistoryItem, HistoryStats, JobSettings, JobSummary, Mode, Page, SystemInfo,
} from "./types";

const BASE = (import.meta.env.VITE_API_BASE as string | undefined) ?? "";
const KEY_STORAGE = "rc.apiKey";

export class ApiError extends Error {
  constructor(public status: number, public code: string, message: string, public requestId?: string | null) {
    super(message);
  }
}

export function getApiKey(): string {
  try { return localStorage.getItem(KEY_STORAGE) ?? ""; } catch { return ""; }
}

export function setApiKey(v: string): void {
  try { if (v) localStorage.setItem(KEY_STORAGE, v); else localStorage.removeItem(KEY_STORAGE); } catch { /* storage unavailable */ }
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers);
  const key = getApiKey();
  if (key) headers.set("X-API-Key", key);
  let res: Response;
  try {
    res = await fetch(BASE + path, { ...init, headers });
  } catch {
    throw new ApiError(0, "network_error", "Cannot reach the RealityCheck API. Is the backend running?");
  }
  if (!res.ok) {
    let code = "http_error", message = `Request failed (${res.status})`, rid: string | null = null;
    try {
      const body = await res.json();
      code = body?.error?.code ?? code; message = body?.error?.message ?? message; rid = body?.error?.request_id ?? null;
    } catch { /* non-JSON error */ }
    throw new ApiError(res.status, code, message, rid ?? res.headers.get("X-Request-ID"));
  }
  if (res.status === 204) return undefined as T;
  const ct = res.headers.get("content-type") ?? "";
  return (ct.includes("application/json") ? res.json() : res.text()) as Promise<T>;
}

/** Signed media URLs are root-relative paths returned by the API. */
export const mediaUrl = (u: string | null | undefined) => (u ? BASE + u : undefined);

export const api = {
  systemInfo: () => request<SystemInfo>("/api/v1/system/info"),
  createVerification: (file: File, description: string, mode: Mode, settings: JobSettings) => {
    const fd = new FormData();
    fd.append("file", file);
    fd.append("description", description);
    fd.append("mode", mode);
    fd.append("settings", JSON.stringify(settings));
    return request<{ job_id: string; status: string; demo_mode: boolean }>("/api/v1/verifications", { method: "POST", body: fd });
  },
  job: (id: string) => request<JobSummary>(`/api/v1/verifications/${encodeURIComponent(id)}`),
  claims: (id: string) => request<Page<ClaimOut>>(`/api/v1/verifications/${encodeURIComponent(id)}/claims?limit=200`),
  evidence: (id: string) => request<Page<Evidence>>(`/api/v1/verifications/${encodeURIComponent(id)}/evidence?limit=1000`),
  frames: (id: string) => request<Frame[]>(`/api/v1/verifications/${encodeURIComponent(id)}/frames`),
  reportJson: (id: string) => request<unknown>(`/api/v1/verifications/${encodeURIComponent(id)}/report`),
  reportMarkdown: (id: string) => request<string>(`/api/v1/verifications/${encodeURIComponent(id)}/report?format=markdown`),
  cancel: (id: string) => request<JobSummary>(`/api/v1/verifications/${encodeURIComponent(id)}/cancel`, { method: "POST" }),
  remove: (id: string) => request<void>(`/api/v1/verifications/${encodeURIComponent(id)}`, { method: "DELETE" }),
  history: (q: { limit: number; offset: number; status?: string; media_type?: string; q?: string }) => {
    const p = new URLSearchParams({ limit: String(q.limit), offset: String(q.offset) });
    if (q.status) p.set("status", q.status);
    if (q.media_type) p.set("media_type", q.media_type);
    if (q.q) p.set("q", q.q);
    return request<Page<HistoryItem>>(`/api/v1/history?${p}`);
  },
  historyStats: () => request<HistoryStats>("/api/v1/history/stats"),
  evaluations: (experiment?: string) =>
    request<EvaluationList>(`/api/v1/evaluations${experiment ? `?experiment=${encodeURIComponent(experiment)}` : ""}`),
  health: () => request<{ status: string; checks: Record<string, string> }>("/api/v1/health/ready"),
};
