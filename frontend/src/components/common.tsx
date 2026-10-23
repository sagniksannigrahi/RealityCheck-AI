import { useCallback, useState, type ReactNode } from "react";
import { ApiError } from "../api/client";
import type { JobStatus, Verdict } from "../api/types";

export const VERDICT_LABEL: Record<Verdict, string> = {
  SUPPORTED: "Supported",
  CONTRADICTED: "Contradicted",
  INSUFFICIENT_EVIDENCE: "Insufficient evidence",
  REVIEW_REQUIRED: "Review required",
  UNVERIFIABLE: "Not visually verifiable",
};

export const VERDICT_ORDER: Verdict[] = ["SUPPORTED", "CONTRADICTED", "REVIEW_REQUIRED", "INSUFFICIENT_EVIDENCE", "UNVERIFIABLE"];

/** Fill color per verdict (status palette); always paired with an icon and a text label. */
export const VERDICT_FILL: Record<Verdict, string> = {
  SUPPORTED: "var(--good)",
  CONTRADICTED: "var(--critical)",
  REVIEW_REQUIRED: "var(--warning)",
  INSUFFICIENT_EVIDENCE: "var(--axis)",
  UNVERIFIABLE: "var(--grid)",
};

export function VerdictIcon({ verdict }: { verdict: Verdict }) {
  const common = { fill: "none", stroke: "currentColor", strokeWidth: 2.4, strokeLinecap: "round" as const, strokeLinejoin: "round" as const };
  switch (verdict) {
    case "SUPPORTED":
      return <svg viewBox="0 0 16 16" aria-hidden="true"><path d="M3 8.5l3.2 3.2L13 5" {...common} /></svg>;
    case "CONTRADICTED":
      return <svg viewBox="0 0 16 16" aria-hidden="true"><path d="M4 4l8 8M12 4l-8 8" {...common} /></svg>;
    case "REVIEW_REQUIRED":
      return <svg viewBox="0 0 16 16" aria-hidden="true"><path d="M8 2.5L14 13H2L8 2.5zM8 7v2.6M8 11.4v.1" {...common} strokeWidth={1.8} /></svg>;
    case "INSUFFICIENT_EVIDENCE":
      return <svg viewBox="0 0 16 16" aria-hidden="true"><circle cx="8" cy="8" r="5.5" {...common} strokeWidth={1.8} /><path d="M5.5 8h5" {...common} /></svg>;
    default:
      return <svg viewBox="0 0 16 16" aria-hidden="true"><circle cx="8" cy="8" r="5.5" {...common} strokeWidth={1.6} strokeDasharray="2 2" /></svg>;
  }
}

export function VerdictBadge({ verdict }: { verdict: Verdict }) {
  return (
    <span className={`badge v-${verdict}`} title={VERDICT_LABEL[verdict]}>
      <VerdictIcon verdict={verdict} />
      {VERDICT_LABEL[verdict]}
    </span>
  );
}

export function StatusPill({ status }: { status: JobStatus }) {
  return (
    <span className={`badge s-${status}`}>
      {(status === "queued" || status === "running") && <span className="spinner" style={{ width: 10, height: 10, borderWidth: 2 }} aria-hidden="true" />}
      {status}
    </span>
  );
}

export function Loading({ label = "Loading…" }: { label?: string }) {
  return (
    <div className="state" role="status" aria-live="polite">
      <span className="spinner" aria-hidden="true" /> <span>{label}</span>
    </div>
  );
}

export function EmptyState({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <div className="state">
      <h3>{title}</h3>
      {children}
    </div>
  );
}

export function ErrorState({ error, onRetry }: { error: unknown; onRetry?: () => void }) {
  const e = error instanceof ApiError ? error : null;
  return (
    <div className="banner error" role="alert">
      <strong>{e ? (e.status === 0 ? "Connection problem" : `Error ${e.status}`) : "Error"}</strong>
      <div>
        <div>{e ? e.message : String((error as Error)?.message ?? error)}</div>
        {e?.code && <div className="small muted">code: <code>{e.code}</code>{e.requestId ? <> · request <code>{e.requestId}</code></> : null}</div>}
        {onRetry && <button className="ghost" onClick={onRetry} style={{ marginTop: 6 }}>Retry</button>}
      </div>
    </div>
  );
}

export const pct = (v: number | null | undefined, d = 1) => (v == null ? "—" : `${(v * 100).toFixed(d)}%`);
export const num = (v: number | null | undefined, d = 3) => (v == null ? "—" : v.toFixed(d));
export const fmtTime = (iso: string | null) => (iso ? new Date(iso).toLocaleString() : "—");
export const fmtSec = (s: number | null | undefined) => (s == null ? "—" : `${s.toFixed(2)}s`);

export interface TipState { x: number; y: number; content: ReactNode }

/** Minimal hover tooltip: follows the pointer, also opened by keyboard focus. */
export function useTooltip() {
  const [tip, setTip] = useState<TipState | null>(null);
  const show = useCallback((e: { clientX: number; clientY: number }, content: ReactNode) => setTip({ x: e.clientX + 12, y: e.clientY + 12, content }), []);
  const showAt = useCallback((el: Element, content: ReactNode) => {
    const r = el.getBoundingClientRect();
    setTip({ x: r.right + 6, y: r.top, content });
  }, []);
  const hide = useCallback(() => setTip(null), []);
  const node = tip ? <div className="tooltip" role="tooltip" style={{ left: Math.min(tip.x, window.innerWidth - 290), top: tip.y }}>{tip.content}</div> : null;
  return { show, showAt, hide, node };
}
