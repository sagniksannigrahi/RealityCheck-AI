import { useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { api, mediaUrl } from "../api/client";
import type { HistoryItem, HistoryStats, Page } from "../api/types";
import { VerdictStack } from "../components/charts";
import { EmptyState, ErrorState, Loading, StatusPill, fmtTime } from "../components/common";

const LIMIT = 15;

export default function HistoryPage() {
  const nav = useNavigate();
  const [page, setPage] = useState<Page<HistoryItem> | null>(null);
  const [stats, setStats] = useState<HistoryStats | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [offset, setOffset] = useState(0);
  const [q, setQ] = useState("");
  const [query, setQuery] = useState("");
  const [status, setStatus] = useState("");
  const [mediaType, setMediaType] = useState("");

  useEffect(() => {
    const t = setTimeout(() => { setQuery(q.trim()); setOffset(0); }, 300);
    return () => clearTimeout(t);
  }, [q]);

  useEffect(() => {
    let alive = true;
    setError(null);
    Promise.all([api.history({ limit: LIMIT, offset, status: status || undefined, media_type: mediaType || undefined, q: query || undefined }),
      api.historyStats()])
      .then(([p, s]) => { if (alive) { setPage(p); setStats(s); } })
      .catch((e) => alive && setError(e));
    return () => { alive = false; };
  }, [offset, query, status, mediaType]);

  return (
    <div className="stack">
      <h1>History</h1>
      {stats && (
        <div className="tiles">
          <div className="tile"><div className="label">Jobs</div><div className="value">{stats.total_jobs}</div>
            <div className="sub">{Object.entries(stats.jobs_by_status).map(([k, v]) => `${v} ${k}`).join(" · ") || "none yet"}</div></div>
          <div className="tile" style={{ gridColumn: "span 2" }}><div className="label">All verified claims</div>
            <div style={{ marginTop: 6 }}><VerdictStack counts={stats.claims_by_verdict} /></div></div>
        </div>
      )}
      <div className="card">
        <div className="row" style={{ marginBottom: 10 }}>
          <input type="search" placeholder="Search descriptions…" aria-label="Search descriptions" value={q} onChange={(e) => setQ(e.target.value)} style={{ maxWidth: 320 }} />
          <select aria-label="Status" value={status} onChange={(e) => { setStatus(e.target.value); setOffset(0); }} style={{ width: "auto" }}>
            <option value="">All statuses</option>
            {["queued", "running", "completed", "failed", "cancelled"].map((s) => <option key={s} value={s}>{s}</option>)}
          </select>
          <select aria-label="Media type" value={mediaType} onChange={(e) => { setMediaType(e.target.value); setOffset(0); }} style={{ width: "auto" }}>
            <option value="">Images & videos</option><option value="image">Images</option><option value="video">Videos</option>
          </select>
        </div>
        {error != null ? <ErrorState error={error} /> : !page ? <Loading /> : page.items.length === 0 ? (
          <EmptyState title={query || status || mediaType ? "No jobs match these filters" : "No verification jobs yet"}>
            <p><Link to="/">Start a verification</Link></p>
          </EmptyState>
        ) : (
          <>
            <div className="table-wrap">
              <table>
                <thead><tr><th></th><th>Description</th><th>Status</th><th>Verdicts</th><th>Media</th><th>Created</th></tr></thead>
                <tbody>
                  {page.items.map((j) => (
                    <tr key={j.job_id} className="clickable" onClick={() => nav(`/jobs/${j.job_id}`)}>
                      <td style={{ width: 56 }}>{j.thumbnail_url ? <img src={mediaUrl(j.thumbnail_url)} alt="" style={{ width: 48, height: 36, objectFit: "cover", borderRadius: 4 }} /> : <span className="tag">{j.media_type}</span>}</td>
                      <td><Link to={`/jobs/${j.job_id}`} onClick={(e) => e.stopPropagation()}>{j.description_preview ?? <em className="muted">(generated description)</em>}</Link>
                        {j.demo_mode && <> <span className="tag warn">demo</span></>}
                        {j.headline && <div className="small muted">{j.headline}</div>}</td>
                      <td><StatusPill status={j.status} /></td>
                      <td style={{ minWidth: 140 }}>{j.verdict_counts ? <VerdictStack counts={j.verdict_counts} compact /> : <span className="muted small">—</span>}</td>
                      <td>{j.media_type}</td>
                      <td className="small num">{fmtTime(j.created_at)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="pager">
              <span className="small muted num">{offset + 1}–{Math.min(offset + LIMIT, page.total)} of {page.total}</span>
              <button disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - LIMIT))}>Previous</button>
              <button disabled={offset + LIMIT >= page.total} onClick={() => setOffset(offset + LIMIT)}>Next</button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
