import { useCallback, useEffect, useMemo, useState, type ReactNode } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { api, mediaUrl } from "../api/client";
import type { ClaimOut, Evidence, Frame, JobSummary, Verdict } from "../api/types";
import { VerdictStack } from "../components/charts";
import { ErrorState, Loading, StatusPill, VERDICT_LABEL, VERDICT_ORDER, VerdictBadge, fmtSec, fmtTime, num, pct } from "../components/common";
import { EvidenceViewer } from "../components/EvidenceViewer";

const TERMINAL = ["completed", "failed", "cancelled"];

function download(name: string, content: string, type: string) {
  const url = URL.createObjectURL(new Blob([content], { type }));
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function escapeRe(s: string) { return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"); }

/** Description with the selected claim's sentence and its subject/object terms highlighted. */
export function HighlightedDescription({ text, claim }: { text: string; claim: ClaimOut["claim"] | null }) {
  if (!claim?.source_span) return <div className="description-text">{text}</div>;
  const { start, end } = claim.source_span;
  const before = text.slice(0, start), sentence = text.slice(start, end), after = text.slice(end);
  const terms = [claim.subject?.phrase, claim.subject?.head, claim.object?.phrase, claim.object?.head, claim.attribute]
    .filter((t): t is string => !!t && t.length > 1);
  let parts: ReactNode[] = [sentence];
  if (terms.length) {
    const re = new RegExp(`\\b(${[...new Set(terms)].sort((a, b) => b.length - a.length).map(escapeRe).join("|")})\\w*`, "gi");
    parts = [];
    let last = 0;
    for (const m of sentence.matchAll(re)) {
      parts.push(sentence.slice(last, m.index));
      parts.push(<mark key={m.index} className="term-hl">{m[0]}</mark>);
      last = (m.index ?? 0) + m[0].length;
    }
    parts.push(sentence.slice(last));
  }
  return <div className="description-text">{before}<mark className="claim-hl">{parts}</mark>{after}</div>;
}

function EvidenceItemView({ e }: { e: Evidence }) {
  const crop = mediaUrl(e.crop_url);
  return (
    <li className={`ev-item ${crop ? "" : "nocrop"}`}>
      {crop && <img src={crop} alt={`Evidence crop from ${e.source}`} loading="lazy" />}
      <div>
        <div className="ev-head">
          <strong>{e.source.replace("_", " ")}</strong>
          <span className="tag">{e.stance.replace("_", " ")}</span>
          {e.strength !== "none" && <span className="tag">{e.strength}</span>}
          {e.raw_score != null && <span className="tag num" title="Model-native score. Not a calibrated probability and not comparable across models.">score {e.raw_score.toFixed(2)}</span>}
          {e.timestamp_s != null && <span className="tag num">t={fmtSec(e.timestamp_s)}</span>}
          {!!e.details?.hypothesis_only && <span className="tag" title="VLM statements are treated as hypotheses, not ground truth.">hypothesis</span>}
        </div>
        <div className="ev-desc">{e.description}</div>
        {e.model_name && <div className="small muted">{e.model_name}{e.model_version ? ` · ${e.model_version}` : ""}</div>}
      </div>
    </li>
  );
}

function ClaimDetail({ c, evidence, claims }: { c: ClaimOut; evidence: Evidence[]; claims: ClaimOut[] }) {
  const r = c.result;
  if (!r) return <p className="muted">No result recorded.</p>;
  const byId = new Map(evidence.map((e) => [e.evidence_id, e]));
  const sup = r.supporting_evidence.map((i) => byId.get(i)).filter((e): e is Evidence => !!e);
  const con = r.contradicting_evidence.map((i) => byId.get(i)).filter((e): e is Evidence => !!e);
  const used = new Set([...r.supporting_evidence, ...r.contradicting_evidence]);
  const other = r.considered_evidence.filter((i) => !used.has(i)).map((i) => byId.get(i)).filter((e): e is Evidence => !!e);
  const disagreement = r.evidence_case === "disagreement";
  const deps = c.claim.depends_on.map((d) => claims.find((x) => x.claim.claim_id === d)).filter(Boolean) as ClaimOut[];
  return (
    <div className="stack">
      <div className="row">
        <VerdictBadge verdict={r.verdict} />
        <span className="tag">evidence: {r.evidence_case}</span>
        <span className="tag">quality: {r.evidence_quality}</span>
        {r.review_recommended && <span className="tag warn">human review recommended</span>}
        {disagreement && <span className="tag warn">⚠ model disagreement</span>}
      </div>
      <p>{r.explanation}</p>
      <dl className="kv">
        <dt>Claim type</dt><dd>{c.claim.claim_type.replace("_", " ")}{c.claim.negated ? " · negated" : ""}{c.claim.hedged ? " · hedged" : ""}</dd>
        <dt>Agreement score</dt><dd className="num">{r.verification_score == null ? "—" : r.verification_score.toFixed(2)} <span className="muted">(−1…+1 evidence agreement; not a probability)</span></dd>
        <dt>Calibrated probability</dt><dd>{r.calibrated_probability == null ? <span className="muted">not available (no evaluated calibrator loaded)</span> : pct(r.calibrated_probability)}</dd>
        {c.claim.subject && <><dt>Subject</dt><dd>{c.claim.subject.phrase}{c.claim.subject.canonical_class ? <span className="muted"> → {c.claim.subject.canonical_class}</span> : null}</dd></>}
        {c.claim.object && <><dt>Object</dt><dd>{c.claim.object.phrase}</dd></>}
        {c.claim.required_evidence.length > 0 && <><dt>Needs</dt><dd>{c.claim.required_evidence.join(", ").replace(/_/g, " ")}</dd></>}
        {deps.length > 0 && <><dt>Depends on</dt><dd>{deps.map((d) => <div key={d.claim.claim_id}>{d.claim.claim_id}: {d.claim.text} {d.result && <VerdictBadge verdict={d.result.verdict} />}</div>)}</dd></>}
        <dt>Time</dt><dd className="num">{r.duration_ms.toFixed(0)} ms</dd>
      </dl>
      {r.uncertainty_reasons.length > 0 && (
        <div>
          <h3>Uncertainty</h3>
          <ul style={{ margin: 0, paddingLeft: "1.2rem" }}>{r.uncertainty_reasons.map((u, i) => <li key={i}>{u}</li>)}</ul>
        </div>
      )}
      <div>
        <h3>Supporting evidence ({sup.length})</h3>
        {sup.length ? <ul className="ev-list">{sup.map((e) => <EvidenceItemView key={e.evidence_id} e={e} />)}</ul> : <p className="small muted">None.</p>}
      </div>
      <div>
        <h3>Contradicting evidence ({con.length})</h3>
        {con.length ? <ul className="ev-list">{con.map((e) => <EvidenceItemView key={e.evidence_id} e={e} />)}</ul> : <p className="small muted">None.</p>}
      </div>
      {other.length > 0 && (
        <details>
          <summary>Other checks: absent, neutral or not applicable ({other.length})</summary>
          <ul className="ev-list" style={{ marginTop: 8 }}>{other.map((e) => <EvidenceItemView key={e.evidence_id} e={e} />)}</ul>
        </details>
      )}
      <details>
        <summary>Decision trace</summary>
        <ol className="small mono" style={{ paddingLeft: "1.2rem" }}>{r.decision_trace.map((t, i) => <li key={i}>{t}</li>)}</ol>
      </details>
    </div>
  );
}

export default function JobPage() {
  const { jobId = "" } = useParams();
  const nav = useNavigate();
  const [job, setJob] = useState<JobSummary | null>(null);
  const [claims, setClaims] = useState<ClaimOut[]>([]);
  const [evidence, setEvidence] = useState<Evidence[]>([]);
  const [frames, setFrames] = useState<Frame[]>([]);
  const [error, setError] = useState<unknown>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [filter, setFilter] = useState<Verdict | "ALL">("ALL");
  const [busy, setBusy] = useState(false);

  const loadResults = useCallback(async (j: JobSummary) => {
    const [c, e, f] = await Promise.all([api.claims(j.job_id), api.evidence(j.job_id),
      j.media_type === "video" ? api.frames(j.job_id) : Promise.resolve([] as Frame[])]);
    setClaims(c.items);
    setEvidence(e.items);
    setFrames(f);
    const firstInteresting = c.items.find((x) => x.result && ["CONTRADICTED", "REVIEW_REQUIRED"].includes(x.result.verdict)) ?? c.items[0];
    setSelected((s) => s ?? firstInteresting?.claim.claim_id ?? null);
  }, []);

  useEffect(() => {
    let alive = true;
    let timer: ReturnType<typeof setTimeout>;
    const tick = async () => {
      try {
        const j = await api.job(jobId);
        if (!alive) return;
        setJob(j);
        setError(null);
        if (j.status === "completed") await loadResults(j);
        else if (!TERMINAL.includes(j.status)) timer = setTimeout(tick, 1500);
      } catch (e) {
        if (alive) setError(e);
      }
    };
    tick();
    return () => { alive = false; clearTimeout(timer); };
  }, [jobId, loadResults]);

  const selClaim = claims.find((c) => c.claim.claim_id === selected) ?? null;
  const selEvidence = useMemo(() => evidence.filter((e) => e.claim_id === selected), [evidence, selected]);
  const visible = claims.filter((c) => filter === "ALL" || c.result?.verdict === filter);

  if (error && !job) return <ErrorState error={error} onRetry={() => location.reload()} />;
  if (!job) return <Loading label="Loading job…" />;

  const s = job.summary;
  const doCancel = async () => { setBusy(true); try { setJob(await api.cancel(job.job_id)); } catch (e) { setError(e); } finally { setBusy(false); } };
  const doDelete = async () => {
    if (!confirm("Delete this job, its results and stored media? This cannot be undone.")) return;
    setBusy(true);
    try { await api.remove(job.job_id); nav("/history"); } catch (e) { setError(e); setBusy(false); }
  };
  const dl = async (fmt: "json" | "md") => {
    try {
      if (fmt === "json") download(`realitycheck-${job.job_id}.json`, JSON.stringify(await api.reportJson(job.job_id), null, 2), "application/json");
      else download(`realitycheck-${job.job_id}.md`, await api.reportMarkdown(job.job_id), "text/markdown");
    } catch (e) { setError(e); }
  };

  return (
    <div className="stack">
      <div className="row" style={{ justifyContent: "space-between" }}>
        <div>
          <div className="small muted"><Link to="/history">History</Link> / job <code>{job.job_id.slice(0, 8)}</code></div>
          <h1 style={{ marginTop: 4 }}>Verification result</h1>
          <div className="row small">
            <StatusPill status={job.status} />
            <span className="tag">{job.media_type}</span>
            <span className="tag">{job.mode === "generate_and_verify" ? "generated + verified" : "supplied description"}</span>
            {job.demo_mode && <span className="tag warn">DEMO — fake models</span>}
            <span className="muted">created {fmtTime(job.created_at)}</span>
          </div>
        </div>
        <div className="row">
          {job.status === "completed" && <><button onClick={() => dl("json")}>Download JSON</button><button onClick={() => dl("md")}>Download Markdown</button></>}
          {!TERMINAL.includes(job.status) && <button className="danger" disabled={busy} onClick={doCancel}>Cancel</button>}
          {job.status !== "running" && <button className="danger ghost" disabled={busy} onClick={doDelete}>Delete</button>}
        </div>
      </div>

      {error != null && <ErrorState error={error} />}
      {job.demo_mode && (
        <div className="banner demo"><strong>Demo mode</strong><span>These results come from deterministic fake models, not real model inference. Do not interpret them as an evaluation.</span></div>
      )}

      {!TERMINAL.includes(job.status) && (
        <div className="card">
          <Loading label={`${job.status === "queued" ? "Queued — waiting for a worker" : `Processing: ${job.progress_stage ?? "starting"}`}…`} />
          <p className="small muted" style={{ textAlign: "center" }}>On CPU, verification can take from several seconds to a few minutes (videos take longer).</p>
        </div>
      )}
      {job.status === "failed" && (
        <div className="banner error" role="alert"><strong>Failed</strong><span>{job.error?.message ?? "Processing failed."} {job.error?.code && <code>({job.error.code})</code>}</span></div>
      )}
      {job.status === "cancelled" && <div className="banner info"><strong>Cancelled</strong><span>This job was cancelled before completion.</span></div>}

      {job.status === "completed" && s && (
        <>
          <section className="card">
            <div className="card-title"><h2>Summary</h2><span className="small muted">{s.headline}</span></div>
            <VerdictStack counts={s.verdict_counts} />
            <div className="tiles" style={{ marginTop: 12 }}>
              <div className="tile"><div className="label">Claims</div><div className="value">{s.total_claims}</div><div className="sub">{s.verifiable_claims} visually verifiable</div></div>
              <div className="tile"><div className="label">Decided (coverage)</div><div className="value">{pct(s.coverage, 0)}</div><div className="sub">{s.decided_claims} supported or contradicted</div></div>
              <div className="tile"><div className="label">Contradicted</div><div className="value">{s.verdict_counts.CONTRADICTED}</div><div className="sub">{pct(s.contradicted_fraction_of_decided, 0)} of decided claims</div></div>
              <div className="tile"><div className="label">Review recommended</div><div className="value">{s.review_recommended}</div><div className="sub">disagreement or weak evidence</div></div>
              <div className="tile"><div className="label">Processing time</div><div className="value num">{job.timings_ms?.total ? `${(job.timings_ms.total / 1000).toFixed(1)}s` : "—"}</div><div className="sub">{job.policy ?? ""} · v{job.pipeline_version}</div></div>
            </div>
            {s.sampling && <p className="small muted" style={{ marginTop: 8 }}>Video sampling: {s.sampling.strategy}, {s.sampling.frames} frames over {fmtSec(s.sampling.duration_s)}; largest gap between sampled frames {fmtSec(s.sampling.max_gap_s)} — events shorter than that can be missed.</p>}
            {job.warnings.length > 0 && <ul className="small ink2" style={{ marginBottom: 0 }}>{job.warnings.map((w, i) => <li key={i}>{w}</li>)}</ul>}
          </section>

          <div className="grid grid-results">
            <section className="card" aria-label="Evidence viewer">
              <div className="card-title"><h2>Evidence</h2>{selClaim && <span className="small ink2">claim {selClaim.claim.claim_id}</span>}</div>
              {job.media && (
                <EvidenceViewer media={job.media} frames={frames} evidence={selEvidence}
                  supporting={selClaim?.result?.supporting_evidence ?? []} contradicting={selClaim?.result?.contradicting_evidence ?? []}
                  claimSelected={!!selClaim} />
              )}
              <h3 style={{ marginTop: 14 }}>{job.mode === "generate_and_verify" ? "Generated description" : "Description"}</h3>
              <HighlightedDescription text={job.description_used ?? ""} claim={selClaim?.claim ?? null} />
            </section>

            <section className="card" aria-label="Claims">
              <div className="card-title">
                <h2>Claims ({claims.length})</h2>
                <select aria-label="Filter claims by verdict" value={filter} onChange={(e) => setFilter(e.target.value as Verdict | "ALL")} style={{ width: "auto" }}>
                  <option value="ALL">All verdicts</option>
                  {VERDICT_ORDER.map((v) => <option key={v} value={v}>{VERDICT_LABEL[v]} ({s.verdict_counts[v] ?? 0})</option>)}
                </select>
              </div>
              {visible.length === 0 ? <p className="muted">No claims match this filter.</p> : (
                <ul className="claim-list">
                  {visible.map((c) => (
                    <li key={c.claim.claim_id}>
                      <button className="claim-item" aria-pressed={selected === c.claim.claim_id} onClick={() => setSelected(c.claim.claim_id)}>
                        <div><span className="muted small mono">{c.claim.claim_id}</span> {c.claim.text}</div>
                        <div className="meta">
                          {c.result && <VerdictBadge verdict={c.result.verdict} />}
                          <span className="tag">{c.claim.claim_type.replace(/_/g, " ")}</span>
                          {c.result && c.result.evidence_quality !== "none" && <span className="tag">{c.result.evidence_quality} quality</span>}
                          {c.result?.evidence_case === "disagreement" && <span className="tag warn">disagreement</span>}
                          {c.result?.review_recommended && c.result.evidence_case !== "disagreement" && <span className="tag warn">review</span>}
                          {c.claim.flags.includes("instruction_like_text") && <span className="tag warn">instruction-like text</span>}
                        </div>
                      </button>
                    </li>
                  ))}
                </ul>
              )}
            </section>
          </div>

          {selClaim && (
            <section className="card" aria-label="Claim detail">
              <div className="card-title"><h2>{selClaim.claim.claim_id}: {selClaim.claim.text}</h2></div>
              <ClaimDetail c={selClaim} evidence={evidence} claims={claims} />
            </section>
          )}
          <details className="card">
            <summary>Processing details</summary>
            <dl className="kv" style={{ marginTop: 8 }}>
              <dt>Started / finished</dt><dd>{fmtTime(job.started_at)} → {fmtTime(job.finished_at)}</dd>
              <dt>Attempts</dt><dd>{job.attempts}</dd>
              <dt>Timings</dt><dd className="num">{Object.entries(job.timings_ms ?? {}).map(([k, v]) => `${k} ${num(v / 1000, 2)}s`).join(" · ")}</dd>
              <dt>Settings</dt><dd className="mono">{JSON.stringify(job.settings)}</dd>
              {s.image_quality && <><dt>Image quality</dt><dd className="mono">{Object.entries(s.image_quality).map(([k, v]) => `${k}=${v}`).join(", ")}</dd></>}
            </dl>
          </details>
        </>
      )}
    </div>
  );
}
