import { useEffect, useMemo, useState } from "react";
import { api } from "../api/client";
import type { EvaluationList, EvaluationRun } from "../api/types";
import { ConfusionMatrix, MetricBars } from "../components/charts";
import { EmptyState, ErrorState, Loading, fmtTime, num, pct } from "../components/common";

const SYSTEM_ORDER = ["vlm_only", "detector_rules", "vlm_grounding", "full", "eav_sa", "eav_sa_default",
  "ablate_no_grounding", "ablate_no_detector", "ablate_no_disagreement", "ablate_no_vlm"];

function groupKey(r: EvaluationRun) { return String(r.config?.experiment_run ?? r.run_id.split(":")[0]); }

export default function EvaluationPage() {
  const [data, setData] = useState<EvaluationList | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [group, setGroup] = useState<string>("");
  const [system, setSystem] = useState<string>("eav_sa");

  useEffect(() => { api.evaluations().then(setData).catch(setError); }, []);

  const groups = useMemo(() => {
    const m = new Map<string, EvaluationRun[]>();
    for (const r of data?.items ?? []) {
      const k = groupKey(r);
      m.set(k, [...(m.get(k) ?? []), r]);
    }
    return [...m.entries()].sort((a, b) => (b[1][0].created_at > a[1][0].created_at ? 1 : -1));
  }, [data]);

  useEffect(() => { if (!group && groups.length) setGroup(groups[0][0]); }, [groups, group]);

  if (error) return <ErrorState error={error} />;
  if (!data) return <Loading label="Loading evaluation runs…" />;
  if (!groups.length) {
    return (
      <div className="stack">
        <h1>Evaluation</h1>
        <EmptyState title="No evaluation runs registered">
          <p className="ink2">Run the harness and register results, e.g.</p>
          <pre className="mono small" style={{ textAlign: "left", display: "inline-block" }}>python scripts/run_evaluation.py --dataset pope --backend real --experiment pope-v1 --register</pre>
        </EmptyState>
      </div>
    );
  }

  const runs = (groups.find(([k]) => k === group)?.[1] ?? []).slice()
    .sort((a, b) => SYSTEM_ORDER.indexOf(a.system) - SYSTEM_ORDER.indexOf(b.system));
  const sel = runs.find((r) => r.system === system) ?? runs[0];
  const first = runs[0];
  const simulated = runs.some((r) => r.simulated);
  const ci = (r: EvaluationRun, k: string) => r.metrics.confidence_intervals_95?.[k] ?? { low: null, high: null };
  const cats = sel ? Object.entries(sel.metrics.per_category ?? {}) : [];
  const comparisons = runs.flatMap((r) => Object.entries(r.metrics.comparisons ?? {}));
  const splitCounts = first?.config?.split_label_counts as Record<string, Record<string, number>> | undefined;

  return (
    <div className="stack">
      <div className="row" style={{ justifyContent: "space-between" }}>
        <h1>Evaluation</h1>
        <label className="row small">Experiment run
          <select value={group} onChange={(e) => setGroup(e.target.value)} style={{ width: "auto", maxWidth: 420 }}>
            {groups.map(([k, rs]) => <option key={k} value={k}>{k} · {rs[0].dataset}{rs[0].simulated ? " · SIMULATED" : ""}</option>)}
          </select>
        </label>
      </div>
      {simulated ? (
        <div className="banner demo"><strong>Simulated</strong><span>This run used deterministic fake models. Its numbers test the harness only and say nothing about real model performance.</span></div>
      ) : (
        <div className="banner info"><strong>Measured</strong><span>Results from executing the listed models on the held-out test split; thresholds were selected on a separate validation split. See the run's <code>report.md</code> for the full protocol.</span></div>
      )}
      {first && (
        <div className="card small">
          <dl className="kv">
            <dt>Dataset</dt><dd>{first.dataset} <span className="muted">({first.dataset_version})</span></dd>
            <dt>Test examples</dt><dd className="num">{first.n_examples}{splitCounts ? ` · label counts ${JSON.stringify(splitCounts)}` : ""}</dd>
            <dt>Models</dt><dd>{Object.entries(first.model_versions).map(([k, v]) => <div key={k}><span className="muted">{k}:</span> {v}</div>)}</dd>
            <dt>Seed / finished</dt><dd>{first.seed} · {fmtTime(first.finished_at)}</dd>
            {first.metrics.runtime && <><dt>Evidence collection</dt><dd className="num">{first.metrics.runtime.evidence_collection_s}s · peak RSS {first.metrics.runtime.peak_rss_mb} MB</dd></>}
          </dl>
        </div>
      )}

      <section className="card">
        <div className="card-title"><h2>Systems on the test split</h2><span className="small muted">P/R/F1 on covered claims · positive class = contradicted (hallucination)</span></div>
        <div className="table-wrap">
          <table>
            <thead><tr><th>System</th><th className="num">Coverage</th><th className="num">Acc. covered</th><th className="num">Acc. effective</th>
              <th className="num">Macro-F1 [95% CI]</th><th className="num">Halluc. P</th><th className="num">Halluc. R</th><th className="num">Halluc. F1</th>
              <th className="num">FPR</th><th className="num">FNR</th><th className="num">ECE</th><th className="num">Latency</th></tr></thead>
            <tbody>
              {runs.map((r) => {
                const o = r.metrics.overall, c = ci(r, "macro_f1");
                return (
                  <tr key={r.run_id} className={`clickable ${r.system === sel?.system ? "selected" : ""}`} onClick={() => setSystem(r.system)} title={r.metrics.description}>
                    <td><strong>{r.system}</strong><div className="small muted">{r.metrics.description}</div></td>
                    <td className="num">{pct(o.coverage)}</td><td className="num">{num(o.accuracy_covered)}</td><td className="num">{num(o.accuracy_effective)}</td>
                    <td className="num">{num(o.macro_f1)} <span className="muted">[{num(c.low, 2)}, {num(c.high, 2)}]</span></td>
                    <td className="num">{num(o.hallucination_precision)}</td><td className="num">{num(o.hallucination_recall)}</td><td className="num">{num(o.hallucination_f1)}</td>
                    <td className="num">{num(o.false_positive_rate)}</td><td className="num">{num(o.false_negative_rate)}</td>
                    <td className="num">{num(r.metrics.calibration?.ece)}</td><td className="num">{o.mean_latency_ms == null ? "—" : `${o.mean_latency_ms.toFixed(0)} ms`}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </section>

      <div className="grid grid-2">
        <section className="card">
          <MetricBars title="Macro-F1 on covered claims (95% CI, cluster bootstrap)" highlight={sel?.system}
            data={runs.map((r) => ({ key: r.system, label: r.system, value: r.metrics.overall.macro_f1, low: ci(r, "macro_f1").low, high: ci(r, "macro_f1").high }))} />
        </section>
        <section className="card">
          <MetricBars title="Coverage (share of claims decided, 95% CI)" highlight={sel?.system}
            data={runs.map((r) => ({ key: r.system, label: r.system, value: r.metrics.overall.coverage, low: ci(r, "coverage").low, high: ci(r, "coverage").high,
              note: "1 − abstention rate" }))} />
        </section>
      </div>

      {sel && (
        <div className="grid grid-2">
          <section className="card">
            <div className="card-title"><h2>Confusion matrix — {sel.system}</h2></div>
            <ConfusionMatrix matrix={sel.metrics.confusion_matrix} />
            <p className="small muted" style={{ marginTop: 8 }}>Selected thresholds (val): <code>{JSON.stringify(sel.metrics.selected_params)}</code></p>
          </section>
          <section className="card">
            <div className="card-title"><h2>Per category — {sel.system}</h2></div>
            <div className="table-wrap">
              <table>
                <thead><tr><th>Category</th><th className="num">n</th><th className="num">Coverage</th><th className="num">Acc. covered</th><th className="num">Macro-F1</th></tr></thead>
                <tbody>{cats.map(([k, m]) => <tr key={k}><td>{k.replace(/_/g, " ")}</td><td className="num">{m.n}</td><td className="num">{pct(m.coverage)}</td><td className="num">{num(m.accuracy_covered)}</td><td className="num">{num(m.macro_f1)}</td></tr>)}
                  {sel.metrics.ambiguous_cases && <tr><td><em>ambiguous subset</em></td><td className="num">{sel.metrics.ambiguous_cases.n}</td><td className="num">{pct(sel.metrics.ambiguous_cases.coverage)}</td><td className="num">{num(sel.metrics.ambiguous_cases.accuracy_covered)}</td><td className="num">{num(sel.metrics.ambiguous_cases.macro_f1)}</td></tr>}
                </tbody>
              </table>
            </div>
            <p className="small muted" style={{ marginTop: 8 }}>Calibration: ECE {num(sel.metrics.calibration?.ece)} · Brier {num(sel.metrics.calibration?.brier)} (n={sel.metrics.calibration?.n_calibration ?? 0})</p>
          </section>
        </div>
      )}

      {comparisons.length > 0 && (
        <section className="card">
          <div className="card-title"><h2>Paired comparisons</h2><span className="small muted">same test examples · cluster bootstrap over images</span></div>
          <div className="table-wrap">
            <table>
              <thead><tr><th>Comparison</th><th>Metric</th><th className="num">Δ</th><th className="num">95% CI</th><th className="num">p</th></tr></thead>
              <tbody>{comparisons.map(([k, c]) => <tr key={k}><td>{k.replace(/_vs_/, " vs ")}</td><td>{c.metric}</td><td className="num">{num(c.diff)}</td>
                <td className="num">[{num(c.ci_low)}, {num(c.ci_high)}]</td><td className="num">{num(c.p_value)}</td></tr>)}</tbody>
            </table>
          </div>
          <p className="small muted" style={{ marginTop: 8 }}>A confidence interval that includes 0 means the difference is not established on this sample.</p>
        </section>
      )}
    </div>
  );
}
