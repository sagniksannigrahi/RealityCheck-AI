import type { ReactNode } from "react";
import type { Verdict } from "../api/types";
import { VERDICT_FILL, VERDICT_LABEL, VERDICT_ORDER, VerdictIcon, pct, useTooltip } from "./common";

export interface BarDatum {
  key: string;
  label: string;
  value: number | null;
  low?: number | null;
  high?: number | null;
  note?: string;
}

/**
 * Horizontal bar chart on a fixed 0..1 scale with optional 95% CI whiskers.
 * One series per chart (small multiples instead of dual axes); direct value labels; hover tooltips.
 */
export function MetricBars({ title, data, format = (v) => pct(v), highlight }: {
  title: string; data: BarDatum[]; format?: (v: number | null) => string; highlight?: string;
}) {
  const tip = useTooltip();
  const rowH = 26, labelW = 170, valueW = 64, width = 640, plotW = width - labelW - valueW, top = 22;
  const height = top + data.length * rowH + 8;
  const x = (v: number) => labelW + Math.max(0, Math.min(1, v)) * plotW;
  return (
    <figure style={{ margin: 0 }}>
      <figcaption className="small" style={{ fontWeight: 600, marginBottom: 4 }}>{title}</figcaption>
      <svg className="chart" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`${title}: ${data.map((d) => `${d.label} ${format(d.value)}`).join(", ")}`}>
        {[0, 0.25, 0.5, 0.75, 1].map((t) => (
          <g key={t}>
            <line className="grid-line" x1={x(t)} x2={x(t)} y1={top - 4} y2={height - 6} />
            <text x={x(t)} y={12} textAnchor="middle" className="num">{Math.round(t * 100)}%</text>
          </g>
        ))}
        <line className="axis-line" x1={labelW} x2={labelW} y1={top - 4} y2={height - 6} />
        {data.map((d, i) => {
          const y = top + i * rowH;
          const v = d.value;
          const content: ReactNode = (
            <div>
              <strong>{d.label}</strong>
              <div className="num">{format(v)}{d.low != null && d.high != null ? ` (95% CI ${format(d.low)}–${format(d.high)})` : ""}</div>
              {d.note && <div className="muted">{d.note}</div>}
            </div>
          );
          return (
            <g key={d.key} onMouseMove={(e) => tip.show(e, content)} onMouseLeave={tip.hide} tabIndex={0}
              onFocus={(e) => tip.showAt(e.currentTarget, content)} onBlur={tip.hide} style={{ outline: "none" }}>
              <rect x={0} y={y} width={width} height={rowH} fill="transparent" />
              <text x={labelW - 8} y={y + rowH / 2 + 4} textAnchor="end" style={{ fontWeight: d.key === highlight ? 700 : 400, fill: "var(--ink)" }}>
                {d.label}
              </text>
              {v != null && (
                <path className="bar" d={roundedBar(labelW, y + 6, Math.max(1, x(v) - labelW), rowH - 12, 4)}
                  opacity={highlight && d.key !== highlight ? 0.55 : 1} />
              )}
              {d.low != null && d.high != null && (
                <g>
                  <line className="whisker" x1={x(d.low)} x2={x(d.high)} y1={y + rowH / 2} y2={y + rowH / 2} />
                  <line className="whisker" x1={x(d.low)} x2={x(d.low)} y1={y + rowH / 2 - 5} y2={y + rowH / 2 + 5} />
                  <line className="whisker" x1={x(d.high)} x2={x(d.high)} y1={y + rowH / 2 - 5} y2={y + rowH / 2 + 5} />
                </g>
              )}
              <text className="val num" x={width - 4} y={y + rowH / 2 + 4} textAnchor="end">{format(v)}</text>
            </g>
          );
        })}
      </svg>
      {tip.node}
    </figure>
  );
}

/** Bar anchored at the baseline with a 4px rounded data-end. */
function roundedBar(x: number, y: number, w: number, h: number, r: number): string {
  const rr = Math.min(r, w / 2, h / 2);
  return `M${x},${y} H${x + w - rr} Q${x + w},${y} ${x + w},${y + rr} V${y + h - rr} Q${x + w},${y + h} ${x + w - rr},${y + h} H${x} Z`;
}

const SEQ = ["var(--seq-100)", "var(--seq-200)", "var(--seq-300)", "var(--seq-400)", "var(--seq-500)", "var(--seq-600)", "var(--seq-700)"];

/** Confusion matrix as a table heatmap (sequential blue; numbers always printed). Rows = gold, columns = predicted. */
export function ConfusionMatrix({ matrix }: { matrix: Record<string, Record<string, number>> }) {
  const rows = Object.keys(matrix);
  const cols = rows.length ? Object.keys(matrix[rows[0]]) : [];
  const max = Math.max(1, ...rows.flatMap((r) => cols.map((c) => matrix[r][c])));
  return (
    <div className="table-wrap">
      <table aria-label="Confusion matrix (rows gold label, columns prediction)">
        <thead>
          <tr><th>gold ↓ / predicted →</th>{cols.map((c) => <th key={c} className="num">{c.toLowerCase()}</th>)}<th className="num">total</th></tr>
        </thead>
        <tbody>
          {rows.map((r) => {
            const total = cols.reduce((s, c) => s + matrix[r][c], 0);
            return (
              <tr key={r}>
                <th scope="row">{r.toLowerCase()}</th>
                {cols.map((c) => {
                  const v = matrix[r][c];
                  const step = Math.min(SEQ.length - 1, Math.floor((v / max) * (SEQ.length - 1)));
                  const dark = step >= 4;
                  return (
                    <td key={c} className="cm-cell" style={{ background: v ? SEQ[step] : "transparent", color: dark ? "#fff" : "var(--ink)" }}
                      title={`${v} of ${total} gold-${r.toLowerCase()} claims predicted ${c.toLowerCase()} (${pct(total ? v / total : null)})`}>
                      {v}<div className="small" style={{ fontWeight: 400, opacity: 0.85 }}>{pct(total ? v / total : null, 0)}</div>
                    </td>
                  );
                })}
                <td className="num">{total}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

/** Stacked bar of verdict counts with a legend (icons + labels, never color alone). */
export function VerdictStack({ counts, compact = false }: { counts: Partial<Record<Verdict, number>>; compact?: boolean }) {
  const tip = useTooltip();
  const total = VERDICT_ORDER.reduce((s, v) => s + (counts[v] ?? 0), 0);
  if (!total) return <span className="muted small">no claims</span>;
  return (
    <div>
      <div className="stackbar" role="img" aria-label={VERDICT_ORDER.map((v) => `${VERDICT_LABEL[v]}: ${counts[v] ?? 0}`).join(", ")}>
        {VERDICT_ORDER.filter((v) => counts[v]).map((v) => (
          <span key={v} style={{ width: `${((counts[v] ?? 0) / total) * 100}%`, background: VERDICT_FILL[v] }}
            onMouseMove={(e) => tip.show(e, <span>{VERDICT_LABEL[v]}: <strong>{counts[v]}</strong> ({pct((counts[v] ?? 0) / total, 0)})</span>)}
            onMouseLeave={tip.hide} />
        ))}
      </div>
      {!compact && (
        <div className="legend">
          {VERDICT_ORDER.map((v) => (
            <span key={v} className="row" style={{ gap: 4 }}>
              <span className={`badge v-${v}`} style={{ padding: "0 0.35rem" }}><VerdictIcon verdict={v} /></span>
              {VERDICT_LABEL[v]} <strong className="num">{counts[v] ?? 0}</strong>
            </span>
          ))}
        </div>
      )}
      {tip.node}
    </div>
  );
}
