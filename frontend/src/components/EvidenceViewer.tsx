import { useMemo, useRef, useState } from "react";
import { mediaUrl } from "../api/client";
import type { Box, Evidence, Frame, Media } from "../api/types";
import { fmtSec } from "./common";

export type BoxKind = "support" | "contradict" | "other";

export interface OverlayBox {
  key: string;
  box: Box;
  kind: BoxKind;
  label: string;
}

const KIND_STYLE: Record<BoxKind, { border: string; bg: string }> = {
  support: { border: "3px solid var(--good)", bg: "var(--good)" },
  contradict: { border: "3px dashed var(--critical)", bg: "var(--critical)" },
  other: { border: "2px dotted #d9d7cf", bg: "#52514e" },
};

function isBox(v: unknown): v is Box {
  return !!v && typeof v === "object" && ["x1", "y1", "x2", "y2"].every((k) => typeof (v as Record<string, unknown>)[k] === "number");
}

/** Turn evidence items into drawable boxes. Only items with real localized regions produce boxes. */
export function overlayBoxes(evidence: Evidence[], supporting: Set<string>, contradicting: Set<string>, frameIndex: number | null): OverlayBox[] {
  const out: OverlayBox[] = [];
  for (const e of evidence) {
    if ((e.frame_index ?? null) !== frameIndex) continue;
    const kind: BoxKind = supporting.has(e.evidence_id) ? "support" : contradicting.has(e.evidence_id) ? "contradict" : "other";
    const src = e.source.replace("_", " ");
    const d = e.details ?? {};
    if (isBox(d.subject_box) && isBox(d.object_box)) {
      out.push({ key: `${e.evidence_id}-s`, box: d.subject_box, kind, label: `${src}: subject` });
      out.push({ key: `${e.evidence_id}-o`, box: d.object_box, kind, label: `${src}: object` });
      continue;
    }
    if (Array.isArray(d.boxes) && d.boxes.length > 1) {
      (d.boxes as unknown[]).filter(isBox).forEach((b, i) => out.push({ key: `${e.evidence_id}-${i}`, box: b, kind, label: `${src} #${i + 1}` }));
      continue;
    }
    if (e.box && e.source !== "vlm") {
      const score = e.raw_score != null ? ` ${e.raw_score.toFixed(2)}` : "";
      out.push({ key: e.evidence_id, box: e.box, kind, label: `${src}${e.label ? ` '${e.label}'` : ""}${score}` });
    }
  }
  // de-duplicate identical regions from several sources (keep first label, append the others)
  const merged: OverlayBox[] = [];
  for (const b of out) {
    const same = merged.find((m) => Math.abs(m.box.x1 - b.box.x1) < 0.01 && Math.abs(m.box.y1 - b.box.y1) < 0.01 &&
      Math.abs(m.box.x2 - b.box.x2) < 0.01 && Math.abs(m.box.y2 - b.box.y2) < 0.01 && m.kind === b.kind);
    if (same) same.label = `${same.label} · ${b.label}`;
    else merged.push({ ...b });
  }
  return merged;
}

function Overlay({ boxes }: { boxes: OverlayBox[] }) {
  return (
    <div style={{ position: "absolute", inset: 0, pointerEvents: "none" }} aria-hidden="true">
      {boxes.map((b) => {
        const st = KIND_STYLE[b.kind];
        const top = b.box.y1 < 0.06;
        return (
          <div key={b.key} style={{ position: "absolute", left: `${b.box.x1 * 100}%`, top: `${b.box.y1 * 100}%`,
            width: `${(b.box.x2 - b.box.x1) * 100}%`, height: `${(b.box.y2 - b.box.y1) * 100}%`, border: st.border,
            borderRadius: 3, boxShadow: "0 0 0 1px rgba(0,0,0,0.45)" }}>
            <span className="box-label" style={{ position: "absolute", left: -3, [top ? "bottom" : "top"]: top ? -22 : -22,
              background: st.bg, color: "#fff", padding: "2px 6px", borderRadius: 4, lineHeight: "16px", whiteSpace: "nowrap",
              maxWidth: 260, overflow: "hidden", textOverflow: "ellipsis" }}>{b.label}</span>
          </div>
        );
      })}
    </div>
  );
}

export function BoxLegend() {
  return (
    <div className="legend" aria-label="Box legend">
      <span><span className="sw support" />supporting (solid)</span>
      <span><span className="sw contradict" />contradicting (dashed)</span>
      <span><span className="sw other" />considered, neutral (dotted)</span>
    </div>
  );
}

interface Props {
  media: Media;
  frames: Frame[];
  evidence: Evidence[]; // evidence of the selected claim (or [])
  supporting: string[];
  contradicting: string[];
  claimSelected: boolean;
}

export function EvidenceViewer({ media, frames, evidence, supporting, contradicting, claimSelected }: Props) {
  const [showOverlay, setShowOverlay] = useState(true);
  const [frameIdx, setFrameIdx] = useState<number | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const sup = useMemo(() => new Set(supporting), [supporting]);
  const con = useMemo(() => new Set(contradicting), [contradicting]);

  const evidenceFrames = useMemo(() => {
    const m = new Map<number, "sup" | "con" | "any">();
    for (const e of evidence) {
      if (e.frame_index == null) continue;
      const k = sup.has(e.evidence_id) ? "sup" : con.has(e.evidence_id) ? "con" : "any";
      const prev = m.get(e.frame_index);
      if (!prev || k === "sup" || (k === "con" && prev === "any")) m.set(e.frame_index, k);
    }
    return m;
  }, [evidence, sup, con]);

  const activeFrame = media.media_type === "video"
    ? frames.find((f) => f.frame_index === frameIdx) ?? null
    : null;
  const boxes = overlayBoxes(evidence, sup, con, media.media_type === "video" ? (activeFrame?.frame_index ?? -1) : null);
  const allBoxesForClaim = media.media_type === "video"
    ? evidence.some((e) => e.box || (e.details && (isBox(e.details.subject_box) || Array.isArray(e.details.boxes))))
    : boxes.length > 0;

  const seek = (t: number) => { if (videoRef.current) { videoRef.current.currentTime = t; } };

  return (
    <div>
      <div className="row" style={{ justifyContent: "space-between", marginBottom: 8 }}>
        <label className="toggle small">
          <input type="checkbox" checked={showOverlay} onChange={(e) => setShowOverlay(e.target.checked)} />
          Show evidence overlay
        </label>
        {media.width && media.height && <span className="small muted num">{media.width}×{media.height}{media.duration_s ? ` · ${fmtSec(media.duration_s)}` : ""}</span>}
      </div>

      {media.media_type === "image" ? (
        <div className="viewer">
          {media.url ? <img src={mediaUrl(media.url)} alt="Uploaded media" /> : <div className="state">Media no longer available (retention policy).</div>}
          {showOverlay && <Overlay boxes={boxes} />}
        </div>
      ) : (
        <div className="stack">
          {media.url ? (
            <video ref={videoRef} src={mediaUrl(media.url)} controls preload="metadata" style={{ width: "100%", maxHeight: "40vh", borderRadius: 8, background: "#000" }} />
          ) : <div className="state">Video no longer available (retention policy).</div>}
          {frames.length > 0 && (
            <>
              <div className="small ink2">Sampled frames — select one to inspect its evidence{claimSelected ? "; colored underline = evidence for the selected claim" : ""}.</div>
              <div className="frame-strip" role="list">
                {frames.map((f) => {
                  const k = evidenceFrames.get(f.frame_index);
                  return (
                    <button key={f.frame_index} role="listitem" className={`frame-thumb ${k === "sup" ? "has-ev" : k === "con" ? "has-con" : ""}`}
                      aria-pressed={frameIdx === f.frame_index}
                      aria-label={`Frame at ${fmtSec(f.timestamp_s)}${k ? ` (${k === "sup" ? "supporting" : k === "con" ? "contradicting" : "considered"} evidence)` : ""}`}
                      onClick={() => { setFrameIdx(f.frame_index); seek(f.timestamp_s); }}>
                      {f.url && <img src={mediaUrl(f.url)} alt="" loading="lazy" />}
                      <span className="ts num">{fmtSec(f.timestamp_s)} · {f.reason.replace("_", " ")}</span>
                    </button>
                  );
                })}
              </div>
              {activeFrame && (
                <div className="viewer">
                  {activeFrame.url && <img src={mediaUrl(activeFrame.url)} alt={`Frame at ${fmtSec(activeFrame.timestamp_s)}`} />}
                  {showOverlay && <Overlay boxes={boxes} />}
                </div>
              )}
            </>
          )}
        </div>
      )}
      {showOverlay && <BoxLegend />}
      {claimSelected && !allBoxesForClaim && (
        <div className="no-evidence" role="note">
          No reliable localized evidence for this claim, so no region is drawn. The verdict relies on whole-image or
          non-spatial evidence (see the evidence panels).
        </div>
      )}
      {claimSelected && media.media_type === "video" && allBoxesForClaim && !activeFrame && (
        <div className="no-evidence" role="note">Select a sampled frame to see regions for this claim.</div>
      )}
    </div>
  );
}
