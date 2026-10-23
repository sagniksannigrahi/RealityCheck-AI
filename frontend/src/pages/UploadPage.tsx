import { useEffect, useMemo, useRef, useState, type DragEvent } from "react";
import { useNavigate } from "react-router-dom";
import { api } from "../api/client";
import type { JobSettings, Mode, SystemInfo } from "../api/types";
import { ErrorState } from "../components/common";

const IMAGE_TYPES = ["image/jpeg", "image/png", "image/webp", "image/gif"];
const VIDEO_TYPES = ["video/mp4", "video/quicktime", "video/webm", "video/x-msvideo"];

export const EXAMPLE = "A brown dog is running across a green field while a man in a red shirt holds a blue ball.";

export function validateFile(f: File, info: SystemInfo | null): string | null {
  const isImage = IMAGE_TYPES.includes(f.type) || /\.(jpe?g|png|webp|gif)$/i.test(f.name);
  const isVideo = VIDEO_TYPES.includes(f.type) || /\.(mp4|mov|webm|avi)$/i.test(f.name);
  if (!isImage && !isVideo) return "Unsupported format. Use JPEG, PNG, WebP or GIF images, or MP4, MOV, WebM or AVI videos.";
  const limitMb = isImage ? info?.limits.max_image_mb ?? 15 : info?.limits.max_video_mb ?? 100;
  if (f.size > limitMb * 1024 * 1024) return `File is ${(f.size / 1048576).toFixed(1)} MB; the ${isImage ? "image" : "video"} limit is ${limitMb} MB.`;
  if (f.size === 0) return "The file is empty.";
  return null;
}

export default function UploadPage({ info }: { info: SystemInfo | null }) {
  const nav = useNavigate();
  const [file, setFile] = useState<File | null>(null);
  const [fileError, setFileError] = useState<string | null>(null);
  const [desc, setDesc] = useState("");
  const [mode, setMode] = useState<Mode>("verify_description");
  const [settings, setSettings] = useState<JobSettings>({ abstention: true, disagreement_analysis: true, use_detector: true,
    use_grounding: true, use_vlm: true, sampling_strategy: "scene_aware", max_frames: null });
  const [drag, setDrag] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const preview = useMemo(() => (file ? URL.createObjectURL(file) : null), [file]);
  useEffect(() => () => { if (preview) URL.revokeObjectURL(preview); }, [preview]);
  const isVideo = !!file && (file.type.startsWith("video/") || /\.(mp4|mov|webm|avi)$/i.test(file.name));
  const maxChars = info?.limits.max_description_chars ?? 4000;

  const pick = (f: File | undefined | null) => {
    if (!f) return;
    const err = validateFile(f, info);
    setFileError(err);
    setFile(err ? null : f);
  };
  const onDrop = (e: DragEvent) => { e.preventDefault(); setDrag(false); pick(e.dataTransfer.files?.[0]); };

  const descMissing = mode === "verify_description" && !desc.trim();
  const noComponent = !settings.use_detector && !settings.use_grounding && !settings.use_vlm;
  const canSubmit = !!file && !descMissing && !noComponent && desc.length <= maxChars && !submitting;

  const submit = async () => {
    if (!file) return;
    setSubmitting(true);
    setError(null);
    try {
      const s: JobSettings = { ...settings };
      if (!isVideo) { delete s.sampling_strategy; delete s.max_frames; }
      if (!s.max_frames) delete s.max_frames;
      const r = await api.createVerification(file, mode === "verify_description" ? desc : "", mode, s);
      nav(`/jobs/${r.job_id}`);
    } catch (e) {
      setError(e);
    } finally {
      setSubmitting(false);
    }
  };

  const set = <K extends keyof JobSettings>(k: K, v: JobSettings[K]) => setSettings((p) => ({ ...p, [k]: v }));

  return (
    <div className="stack">
      <div>
        <h1>Verify a description against visual evidence</h1>
        <p className="ink2">Upload an image or a short video and a description. RealityCheck splits the description into atomic
          claims and checks each one with independent evidence (object detection, open-vocabulary grounding, pixel analysis and a
          separate vision-language check). It can abstain when evidence is inadequate.</p>
      </div>
      {info && (
        <div className={`banner ${info.demo_mode ? "demo" : "info"}`}>
          <strong>{info.demo_mode ? "Demo mode" : info.privacy.processing === "local" ? "Local processing" : "External processing"}</strong>
          <span>{info.demo_mode
            ? "ML models are not available on this server, so deterministic fake models are used. Results are labelled DEMO and are not real model outputs; unknown images will mostly receive “insufficient evidence”."
            : info.privacy.statement}</span>
        </div>
      )}

      <div className="grid grid-2">
        <section className="card" aria-labelledby="media-h">
          <h2 id="media-h">1. Media</h2>
          <div className={`dropzone ${drag ? "drag" : ""}`} role="button" tabIndex={0}
            onClick={() => inputRef.current?.click()} onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") inputRef.current?.click(); }}
            onDragOver={(e) => { e.preventDefault(); setDrag(true); }} onDragLeave={() => setDrag(false)} onDrop={onDrop}
            aria-describedby="media-hint">
            <div><strong>Drop an image or video here</strong>, or click to choose a file</div>
            <div id="media-hint" className="hint">
              Images: JPEG, PNG, WebP, GIF up to {info?.limits.max_image_mb ?? 15} MB · Videos: MP4, MOV, WebM, AVI up to{" "}
              {info?.limits.max_video_mb ?? 100} MB and {info?.limits.max_video_seconds ?? 120} s
            </div>
            <input ref={inputRef} type="file" hidden accept={[...IMAGE_TYPES, ...VIDEO_TYPES, ".mov"].join(",")}
              onChange={(e) => pick(e.target.files?.[0])} data-testid="file-input" />
          </div>
          {fileError && <div className="banner error" role="alert" style={{ marginTop: 8 }}>{fileError}</div>}
          {file && preview && (
            <div>
              {isVideo ? <video className="preview-media" src={preview} controls muted /> : <img className="preview-media" src={preview} alt="Selected file preview" />}
              <div className="row small muted" style={{ justifyContent: "center", marginTop: 6 }}>
                {file.name} · {(file.size / 1048576).toFixed(2)} MB
                <button className="ghost small" onClick={() => setFile(null)}>Remove</button>
              </div>
            </div>
          )}
        </section>

        <section className="card" aria-labelledby="desc-h">
          <h2 id="desc-h">2. Description</h2>
          <div className="segmented" role="group" aria-label="Mode" style={{ marginBottom: 10 }}>
            <button aria-pressed={mode === "verify_description"} onClick={() => setMode("verify_description")}>Verify my description</button>
            <button aria-pressed={mode === "generate_and_verify"} onClick={() => setMode("generate_and_verify")}>Generate, then verify</button>
          </div>
          {mode === "verify_description" ? (
            <>
              <label className="field" htmlFor="desc">Description to verify</label>
              <textarea id="desc" value={desc} onChange={(e) => setDesc(e.target.value)} maxLength={maxChars + 1}
                placeholder={EXAMPLE} aria-invalid={desc.length > maxChars} />
              <div className="row" style={{ justifyContent: "space-between" }}>
                <button className="ghost small" onClick={() => setDesc(EXAMPLE)}>Use example</button>
                <span className={`small num ${desc.length > maxChars ? "" : "muted"}`}>{desc.length}/{maxChars}</span>
              </div>
            </>
          ) : (
            <p className="ink2 small">A vision-language model ({info?.configured_models.vlm_generator ?? "configured generator"}) writes a
              description, which is then verified independently: the verifier only receives narrow per-claim questions and is
              cross-checked by the detector, grounding model and pixel analysis.</p>
          )}

          <details style={{ marginTop: 12 }}>
            <summary>Verification settings</summary>
            <fieldset style={{ marginTop: 8 }}>
              <legend>Evidence components</legend>
              <label className="toggle"><input type="checkbox" checked={settings.use_detector} onChange={(e) => set("use_detector", e.target.checked)} /> Object detector</label>
              <label className="toggle"><input type="checkbox" checked={settings.use_grounding} onChange={(e) => set("use_grounding", e.target.checked)} /> Open-vocabulary grounding</label>
              <label className="toggle"><input type="checkbox" checked={settings.use_vlm} onChange={(e) => set("use_vlm", e.target.checked)} /> Vision-language check</label>
              {noComponent && <div className="hint" role="alert">Enable at least one component.</div>}
            </fieldset>
            <fieldset style={{ marginTop: 8 }}>
              <legend>Decision</legend>
              <label className="toggle"><input type="checkbox" checked={settings.abstention} onChange={(e) => set("abstention", e.target.checked)} /> Allow abstention (insufficient evidence)</label>
              <label className="toggle"><input type="checkbox" checked={settings.disagreement_analysis} onChange={(e) => set("disagreement_analysis", e.target.checked)} /> Flag disagreements for review</label>
            </fieldset>
            {isVideo && (
              <fieldset style={{ marginTop: 8 }}>
                <legend>Video sampling</legend>
                <label className="field" htmlFor="strategy">Strategy</label>
                <select id="strategy" value={settings.sampling_strategy ?? "scene_aware"}
                  onChange={(e) => set("sampling_strategy", e.target.value as JobSettings["sampling_strategy"])}>
                  <option value="scene_aware">Scene-change aware</option>
                  <option value="uniform">Uniform</option>
                  <option value="interval">Fixed interval</option>
                </select>
                <label className="field" htmlFor="maxf" style={{ marginTop: 8 }}>Max sampled frames (2–32)</label>
                <input id="maxf" type="number" min={2} max={32} value={settings.max_frames ?? ""} placeholder={`${info?.limits.max_sampled_frames ?? 16} (server default)`}
                  onChange={(e) => set("max_frames", e.target.value ? Math.max(2, Math.min(32, Number(e.target.value))) : null)} />
              </fieldset>
            )}
          </details>

          <div className="row" style={{ marginTop: 14, justifyContent: "flex-end" }}>
            {!file && <span className="small muted">Choose a file first.</span>}
            {file && descMissing && <span className="small muted">Enter a description or switch mode.</span>}
            <button className="primary" disabled={!canSubmit} onClick={submit}>
              {submitting ? <><span className="spinner" style={{ width: 14, height: 14, borderWidth: 2 }} /> Uploading…</> : "Start verification"}
            </button>
          </div>
          {error != null && <div style={{ marginTop: 10 }}><ErrorState error={error} /></div>}
        </section>
      </div>
    </div>
  );
}
