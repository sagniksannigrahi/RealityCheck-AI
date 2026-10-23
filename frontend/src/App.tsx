import { Fragment, useEffect, useState } from "react";
import { BrowserRouter, NavLink, Route, Routes } from "react-router-dom";
import { api, getApiKey, setApiKey } from "./api/client";
import type { SystemInfo } from "./api/types";
import { ErrorState, Loading } from "./components/common";
import EvaluationPage from "./pages/EvaluationPage";
import HistoryPage from "./pages/HistoryPage";
import JobPage from "./pages/JobPage";
import UploadPage from "./pages/UploadPage";

function SettingsPage({ info, reload }: { info: SystemInfo | null; reload: () => void }) {
  const [key, setKey] = useState(getApiKey());
  const [saved, setSaved] = useState(false);
  return (
    <div className="stack" style={{ maxWidth: 760 }}>
      <h1>Settings & system</h1>
      <section className="card">
        <h2>API key</h2>
        <p className="small ink2">Required only if the server sets <code>RC_AUTH_REQUIRED=true</code>. Stored in this browser only.</p>
        <div className="row">
          <input type="password" aria-label="API key" value={key} onChange={(e) => { setKey(e.target.value); setSaved(false); }} style={{ maxWidth: 360 }} />
          <button className="primary" onClick={() => { setApiKey(key.trim()); setSaved(true); reload(); }}>Save</button>
          {saved && <span className="small muted" role="status">Saved.</span>}
        </div>
      </section>
      {info && (
        <section className="card">
          <h2>Models</h2>
          <dl className="kv">
            <dt>Mode</dt><dd>{info.demo_mode ? "DEMO (fake models)" : "real models"} · device {info.models.device}</dd>
            {Object.entries(info.configured_models).map(([k, v]) => <Fragment key={k}><dt>{k.replace(/_/g, " ")}</dt><dd>{v}</dd></Fragment>)}
            {Object.entries(info.models.components).map(([k, v]) => <Fragment key={k}><dt>{k} status</dt><dd>{v.loaded ? `loaded (${v.model})` : v.error ? `error: ${v.error}` : "not loaded yet (lazy)"}</dd></Fragment>)}
            <dt>Processing</dt><dd>{info.privacy.statement}</dd>
            <dt>Media retention</dt><dd>{info.privacy.media_retention_days} days</dd>
            <dt>Limits</dt><dd>{info.limits.max_image_mb} MB images · {info.limits.max_video_mb} MB / {info.limits.max_video_seconds}s videos · {info.limits.max_claims_per_job} claims per job</dd>
          </dl>
        </section>
      )}
    </div>
  );
}

export default function App() {
  const [info, setInfo] = useState<SystemInfo | null>(null);
  const [infoError, setInfoError] = useState<unknown>(null);
  const [tick, setTick] = useState(0);
  useEffect(() => { api.systemInfo().then((i) => { setInfo(i); setInfoError(null); }).catch(setInfoError); }, [tick]);

  return (
    <BrowserRouter>
      <header className="topbar">
        <div className="topbar-inner">
          <NavLink to="/" className="brand"><img src="/favicon.svg" alt="" /> RealityCheck AI</NavLink>
          <nav className="nav" aria-label="Main">
            <NavLink to="/" end>Verify</NavLink>
            <NavLink to="/history">History</NavLink>
            <NavLink to="/evaluation">Evaluation</NavLink>
            <NavLink to="/settings">Settings</NavLink>
          </nav>
          <span className="spacer" />
          {info?.demo_mode && <span className="tag warn" title="Fake models are in use">DEMO MODE</span>}
          {info && !info.demo_mode && <span className="tag">{info.privacy.processing === "local" ? "local models" : "external VLM"}</span>}
        </div>
      </header>
      <main>
        {infoError != null && <div style={{ marginBottom: 12 }}><ErrorState error={infoError} onRetry={() => setTick((t) => t + 1)} /></div>}
        <Routes>
          <Route path="/" element={<UploadPage info={info} />} />
          <Route path="/jobs/:jobId" element={<JobPage />} />
          <Route path="/history" element={<HistoryPage />} />
          <Route path="/evaluation" element={<EvaluationPage />} />
          <Route path="/settings" element={<SettingsPage info={info} reload={() => setTick((t) => t + 1)} />} />
          <Route path="*" element={<div className="state"><h2>Page not found</h2></div>} />
        </Routes>
        {!info && !infoError && <Loading label="Connecting to the API…" />}
      </main>
      <footer className="footer">
        RealityCheck AI · verdicts combine independent model outputs with rule-based checks; scores are not guarantees of truth.
      </footer>
    </BrowserRouter>
  );
}
