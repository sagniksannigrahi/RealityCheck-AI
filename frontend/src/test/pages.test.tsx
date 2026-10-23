import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";
import { api, ApiError } from "../api/client";
import JobPage, { HighlightedDescription } from "../pages/JobPage";
import UploadPage, { validateFile } from "../pages/UploadPage";

afterEach(() => vi.restoreAllMocks());

const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json" } });

describe("validateFile", () => {
  it("rejects unsupported formats and oversize files", () => {
    expect(validateFile(new File(["x"], "a.pdf", { type: "application/pdf" }), null)).toMatch(/Unsupported/);
    const big = new File([new Uint8Array(16 * 1024 * 1024)], "a.png", { type: "image/png" });
    expect(validateFile(big, null)).toMatch(/limit is 15 MB/);
    expect(validateFile(new File(["x"], "a.png", { type: "image/png" }), null)).toBeNull();
  });
});

describe("api client", () => {
  it("parses the backend error envelope", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(json({ error: { code: "job_not_found", message: "Verification job not found.", request_id: "r1" } }, 404));
    await expect(api.job("x")).rejects.toMatchObject({ status: 404, code: "job_not_found", requestId: "r1" });
  });
  it("reports network failures clearly", async () => {
    vi.spyOn(globalThis, "fetch").mockRejectedValue(new TypeError("Failed to fetch"));
    await expect(api.job("x")).rejects.toBeInstanceOf(ApiError);
  });
});

describe("UploadPage", () => {
  it("requires a description in verify mode", async () => {
    render(<MemoryRouter><UploadPage info={null} /></MemoryRouter>);
    const btn = screen.getByRole("button", { name: /Start verification/ });
    expect(btn).toBeDisabled();
    await userEvent.upload(screen.getByTestId("file-input"), new File(["x"], "a.png", { type: "image/png" }));
    expect(btn).toBeDisabled();
    await userEvent.type(screen.getByLabelText("Description to verify"), "A dog.");
    expect(btn).toBeEnabled();
  });
});

describe("HighlightedDescription", () => {
  it("highlights the claim sentence and its terms", () => {
    const claim = { claim_id: "c1", text: "A dog is visible.", claim_type: "object_existence", subject: { entity_id: "e1", head: "dog", phrase: "dog", canonical_class: "dog", category: "animal", is_scene: false, plural: false },
      predicate: null, attribute: null, object: null, spatial_relation: null, temporal_scope: null, count: null, count_comparator: null, negated: false,
      hedged: false, verifiable: true, unverifiable_reason: null, required_evidence: [], depends_on: [], source_span: { start: 11, end: 26 }, sentence_index: 1, flags: [] };
    const { container } = render(<HighlightedDescription text="A cat sat. A dog barked." claim={claim} />);
    expect(container.querySelector("mark.claim-hl")).toHaveTextContent("A dog barked.");
    expect(container.querySelector("mark.term-hl")).toHaveTextContent("dog");
  });
});

describe("JobPage", () => {
  it("renders claims, verdicts and evidence from the API", async () => {
    const job = { job_id: "j1", status: "completed", mode: "verify_description", media_type: "image", created_at: "2026-10-23T00:00:00Z",
      started_at: null, finished_at: null, progress_stage: "completed", demo_mode: false, attempts: 1, description_original: "A dog.",
      description_generated: null, description_used: "A dog.", warnings: [], settings: {}, error: null, pipeline_version: "1.0.0", policy: "eav-sa",
      timings_ms: { total: 1200 },
      media: { media_type: "image", mime_type: "image/png", url: "/api/v1/files?key=media/a.png", width: 10, height: 10, duration_s: null, fps: null, frame_count: null, size_bytes: 1, original_filename: null },
      summary: { total_claims: 1, verdict_counts: { SUPPORTED: 0, CONTRADICTED: 1, INSUFFICIENT_EVIDENCE: 0, REVIEW_REQUIRED: 0, UNVERIFIABLE: 0 },
        verifiable_claims: 1, decided_claims: 1, coverage: 1, contradicted_fraction_of_decided: 1, review_recommended: 0, headline: "1 claims: 0 supported, 1 contradicted." } };
    const claims = { items: [{ claim: { claim_id: "c1", text: "A dog is visible.", claim_type: "object_existence", subject: null, predicate: "exists", attribute: null, object: null,
      spatial_relation: null, temporal_scope: null, count: null, count_comparator: null, negated: false, hedged: false, verifiable: true, unverifiable_reason: null,
      required_evidence: [], depends_on: [], source_span: { start: 0, end: 6 }, sentence_index: 0, flags: [] },
      result: { verdict: "CONTRADICTED", evidence_case: "contradictory", verification_score: -0.6, calibrated_probability: null, evidence_quality: "medium",
        explanation: "Contradicted by visual evidence.", uncertainty_reasons: [], review_recommended: false, supporting_evidence: [], contradicting_evidence: ["ev1"],
        considered_evidence: ["ev1"], decision_trace: ["score -0.6"], models: [], duration_ms: 12 } }], total: 1, limit: 200, offset: 0 };
    const evidence = { items: [{ evidence_id: "ev1", claim_id: "c1", source: "vlm", model_name: "smolvlm", model_version: "x", stance: "contradicts", strength: "strong",
      raw_score: 0.02, box: null, label: null, frame_index: null, timestamp_s: null, crop_url: null, description: "asked 'Is there a dog' -> no", details: { hypothesis_only: true } }],
      total: 1, limit: 1000, offset: 0 };
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const u = String(input);
      if (u.endsWith("/claims?limit=200")) return json(claims);
      if (u.includes("/evidence")) return json(evidence);
      return json(job);
    });
    render(<MemoryRouter initialEntries={["/jobs/j1"]}><Routes><Route path="/jobs/:jobId" element={<JobPage />} /></Routes></MemoryRouter>);
    await waitFor(() => expect(screen.getByRole("heading", { name: /c1: A dog is visible/ })).toBeInTheDocument());
    expect(screen.getAllByText("Contradicted").length).toBeGreaterThan(0);
    expect(screen.getByText(/Contradicting evidence \(1\)/)).toBeInTheDocument();
    expect(screen.getByText(/not a probability/)).toBeInTheDocument();
    expect(screen.getByRole("note")).toHaveTextContent(/No reliable localized evidence/);
  });
});
