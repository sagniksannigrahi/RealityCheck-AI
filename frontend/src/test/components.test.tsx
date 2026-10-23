import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { Evidence, Media } from "../api/types";
import { ConfusionMatrix, MetricBars, VerdictStack } from "../components/charts";
import { ErrorState, VerdictBadge } from "../components/common";
import { EvidenceViewer, overlayBoxes } from "../components/EvidenceViewer";
import { ApiError } from "../api/client";

const ev = (over: Partial<Evidence>): Evidence => ({
  evidence_id: "e1", claim_id: "c1", source: "detector", model_name: "m", model_version: "1", stance: "supports",
  strength: "strong", raw_score: 0.9, box: { x1: 0.1, y1: 0.1, x2: 0.5, y2: 0.5 }, label: "dog", frame_index: null,
  timestamp_s: null, crop_url: null, description: "detected dog", details: {}, ...over,
});

const media: Media = { media_type: "image", mime_type: "image/png", url: "/api/v1/files?key=media/a.png", width: 640,
  height: 480, duration_s: null, fps: null, frame_count: null, size_bytes: 10, original_filename: "a.png" };

describe("VerdictBadge", () => {
  it("shows a text label, not color alone", () => {
    render(<VerdictBadge verdict="INSUFFICIENT_EVIDENCE" />);
    expect(screen.getByText("Insufficient evidence")).toBeInTheDocument();
  });
});

describe("overlayBoxes", () => {
  it("classifies boxes by supporting/contradicting membership", () => {
    const boxes = overlayBoxes([ev({ evidence_id: "a" }), ev({ evidence_id: "b", box: { x1: 0.6, y1: 0.6, x2: 0.9, y2: 0.9 } })],
      new Set(["a"]), new Set(["b"]), null);
    expect(boxes.map((b) => b.kind)).toEqual(["support", "contradict"]);
  });
  it("never draws VLM-only evidence as a box", () => {
    expect(overlayBoxes([ev({ source: "vlm" })], new Set(["e1"]), new Set(), null)).toHaveLength(0);
  });
  it("draws subject and object boxes for geometry evidence", () => {
    const g = ev({ source: "geometry", details: { subject_box: { x1: 0, y1: 0, x2: 0.2, y2: 0.2 }, object_box: { x1: 0.5, y1: 0.5, x2: 0.9, y2: 0.9 } } });
    expect(overlayBoxes([g], new Set(), new Set(), null).map((b) => b.label)).toEqual(["geometry: subject", "geometry: object"]);
  });
  it("filters by frame for videos", () => {
    expect(overlayBoxes([ev({ frame_index: 3 })], new Set(), new Set(), 4)).toHaveLength(0);
    expect(overlayBoxes([ev({ frame_index: 3 })], new Set(), new Set(), 3)).toHaveLength(1);
  });
});

describe("EvidenceViewer", () => {
  it("states explicitly when a claim has no localized evidence", () => {
    render(<EvidenceViewer media={media} frames={[]} evidence={[ev({ source: "vlm", box: null })]} supporting={["e1"]} contradicting={[]} claimSelected />);
    expect(screen.getByRole("note")).toHaveTextContent(/No reliable localized evidence/);
  });
  it("renders the original media", () => {
    render(<EvidenceViewer media={media} frames={[]} evidence={[ev({})]} supporting={["e1"]} contradicting={[]} claimSelected />);
    expect(screen.getByAltText("Uploaded media")).toBeInTheDocument();
    expect(screen.queryByRole("note")).toBeNull();
  });
});

describe("charts", () => {
  it("confusion matrix prints every count", () => {
    render(<ConfusionMatrix matrix={{ SUPPORTED: { SUPPORTED: 5, CONTRADICTED: 1, ABSTAIN: 2 }, CONTRADICTED: { SUPPORTED: 0, CONTRADICTED: 7, ABSTAIN: 3 } }} />);
    expect(screen.getByText("7")).toBeInTheDocument();
    expect(screen.getAllByText("8").length).toBeGreaterThan(0); // row total
  });
  it("metric bars expose an accessible summary", () => {
    render(<MetricBars title="Macro-F1" data={[{ key: "a", label: "sys-a", value: 0.5, low: 0.4, high: 0.6 }]} />);
    expect(screen.getByRole("img")).toHaveAttribute("aria-label", expect.stringContaining("sys-a 50.0%"));
  });
  it("verdict stack shows legend counts", () => {
    render(<VerdictStack counts={{ SUPPORTED: 3, CONTRADICTED: 1 }} />);
    expect(screen.getByRole("img")).toHaveAttribute("aria-label", expect.stringContaining("Supported: 3"));
  });
});

describe("ErrorState", () => {
  it("shows API error code and request id", () => {
    render(<ErrorState error={new ApiError(422, "invalid_image", "Could not decode image", "rid123")} />);
    expect(screen.getByRole("alert")).toHaveTextContent("invalid_image");
    expect(screen.getByRole("alert")).toHaveTextContent("rid123");
  });
});
