// Types mirroring backend/app/schemas (keep in sync with the OpenAPI schema at /api/openapi.json).

export type Verdict = "SUPPORTED" | "CONTRADICTED" | "INSUFFICIENT_EVIDENCE" | "REVIEW_REQUIRED" | "UNVERIFIABLE";
export type JobStatus = "queued" | "running" | "completed" | "failed" | "cancelled";
export type Mode = "verify_description" | "generate_and_verify";

export interface Box { x1: number; y1: number; x2: number; y2: number }

export interface EntityRef {
  entity_id: string; head: string; phrase: string; canonical_class: string | null; category: string | null;
  is_scene: boolean; plural: boolean;
}

export interface AtomicClaim {
  claim_id: string; text: string; claim_type: string; subject: EntityRef | null; predicate: string | null;
  attribute: string | null; object: EntityRef | null; spatial_relation: string | null; temporal_scope: string | null;
  count: number | null; count_comparator: string | null; negated: boolean; hedged: boolean; verifiable: boolean;
  unverifiable_reason: string | null; required_evidence: string[]; depends_on: string[];
  source_span: { start: number; end: number } | null; sentence_index: number; flags: string[];
}

export interface ClaimResult {
  verdict: Verdict; evidence_case: string; verification_score: number | null; calibrated_probability: number | null;
  evidence_quality: "high" | "medium" | "low" | "none"; explanation: string; uncertainty_reasons: string[];
  review_recommended: boolean; supporting_evidence: string[]; contradicting_evidence: string[];
  considered_evidence: string[]; decision_trace: string[]; models: Record<string, unknown>[]; duration_ms: number;
}

export interface ClaimOut { claim: AtomicClaim; result: ClaimResult | null }

export interface Evidence {
  evidence_id: string; claim_id: string | null; source: string; model_name: string | null;
  model_version: string | null; stance: "supports" | "contradicts" | "neutral" | "absent" | "not_applicable";
  strength: "strong" | "weak" | "none"; raw_score: number | null; box: Box | null; label: string | null;
  frame_index: number | null; timestamp_s: number | null; crop_url: string | null; description: string;
  details: Record<string, unknown>;
}

export interface Frame {
  frame_index: number; timestamp_s: number; url: string | null; reason: string;
  scene_change_score: number | null; width: number; height: number;
}

export interface Media {
  media_type: "image" | "video"; mime_type: string; url: string | null; width: number | null; height: number | null;
  duration_s: number | null; fps: number | null; frame_count: number | null; size_bytes: number;
  original_filename: string | null;
}

export interface JobSummary {
  job_id: string; status: JobStatus; mode: Mode; media_type: "image" | "video"; created_at: string;
  started_at: string | null; finished_at: string | null; progress_stage: string | null; demo_mode: boolean;
  attempts: number; description_original: string | null; description_generated: string | null;
  description_used: string | null; summary: JobRunSummary | null; warnings: string[];
  settings: Record<string, unknown>; error: { code: string | null; message: string | null } | null;
  media: Media | null; pipeline_version: string | null; policy: string | null; timings_ms: Record<string, number> | null;
}

export interface JobRunSummary {
  total_claims: number; verdict_counts: Record<Verdict, number>; verifiable_claims: number; decided_claims: number;
  coverage: number; contradicted_fraction_of_decided: number | null; review_recommended: number; headline: string;
  image_quality?: Record<string, number>; sampling?: { strategy: string; max_gap_s: number; frames: number; duration_s: number };
  frame_verdicts?: Record<string, [number, number, string][]>;
}

export interface Page<T> { items: T[]; total: number; limit: number; offset: number }

export interface HistoryItem {
  job_id: string; status: JobStatus; mode: Mode; media_type: "image" | "video"; created_at: string;
  finished_at: string | null; demo_mode: boolean; description_preview: string | null;
  verdict_counts: Record<Verdict, number> | null; headline: string | null; thumbnail_url: string | null;
}

export interface HistoryStats {
  jobs_by_status: Record<string, number>; claims_by_verdict: Record<string, number>; total_jobs: number;
}

export interface JobSettings {
  abstention: boolean; disagreement_analysis: boolean; use_detector: boolean; use_grounding: boolean;
  use_vlm: boolean; sampling_strategy?: "uniform" | "interval" | "scene_aware" | null; max_frames?: number | null;
}

export interface SystemInfo {
  demo_mode: boolean;
  models: { device: string; components: Record<string, { loaded: boolean; model: string | null; error: string | null }> };
  configured_models: Record<string, string>;
  limits: { max_image_mb: number; max_video_mb: number; max_video_seconds: number; max_description_chars: number;
    max_claims_per_job: number; max_sampled_frames: number };
  privacy: { processing: "local" | "external"; statement: string; media_retention_days: number };
  auth_required: boolean; authenticated: boolean;
}

export interface CoreMetrics {
  n: number; n_covered: number; coverage: number | null; abstention_rate: number | null;
  accuracy_covered: number | null; accuracy_effective: number | null; macro_f1: number | null; micro_f1: number | null;
  hallucination_precision: number | null; hallucination_recall: number | null; hallucination_f1: number | null;
  hallucination_recall_effective: number | null; supported_precision: number | null; supported_recall: number | null;
  supported_f1: number | null; false_positive_rate: number | null; false_negative_rate: number | null;
  mean_latency_ms: number | null;
}

export interface SystemMetrics {
  description: string; selected_params: Record<string, number>; val_objective: number | null;
  overall: CoreMetrics; confidence_intervals_95: Record<string, { low: number | null; high: number | null }>;
  confusion_matrix: Record<string, Record<string, number>>; per_category: Record<string, CoreMetrics>;
  ambiguous_cases: CoreMetrics | null; calibration: { ece: number | null; brier: number | null; n_calibration: number };
  comparisons?: Record<string, { metric: string; diff: number | null; ci_low: number | null; ci_high: number | null; p_value: number | null }>;
  runtime?: { evidence_collection_s: number; peak_rss_mb: number };
}

export interface EvaluationRun {
  run_id: string; experiment: string; system: string; dataset: string; dataset_version: string | null; split: string;
  seed: number; status: string; simulated: boolean; n_examples: number; model_versions: Record<string, string>;
  created_at: string; finished_at: string | null; config: Record<string, unknown>; metrics: SystemMetrics;
}

export interface EvaluationList {
  items: EvaluationRun[]; total: number; experiments: { experiment: string; dataset: string; split: string }[];
}
