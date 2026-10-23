"""End-to-end verification pipeline (image and video).

input -> (optional) description generation -> atomic claim extraction -> independent evidence collection
-> claim-evidence decision (EAV-SA) -> dependency propagation -> report.

Generation and verification are separated: the generated description only flows through the claim
extractor; the verifier VLM receives narrow per-claim questions and never sees the generated text.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from PIL import Image

from app.claim_extraction.extractor import RuleBasedClaimExtractor
from app.core.imaging import crop, encode_jpeg, image_fingerprint, image_quality
from app.schemas.claims import AtomicClaim, ClaimExtractionResult
from app.schemas.evidence import (
    BoundingBox,
    ClaimVerification,
    EvidenceItem,
    ModelInfo,
    Verdict,
)
from app.services.fake_models import FINGERPRINT_KEY
from app.services.storage import StorageBackend, new_key
from app.verification.collector import EvidenceCollector, FrameContext
from app.verification.decision import Decision, apply_dependencies, decide
from app.verification.policy import PlattCalibrator, VerificationPolicy

PIPELINE_VERSION = "1.0.0"
CheckFn = Callable[[str], None]


@dataclass
class FrameRecord:
    frame_index: int
    timestamp_s: float
    storage_key: str | None
    reason: str
    scene_change_score: float | None
    width: int
    height: int


@dataclass
class PipelineResult:
    description_used: str
    generated_description: str | None
    extraction: ClaimExtractionResult
    results: list[ClaimVerification]
    evidence: list[EvidenceItem]
    frames: list[FrameRecord] = field(default_factory=list)
    model_executions: list[dict[str, Any]] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    demo: bool = False
    timings_ms: dict[str, float] = field(default_factory=dict)
    media_info: dict[str, Any] = field(default_factory=dict)


def summarize(results: list[ClaimVerification]) -> dict[str, Any]:
    counts = {v.value: 0 for v in Verdict}
    for r in results:
        counts[r.verdict.value] += 1
    decided = counts["SUPPORTED"] + counts["CONTRADICTED"]
    verifiable = len(results) - counts["UNVERIFIABLE"]
    return {
        "total_claims": len(results),
        "verdict_counts": counts,
        "verifiable_claims": verifiable,
        "decided_claims": decided,
        "coverage": round(decided / verifiable, 4) if verifiable else 0.0,
        "contradicted_fraction_of_decided": round(counts["CONTRADICTED"] / decided, 4) if decided else None,
        "review_recommended": sum(1 for r in results if r.review_recommended),
        "headline": _headline(counts, len(results)),
    }


def _headline(c: dict[str, int], total: int) -> str:
    if total == 0:
        return "No claims were extracted from the description."
    parts = [f"{c['SUPPORTED']} supported", f"{c['CONTRADICTED']} contradicted",
             f"{c['INSUFFICIENT_EVIDENCE']} insufficient evidence"]
    if c["REVIEW_REQUIRED"]:
        parts.append(f"{c['REVIEW_REQUIRED']} need review")
    if c["UNVERIFIABLE"]:
        parts.append(f"{c['UNVERIFIABLE']} not visually verifiable")
    return f"{total} claims: " + ", ".join(parts) + "."


class VerificationPipeline:
    def __init__(self, registry, *, policy: VerificationPolicy | None = None,
                 calibrator: PlattCalibrator | None = None, extractor=None, storage: StorageBackend | None = None,
                 settings=None) -> None:
        from app.core.config import get_settings

        self.registry = registry
        self.settings = settings or get_settings()
        self.policy = policy or VerificationPolicy.load(self.settings.policy_path)
        self.calibrator = calibrator if calibrator is not None else PlattCalibrator.load(self.settings.calibration_path)
        self.extractor = extractor or RuleBasedClaimExtractor()
        self.storage = storage

    # ------------------------------------------------------------------ helpers
    def _crop_sink(self, job_id: str) -> Callable[[Image.Image, BoundingBox, int | None], str | None]:
        cache: dict[tuple, str] = {}

        def sink(img: Image.Image, box: BoundingBox, frame_index: int | None) -> str | None:
            if self.storage is None:
                return None
            k = (frame_index, round(box.x1, 3), round(box.y1, 3), round(box.x2, 3), round(box.y2, 3))
            if k not in cache:
                key = new_key("crops", "jpg", job_id)
                self.storage.put(key, encode_jpeg(crop(img, box, pad=0.08), quality=85, max_side=384))
                cache[k] = key
            return cache[k]

        return sink

    def _models(self) -> dict[str, Any]:
        reg = self.registry
        comps = self.policy.components
        return {
            "detector": reg.detector if comps.detector else None,
            "grounder": reg.grounder if comps.grounding else None,
            "verifier": reg.verifier if comps.vlm else None,
        }

    @staticmethod
    def _model_infos(*comps) -> list[ModelInfo]:
        return [c.info for c in comps if c is not None and getattr(c, "info", None) is not None]

    def _extract(self, description: str, video: bool) -> ClaimExtractionResult:
        return self.extractor.extract(description, max_claims=self.settings.max_claims_per_job, video=video)

    def _executions(self, collector: EvidenceCollector, models: dict[str, Any], extra: list[dict]) -> list[dict]:
        out = list(extra)
        names = {"detector": models.get("detector"), "grounding": models.get("grounder"), "vlm": models.get("verifier")}
        for comp, st in collector.stats.items():
            m = names.get(comp)
            info = getattr(m, "info", None)
            out.append({"component": comp, "model_name": info.name if info else comp,
                        "model_version": info.version if info else None, "device": info.device if info else None,
                        "calls": st.calls, "total_ms": round(st.total_ms, 1), "demo": bool(info and info.demo),
                        "external": bool(info and info.external)})
        return out

    def _to_verification(self, claim: AtomicClaim, d: Decision, models: list[ModelInfo], ms: float) -> ClaimVerification:
        return ClaimVerification(
            claim_id=claim.claim_id, claim_text=claim.text, claim_type=claim.claim_type.value, verdict=d.verdict,
            evidence_case=d.case, verification_score=d.score, calibrated_probability=d.calibrated_probability,
            evidence_quality=d.quality, supporting_evidence=d.supporting, contradicting_evidence=d.contradicting,
            considered_evidence=d.considered, explanation=d.explanation, uncertainty_reasons=d.uncertainty,
            review_recommended=d.review, models=models, decision_trace=d.trace, duration_ms=round(ms, 1),
        )

    def _generate(self, image: Image.Image) -> str:
        gen = self.registry.generator
        with self.registry.inference():
            return gen.describe(image)

    # ------------------------------------------------------------------ image
    def run_image(self, image: Image.Image, description: str | None, *, generate: bool, job_id: str,
                  check: CheckFn | None = None) -> PipelineResult:
        check = check or (lambda stage: None)
        timings: dict[str, float] = {}
        warnings: list[str] = []
        image.info[FINGERPRINT_KEY] = image.info.get(FINGERPRINT_KEY) or image_fingerprint(image)
        t0 = time.perf_counter()
        generated = None
        extra_exec: list[dict] = []
        if generate:
            check("generation")
            tg = time.perf_counter()
            generated = self._generate(image)
            gen_info = self.registry.generator.info
            extra_exec.append({"component": "vlm_generator", "model_name": gen_info.name, "model_version": gen_info.version,
                               "device": gen_info.device, "calls": 1, "total_ms": round((time.perf_counter() - tg) * 1000, 1),
                               "demo": gen_info.demo, "external": gen_info.external,
                               "shares_weights_with_verifier": self.registry.generator_shares_verifier_weights()})
            if not generated:
                warnings.append("The description generator returned no text.")
            if self.registry.generator_shares_verifier_weights():
                warnings.append("Generator and verifier VLM share weights; verification of generated text also relies "
                                "on independent detector, grounding and pixel evidence to limit confirmation bias.")
            timings["generation"] = (time.perf_counter() - tg) * 1000
        desc = generated if generate else (description or "")
        check("claim_extraction")
        te = time.perf_counter()
        extraction = self._extract(desc, video=False)
        timings["claim_extraction"] = (time.perf_counter() - te) * 1000
        warnings.extend(extraction.warnings)

        models = self._models()
        collector = EvidenceCollector(detector=models["detector"], grounder=models["grounder"], vlm=models["verifier"],
                                      policy=self.policy, crop_sink=self._crop_sink(job_id),
                                      inference_guard=self.registry.inference)
        quality = image_quality(image)
        ctx = FrameContext(image=image, quality=quality)
        decisions: dict[str, Decision] = {}
        evidence: list[EvidenceItem] = []
        durations: dict[str, float] = {}
        infos = self._model_infos(models["detector"], models["grounder"], models["verifier"])
        tv = time.perf_counter()
        for claim in extraction.claims:
            check(f"claim {claim.claim_id}")
            tc = time.perf_counter()
            items = collector.collect(claim, ctx)
            decisions[claim.claim_id] = decide(claim, items, self.policy, image_quality=quality, calibrator=self.calibrator)
            evidence.extend(items)
            durations[claim.claim_id] = (time.perf_counter() - tc) * 1000
        apply_dependencies(extraction.claims, decisions, self.policy)
        timings["verification"] = (time.perf_counter() - tv) * 1000
        results = [self._to_verification(c, decisions[c.claim_id], infos, durations[c.claim_id]) for c in extraction.claims]
        timings["total"] = (time.perf_counter() - t0) * 1000
        return PipelineResult(
            description_used=extraction.description, generated_description=generated, extraction=extraction,
            results=results, evidence=evidence, model_executions=self._executions(collector, models, extra_exec),
            summary={**summarize(results), "image_quality": quality, "policy": self.policy.name,
                     "policy_version": self.policy.version},
            warnings=warnings, demo=self.registry.demo, timings_ms={k: round(v, 1) for k, v in timings.items()},
            media_info={"width": image.size[0], "height": image.size[1]},
        )

    # ------------------------------------------------------------------ video
    def run_video(self, path, description: str | None, *, generate: bool, job_id: str,
                  check: CheckFn | None = None) -> PipelineResult:
        from app.video.sampling import coverage, probe_video, sample_frames, validate_video
        from app.video.temporal import verify_video_claim

        s = self.settings
        check = check or (lambda stage: None)
        timings: dict[str, float] = {}
        warnings: list[str] = []
        t0 = time.perf_counter()
        info = probe_video(path)
        validate_video(info, max_seconds=s.max_video_seconds, max_side=s.max_video_side)
        check("frame_sampling")
        frames = sample_frames(path, info, strategy=s.video_sampling_strategy, max_frames=s.video_max_sampled_frames,
                               interval_s=s.video_interval_seconds, scene_threshold=s.video_scene_threshold)
        cov = coverage(frames, info.duration_s)
        timings["frame_sampling"] = (time.perf_counter() - t0) * 1000
        frame_records: list[FrameRecord] = []
        ctxs: list[FrameContext] = []
        for f in frames:
            f.image.info[FINGERPRINT_KEY] = image_fingerprint(f.image)
            key = None
            if self.storage is not None:
                key = new_key("frames", "jpg", job_id)
                self.storage.put(key, encode_jpeg(f.image, quality=85, max_side=960))
            frame_records.append(FrameRecord(f.index, f.timestamp_s, key, f.reason, f.scene_change_score,
                                             f.image.size[0], f.image.size[1]))
            ctxs.append(FrameContext(image=f.image, frame_index=f.index, timestamp_s=f.timestamp_s,
                                     quality=image_quality(f.image)))

        generated = None
        extra_exec: list[dict] = []
        if generate:
            check("generation")
            tg = time.perf_counter()
            key_frames = [ctxs[0], ctxs[len(ctxs) // 2], ctxs[-1]] if len(ctxs) >= 3 else ctxs
            texts = []
            for kf in key_frames:
                txt = self._generate(kf.image)
                if txt:
                    texts.append(txt.strip())
            generated = " ".join(dict.fromkeys(texts))
            gi = self.registry.generator.info
            extra_exec.append({"component": "vlm_generator", "model_name": gi.name, "model_version": gi.version,
                               "device": gi.device, "calls": len(key_frames),
                               "total_ms": round((time.perf_counter() - tg) * 1000, 1), "demo": gi.demo,
                               "external": gi.external,
                               "shares_weights_with_verifier": self.registry.generator_shares_verifier_weights()})
            warnings.append("Generated video descriptions are produced from up to 3 keyframes (start, middle, end).")
            timings["generation"] = (time.perf_counter() - tg) * 1000

        desc = generated if generate else (description or "")
        extraction = self._extract(desc, video=True)
        warnings.extend(extraction.warnings)
        models = self._models()
        collector = EvidenceCollector(detector=models["detector"], grounder=models["grounder"], vlm=models["verifier"],
                                      policy=self.policy, crop_sink=self._crop_sink(job_id),
                                      inference_guard=self.registry.inference)
        # detector on every sampled frame (cheap) to build presence timelines
        check("detection")
        for c in ctxs:
            collector.detections(c)
        infos = self._model_infos(models["detector"], models["grounder"], models["verifier"])
        decisions: dict[str, Decision] = {}
        evidence: list[EvidenceItem] = []
        durations: dict[str, float] = {}
        frame_verdicts: dict[str, list] = {}
        tv = time.perf_counter()
        for claim in extraction.claims:
            check(f"claim {claim.claim_id}")
            tc = time.perf_counter()
            res = verify_video_claim(claim, ctxs, collector, self.policy, k_frames=s.video_max_vlm_frames_per_claim,
                                     max_gap_s=cov["max_gap_s"], calibrator=self.calibrator)
            decisions[claim.claim_id] = res.decision
            evidence.extend(res.items)
            frame_verdicts[claim.claim_id] = res.frame_verdicts
            durations[claim.claim_id] = (time.perf_counter() - tc) * 1000
        apply_dependencies(extraction.claims, decisions, self.policy)
        timings["verification"] = (time.perf_counter() - tv) * 1000
        results = [self._to_verification(c, decisions[c.claim_id], infos, durations[c.claim_id]) for c in extraction.claims]
        timings["total"] = (time.perf_counter() - t0) * 1000
        summary = {**summarize(results), "policy": self.policy.name, "policy_version": self.policy.version,
                   "sampling": {"strategy": s.video_sampling_strategy, **cov}, "frame_verdicts": frame_verdicts}
        return PipelineResult(
            description_used=extraction.description, generated_description=generated, extraction=extraction,
            results=results, evidence=evidence, frames=frame_records,
            model_executions=self._executions(collector, models, extra_exec), summary=summary, warnings=warnings,
            demo=self.registry.demo, timings_ms={k: round(v, 1) for k, v in timings.items()},
            media_info={"width": info.width, "height": info.height, "fps": info.fps, "duration_s": info.duration_s,
                        "frame_count": info.frame_count, "codec": info.fourcc},
        )
