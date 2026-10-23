import pytest

from app.core.config import get_settings
from app.schemas.evidence import EvidenceSource, Verdict
from app.verification.pipeline import VerificationPipeline
from app.verification.policy import VerificationPolicy
from app.video.sampling import (
    VideoDecodeError,
    coverage,
    probe_video,
    sample_frames,
    select_indices,
    validate_video,
)
from tests.doubles import BlobDetector, BlobGrounder, ScriptedVLM, StaticRegistry, make_video

PERSON = (0.4, 0.3, 0.6, 0.9)
C2L = {"red": "person", "blue": "dog"}


@pytest.fixture()
def video_dir(tmp_path):
    return tmp_path


def _pipeline(vlm_answers=None, max_frames=8, strategy="uniform"):
    s = get_settings().model_copy(update={"video_max_sampled_frames": max_frames, "video_sampling_strategy": strategy,
                                          "video_max_vlm_frames_per_claim": 3})
    reg = StaticRegistry(BlobDetector(C2L), BlobGrounder(C2L), ScriptedVLM(vlm_answers or {}, C2L))
    return VerificationPipeline(reg, policy=VerificationPolicy(), settings=s, calibrator=None), reg


def test_probe_and_timestamps(video_dir):
    p = make_video(video_dir / "a.mp4", [[], [("red", PERSON)]] * 2, fps=10, repeat=5)
    info = probe_video(p)
    assert info.frame_count == 20 and abs(info.fps - 10) < 0.01 and abs(info.duration_s - 2.0) < 0.01
    frames = sample_frames(p, info, strategy="uniform", max_frames=5)
    assert [f.timestamp_s for f in frames] == [round(f.index / 10, 3) for f in frames]
    assert frames[0].index == 0 and len(frames) <= 5
    cov = coverage(frames, info.duration_s)
    assert cov["max_gap_s"] <= 1.0


def test_scene_aware_sampling_catches_scene_cut(video_dir):
    # 30 frames gray, then an abrupt cut to a full-red scene for 30 frames
    p = make_video(video_dir / "cut.mp4", [[]] * 6 + [[("red", (0.0, 0.0, 1.0, 1.0))]] * 6, fps=10, repeat=5)
    info = probe_video(p)
    idx = select_indices(info, "scene_aware", max_frames=4, interval_s=1.0, scene_threshold=0.35, path=p)
    cuts = [i for i, (reason, _) in idx.items() if reason == "scene_change"]
    assert cuts and any(28 <= i <= 32 for i in cuts)


def test_interval_sampling(video_dir):
    p = make_video(video_dir / "i.mp4", [[]] * 6, fps=10, repeat=5)
    idx = select_indices(probe_video(p), "interval", max_frames=10, interval_s=1.0, scene_threshold=0.3)
    assert sorted(idx) == [0, 10, 20]


def test_limits_reject_long_or_large_video(video_dir):
    p = make_video(video_dir / "long.mp4", [[]] * 10, fps=10, repeat=5)
    info = probe_video(p)
    with pytest.raises(VideoDecodeError):
        validate_video(info, max_seconds=2.0, max_side=4096)
    with pytest.raises(VideoDecodeError):
        validate_video(info, max_seconds=100, max_side=100)


def test_malformed_video_is_rejected(video_dir):
    bad = video_dir / "bad.mp4"
    bad.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 500)
    with pytest.raises(VideoDecodeError):
        probe_video(bad)


def test_person_entering_is_supported_with_timestamps(video_dir):
    # empty room for 2 s, then a person present for 2 s
    p = make_video(video_dir / "enter.mp4", [[]] * 4 + [[("red", PERSON)]] * 4, fps=10, repeat=5)
    pipe, _ = _pipeline()
    res = pipe.run_video(p, "A man enters the room.", generate=False, job_id="v1")
    r = {x.claim_text: x for x in res.results}
    v = r["The man enters the room."]
    assert v.verdict == Verdict.SUPPORTED
    temporal = [e for e in res.evidence if e.source == EvidenceSource.TEMPORAL]
    assert temporal and temporal[0].timestamp_s is not None and 1.5 <= temporal[0].timestamp_s <= 2.5
    assert r["A man is visible."].verdict == Verdict.SUPPORTED
    assert len(res.frames) >= 4 and all(f.timestamp_s >= 0 for f in res.frames)


def test_static_person_does_not_establish_entering(video_dir):
    p = make_video(video_dir / "static.mp4", [[("red", PERSON)]] * 8, fps=10, repeat=5)
    pipe, _ = _pipeline()
    res = pipe.run_video(p, "A man enters the room.", generate=False, job_id="v2")
    v = {x.claim_text: x for x in res.results}["The man enters the room."]
    assert v.verdict == Verdict.CONTRADICTED
    assert any("gaps" in u for u in v.uncertainty_reasons)


def test_single_frame_never_establishes_an_event(video_dir):
    p = make_video(video_dir / "one.mp4", [[("red", PERSON)]], fps=10, repeat=1)
    pipe, _ = _pipeline(max_frames=1)
    res = pipe.run_video(p, "A man enters the room.", generate=False, job_id="v3")
    v = {x.claim_text: x for x in res.results}["The man enters the room."]
    assert v.verdict == Verdict.INSUFFICIENT_EVIDENCE


def test_object_absent_from_all_frames_is_contradicted_with_coverage_note(video_dir):
    p = make_video(video_dir / "nodog.mp4", [[("red", PERSON)]] * 6, fps=10, repeat=5)
    pipe, _ = _pipeline()
    res = pipe.run_video(p, "A dog is lying on the floor.", generate=False, job_id="v4")
    v = {x.claim_text: x for x in res.results}["A dog is visible."]
    assert v.verdict == Verdict.CONTRADICTED
    assert any("sampled" in u for u in v.uncertainty_reasons)


def test_object_visible_in_some_frames_is_supported(video_dir):
    p = make_video(video_dir / "dogleaves.mp4", [[("blue", (0.1, 0.5, 0.3, 0.8))]] * 2 + [[]] * 6, fps=10, repeat=5)
    pipe, _ = _pipeline()
    res = pipe.run_video(p, "A dog is visible.", generate=False, job_id="v5")
    v = {x.claim_text: x for x in res.results}["A dog is visible."]
    assert v.verdict == Verdict.SUPPORTED


def test_negated_existence_over_video(video_dir):
    p = make_video(video_dir / "neg.mp4", [[]] * 6 + [[("blue", (0.1, 0.5, 0.3, 0.8))]] * 2, fps=10, repeat=5)
    pipe, _ = _pipeline(max_frames=8)
    res = pipe.run_video(p, "There is no dog.", generate=False, job_id="v6")
    v = {x.claim_text: x for x in res.results}["There is no dog."]
    assert v.verdict == Verdict.CONTRADICTED


def test_motion_analysis_flags_static_subject_for_motion_verb(video_dir):
    p = make_video(video_dir / "still.mp4", [[("red", PERSON)]] * 8, fps=10, repeat=5)
    pipe, _ = _pipeline({"is the man running": 0.95})
    res = pipe.run_video(p, "A man is running.", generate=False, job_id="v7")
    v = {x.claim_text: x for x in res.results}["The man is running."]
    assert v.verdict == Verdict.REVIEW_REQUIRED
    assert any(e.source == EvidenceSource.MOTION and e.stance.value == "contradicts" for e in res.evidence)


def test_moving_subject_supports_motion(video_dir):
    layouts = [[("red", (0.05 + 0.1 * i, 0.3, 0.2 + 0.1 * i, 0.9))] for i in range(8)]
    p = make_video(video_dir / "moving.mp4", layouts, fps=10, repeat=5)
    pipe, _ = _pipeline({"is the man running": 0.95})
    res = pipe.run_video(p, "A man is running.", generate=False, job_id="v8")
    v = {x.claim_text: x for x in res.results}["The man is running."]
    assert v.verdict == Verdict.SUPPORTED
    assert any(e.source == EvidenceSource.MOTION and e.stance.value == "supports" for e in res.evidence)


def test_vlm_calls_are_bounded_per_claim(video_dir):
    p = make_video(video_dir / "bounded.mp4", [[("red", PERSON)]] * 16, fps=10, repeat=3)
    pipe, reg = _pipeline({"is the man standing": 0.9}, max_frames=16)
    pipe.run_video(p, "A man is standing.", generate=False, job_id="v9")
    standing_q = [q for q in reg.verifier.questions if "standing" in q]
    assert 1 <= len(standing_q) <= 3
