"""Application configuration.

All settings are read from environment variables prefixed with ``RC_`` (see ``.env.example``).
Secrets are typed as ``SecretStr`` so they are never printed in logs or reprs.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Hugging Face commits the default models were tested and evaluated with. Pinning stops an upstream repo change
# from silently swapping the weights or loading code (see docs/SECURITY.md).
DEFAULT_MODEL_REVISIONS: dict[str, str] = {
    "IDEA-Research/grounding-dino-tiny": "a2bb814dd30d776dcf7e30523b00659f4f141c71",
    "HuggingFaceTB/SmolVLM-500M-Instruct": "a7da5b986cb59b408707209984f360a5f4ad7e47",
}


class ModelBackend(StrEnum):
    """Which model adapters the pipeline uses.

    - ``real``: local ML models (YOLO, Grounding DINO, SmolVLM). Fails readiness if unavailable.
    - ``demo``: deterministic fake adapters. Results are labelled DEMO everywhere and are NOT evaluations.
    - ``auto``: ``real`` if the ML dependencies import, otherwise ``demo``.
    """

    REAL = "real"
    DEMO = "demo"
    AUTO = "auto"


class WorkerMode(StrEnum):
    INLINE = "inline"  # API process runs a background worker thread (single-node dev)
    EXTERNAL = "external"  # separate `python -m app.workers.worker` process(es)


class VLMProvider(StrEnum):
    SMOLVLM = "smolvlm"  # local, runs on this machine
    ANTHROPIC = "anthropic"  # external API; media leaves the machine; requires explicit opt-in


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RC_", env_file=".env", extra="ignore")

    # --- general ---
    environment: str = "development"
    log_level: str = "INFO"
    log_json: bool = True
    secret_key: SecretStr = SecretStr("dev-insecure-change-me")

    # --- database ---
    database_url: str = f"sqlite:///{PROJECT_ROOT / 'data' / 'realitycheck.db'}"
    db_echo: bool = False

    # --- storage ---
    storage_backend: str = "local"  # local | s3
    storage_root: Path = PROJECT_ROOT / "data" / "storage"
    s3_bucket: str | None = None
    s3_endpoint_url: str | None = None
    media_retention_days: int = 30
    signed_url_ttl_seconds: int = 3600

    # --- auth ---
    auth_required: bool = False
    api_keys: SecretStr = SecretStr("")  # comma-separated raw keys; owner id = sha256(key)[:16]
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    rate_limit_per_minute: int = 60

    # --- upload limits ---
    max_image_bytes: int = 15 * 1024 * 1024
    max_video_bytes: int = 100 * 1024 * 1024
    max_image_pixels: int = 40_000_000
    max_image_side: int = 8192
    max_video_seconds: float = 120.0
    max_video_side: int = 3840
    max_description_chars: int = 4000
    max_claims_per_job: int = 40

    # --- video sampling ---
    video_sampling_strategy: str = "scene_aware"  # uniform | interval | scene_aware
    video_max_sampled_frames: int = 16
    video_interval_seconds: float = 1.0
    video_scene_threshold: float = 0.35
    video_max_vlm_frames_per_claim: int = 3
    video_max_grounding_frames: int = 6

    # --- processing ---
    job_timeout_seconds: int = 900
    job_max_attempts: int = 2
    worker_mode: WorkerMode = WorkerMode.INLINE
    worker_poll_seconds: float = 1.0
    worker_stale_lock_seconds: int = 1200

    # --- models ---
    model_backend: ModelBackend = ModelBackend.AUTO
    device: str = "cpu"  # cpu | mps | cuda | auto
    max_concurrent_inference: int = 1
    model_idle_unload_seconds: int = 0  # 0 = keep loaded
    detector_weights: str = "yolo11n.pt"
    grounding_model_id: str = "IDEA-Research/grounding-dino-tiny"
    vlm_provider: VLMProvider = VLMProvider.SMOLVLM
    vlm_model_id: str = "HuggingFaceTB/SmolVLM-500M-Instruct"
    generator_model_id: str = "HuggingFaceTB/SmolVLM-500M-Instruct"
    # SmolVLM image tiling: ~10x slower on CPU when enabled (measured 3.6 s vs 0.36 s per question)
    vlm_image_splitting: bool = False
    model_cache_dir: Path | None = None
    # Hub model id -> commit hash. Override with RC_MODEL_REVISIONS='{"org/model": "<sha>"}'; models not listed
    # load the latest revision. RC_PIN_MODEL_REVISIONS=false always loads the latest.
    model_revisions: dict[str, str] = Field(default_factory=lambda: dict(DEFAULT_MODEL_REVISIONS))
    pin_model_revisions: bool = True

    # --- external providers (opt-in only) ---
    allow_external_providers: bool = False
    anthropic_api_key: SecretStr | None = None
    anthropic_model: str = "claude-opus-5-5"
    anthropic_effort: str = "low"

    # --- verification policy ---
    policy_path: Path | None = None  # optional JSON VerificationPolicy override
    calibration_path: Path | None = None  # optional calibrator fitted by the evaluation harness

    # --- frontend ---
    frontend_dist: Path = PROJECT_ROOT / "frontend" / "dist"

    demo_manifest_dir: Path = PROJECT_ROOT / "datasets" / "demo"

    @field_validator("device")
    @classmethod
    def _device(cls, v: str) -> str:
        if v not in {"cpu", "mps", "cuda", "auto"}:
            raise ValueError("device must be cpu|mps|cuda|auto")
        return v

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def api_key_list(self) -> list[str]:
        return [k.strip() for k in self.api_keys.get_secret_value().split(",") if k.strip()]

    @property
    def is_sqlite(self) -> bool:
        return self.database_url.startswith("sqlite")

    def revision_for(self, model_id: str) -> str | None:
        """Pinned Hub commit for ``model_id``, or None to load the latest revision."""
        return self.model_revisions.get(model_id) if self.pin_model_revisions else None

    external_media_note: str = Field(
        default="Media is sent to an external provider only when RC_ALLOW_EXTERNAL_PROVIDERS=true "
        "and RC_VLM_PROVIDER=anthropic.",
        exclude=True,
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()


def reset_settings_cache() -> None:
    get_settings.cache_clear()
