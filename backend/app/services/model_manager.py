"""Managed model lifecycle.

- Models are loaded lazily, once per process, behind a lock (never per request).
- Inference is bounded by a semaphore (default 1) to protect memory on CPU-only machines.
- Idle models can be unloaded after ``RC_MODEL_IDLE_UNLOAD_SECONDS``.
- In DEMO mode (or AUTO without ML dependencies) deterministic fakes are used and every result is labelled.
"""

from __future__ import annotations

import gc
import importlib.util
import threading
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from app.core.config import ModelBackend, Settings, VLMProvider, get_settings
from app.observability.logging import get_logger, register_secret

log = get_logger(__name__)


class ModelUnavailableError(RuntimeError):
    pass


def ml_dependencies_available() -> bool:
    return all(importlib.util.find_spec(m) is not None for m in ("torch", "ultralytics", "transformers"))


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:  # noqa: BLE001, S110 - device probing is best-effort
        pass
    return "cpu"


@dataclass
class _Slot:
    factory: Callable[[], Any]
    instance: Any = None
    loaded_at: float | None = None
    last_used: float = 0.0
    load_seconds: float | None = None
    error: str | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)


class ModelRegistry:
    COMPONENTS = ("detector", "grounder", "verifier", "generator")

    def __init__(self, settings: Settings | None = None, *, force_demo: bool | None = None) -> None:
        self.settings = settings or get_settings()
        backend = self.settings.model_backend
        if force_demo is not None:
            self.demo = force_demo
        elif backend == ModelBackend.DEMO:
            self.demo = True
        elif backend == ModelBackend.REAL:
            self.demo = False
        else:
            self.demo = not ml_dependencies_available()
        self.device = resolve_device(self.settings.device)
        self._sem = threading.BoundedSemaphore(max(1, self.settings.max_concurrent_inference))
        self._slots: dict[str, _Slot] = {name: _Slot(factory=self._factory(name)) for name in self.COMPONENTS}
        if self.settings.anthropic_api_key:
            register_secret(self.settings.anthropic_api_key.get_secret_value())
        log.info("model registry initialised", extra={"demo": self.demo, "device": self.device})

    # ------------------------------------------------------------------ factories
    def _factory(self, name: str) -> Callable[[], Any]:
        s = self.settings
        cache = str(s.model_cache_dir) if s.model_cache_dir else None
        if self.demo:
            from app.services import fake_models as fm

            if s.demo_manifest_dir and not getattr(ModelRegistry, "_demo_loaded", False):
                fm.SCENES.load_manifest_dir(s.demo_manifest_dir)
                ModelRegistry._demo_loaded = True
            return {
                "detector": fm.FakeDetector,
                "grounder": fm.FakeGrounder,
                "verifier": lambda: fm.FakeVLM(role="verifier"),
                "generator": lambda: fm.FakeVLM(role="generator"),
            }[name]

        def detector():
            from app.detection.ultralytics_detector import UltralyticsDetector

            return UltralyticsDetector(weights=s.detector_weights, device=self.device)

        def grounder():
            from app.grounding.grounding_dino import GroundingDinoGrounder

            return GroundingDinoGrounder(model_id=s.grounding_model_id, device=self.device, cache_dir=cache,
                                         revision=s.revision_for(s.grounding_model_id))

        def vlm(role: str, model_id: str):
            def make():
                if s.vlm_provider == VLMProvider.ANTHROPIC:
                    if not s.allow_external_providers:
                        raise ModelUnavailableError(
                            "RC_VLM_PROVIDER=anthropic requires RC_ALLOW_EXTERNAL_PROVIDERS=true "
                            "(media would be sent to an external service)."
                        )
                    if not s.anthropic_api_key:
                        raise ModelUnavailableError("RC_ANTHROPIC_API_KEY is not set.")
                    from app.vlm.anthropic_vlm import AnthropicVLMAdapter

                    return AnthropicVLMAdapter(api_key=s.anthropic_api_key.get_secret_value(),
                                               model=s.anthropic_model, effort=s.anthropic_effort, role=role)
                from app.vlm.smolvlm import SmolVLMAdapter

                return SmolVLMAdapter(model_id=model_id, device=self.device, cache_dir=cache, role=role,
                                      image_splitting=s.vlm_image_splitting, revision=s.revision_for(model_id))

            return make

        return {
            "detector": detector,
            "grounder": grounder,
            "verifier": vlm("verifier", s.vlm_model_id),
            "generator": vlm("generator", s.generator_model_id),
        }[name]

    # ------------------------------------------------------------------ access
    def get(self, name: str) -> Any:
        slot = self._slots[name]
        if slot.instance is None:
            with slot.lock:
                if slot.instance is None:
                    # Share weights when generator and verifier are the same local model (saves ~2 GB RAM).
                    if name == "generator" and self._shares_weights():
                        slot.instance = self._shared_generator()
                    else:
                        t0 = time.perf_counter()
                        try:
                            slot.instance = slot.factory()
                        except ModelUnavailableError:
                            raise
                        except Exception as exc:  # noqa: BLE001
                            slot.error = f"{type(exc).__name__}: {exc}"
                            raise ModelUnavailableError(f"Failed to load {name}: {slot.error}") from exc
                        slot.load_seconds = round(time.perf_counter() - t0, 2)
                    slot.loaded_at = time.time()
                    slot.error = None
                    log.info("model loaded", extra={"component": name, "load_seconds": slot.load_seconds})
        slot.last_used = time.time()
        return slot.instance

    def _shares_weights(self) -> bool:
        s = self.settings
        return (not self.demo and s.vlm_provider == VLMProvider.SMOLVLM and s.vlm_model_id == s.generator_model_id)

    def _shared_generator(self):
        verifier = self.get("verifier")

        class _GeneratorView:
            """Same weights as the verifier, but a separate role with its own prompt and metadata."""

            def __init__(self, inner):
                self._inner = inner
                self.info = inner.info.model_copy(update={"component": "vlm_generator"})

            def describe(self, image, prompt=None, max_new_tokens=120):
                return self._inner.describe(image, prompt, max_new_tokens)

            def judge(self, image, question):
                return self._inner.judge(image, question)

        return _GeneratorView(verifier)

    @property
    def detector(self):
        return self.get("detector")

    @property
    def grounder(self):
        return self.get("grounder")

    @property
    def verifier(self):
        return self.get("verifier")

    @property
    def generator(self):
        return self.get("generator")

    @contextmanager
    def inference(self):
        """Bound concurrent heavy inference across threads."""
        self._sem.acquire()
        try:
            yield
        finally:
            self._sem.release()

    def generator_shares_verifier_weights(self) -> bool:
        return self._shares_weights() or self.demo

    # ------------------------------------------------------------------ lifecycle
    def unload(self, name: str | None = None) -> None:
        names = [name] if name else list(self._slots)
        for n in names:
            slot = self._slots[n]
            with slot.lock:
                slot.instance = None
                slot.loaded_at = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if hasattr(torch, "mps") and torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:  # noqa: BLE001, S110 - cache release is best-effort
            pass

    def unload_idle(self) -> list[str]:
        ttl = self.settings.model_idle_unload_seconds
        if ttl <= 0:
            return []
        now = time.time()
        out = [n for n, s in self._slots.items() if s.instance is not None and now - s.last_used > ttl]
        for n in out:
            self.unload(n)
        return out

    def status(self) -> dict[str, Any]:
        return {
            "demo": self.demo,
            "device": self.device,
            "ml_dependencies": ml_dependencies_available(),
            "vlm_provider": self.settings.vlm_provider.value,
            "external_provider_enabled": self.settings.allow_external_providers
            and self.settings.vlm_provider == VLMProvider.ANTHROPIC,
            "components": {
                n: {
                    "loaded": s.instance is not None,
                    "load_seconds": s.load_seconds,
                    "error": s.error,
                    "model": getattr(getattr(s.instance, "info", None), "name", None),
                }
                for n, s in self._slots.items()
            },
        }


_registry: ModelRegistry | None = None
_registry_lock = threading.Lock()


def get_registry() -> ModelRegistry:
    global _registry
    if _registry is None:
        with _registry_lock:
            if _registry is None:
                _registry = ModelRegistry()
    return _registry


def set_registry(reg: ModelRegistry | None) -> None:
    global _registry
    _registry = reg
