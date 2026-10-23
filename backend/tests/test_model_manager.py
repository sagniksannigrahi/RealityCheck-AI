"""Model lifecycle: lazy shared instances, unload, external-provider opt-in."""

import pytest
from pydantic import SecretStr

from app.core.config import ModelBackend, Settings, VLMProvider
from app.services.fake_models import FakeDetector, FakeVLM
from app.services.model_manager import ModelRegistry, ModelUnavailableError


def test_models_are_lazy_and_shared():
    reg = ModelRegistry(force_demo=True)
    assert not any(c["loaded"] for c in reg.status()["components"].values())
    d1, d2 = reg.detector, reg.detector
    assert d1 is d2 and isinstance(d1, FakeDetector)
    assert reg.status()["components"]["detector"]["loaded"]
    assert isinstance(reg.verifier, FakeVLM)
    reg.unload()
    assert not reg.status()["components"]["detector"]["loaded"]
    assert reg.detector is not d1  # reloaded after unload


def test_external_provider_requires_explicit_opt_in():
    s = Settings(model_backend=ModelBackend.REAL, vlm_provider=VLMProvider.ANTHROPIC, allow_external_providers=False,
                 anthropic_api_key=SecretStr("sk-ant-test-key-123456"))
    reg = ModelRegistry(s, force_demo=False)
    with pytest.raises(ModelUnavailableError, match="ALLOW_EXTERNAL_PROVIDERS"):
        reg.get("verifier")
    assert reg.status()["external_provider_enabled"] is False


def test_external_provider_requires_key():
    s = Settings(model_backend=ModelBackend.REAL, vlm_provider=VLMProvider.ANTHROPIC, allow_external_providers=True,
                 anthropic_api_key=None)
    with pytest.raises(ModelUnavailableError, match="ANTHROPIC_API_KEY"):
        ModelRegistry(s, force_demo=False).get("verifier")


def test_demo_generator_and_verifier_roles():
    reg = ModelRegistry(force_demo=True)
    assert reg.generator.info.component == "vlm_generator"
    assert reg.verifier.info.component == "vlm_verifier"
    assert reg.generator_shares_verifier_weights() is True


def test_inference_semaphore_is_reentrant_safe():
    reg = ModelRegistry(force_demo=True)
    with reg.inference():
        pass
    with reg.inference():
        pass
