"""Optional external VLM adapter using the Anthropic Messages API.

PRIVACY: this adapter sends the image (or crop) to an external service. It is only constructed when
RC_ALLOW_EXTERNAL_PROVIDERS=true and RC_VLM_PROVIDER=anthropic; it is never used by the test suite.
It has not been exercised against the live API in this repository's CI (no credentials are available).
"""

from __future__ import annotations

import base64
import io
import json
from typing import Literal

from PIL import Image
from pydantic import BaseModel, Field

from app.schemas.evidence import ModelInfo
from app.vlm.base import DESCRIBE_PROMPT, VERIFIER_PREAMBLE, VLMJudgment


class _StructuredJudgment(BaseModel):
    answer: Literal["yes", "no", "uncertain"]
    visible_evidence: str = Field(description="What is actually visible that bears on the statement.")
    undetermined: str = Field(description="What cannot be determined from the image.")
    relevant_region: str = Field(description="Where in the image the relevant evidence is, or 'none'.")
    conflicts: bool = Field(description="True if visible evidence conflicts with the statement.")


_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string", "enum": ["yes", "no", "uncertain"]},
        "visible_evidence": {"type": "string"},
        "undetermined": {"type": "string"},
        "relevant_region": {"type": "string"},
        "conflicts": {"type": "boolean"},
    },
    "required": ["answer", "visible_evidence", "undetermined", "relevant_region", "conflicts"],
    "additionalProperties": False,
}


def _b64_jpeg(image: Image.Image, max_side: int = 1024) -> str:
    img = image.copy()
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=88)
    return base64.standard_b64encode(buf.getvalue()).decode("ascii")


class AnthropicVLMAdapter:
    def __init__(self, api_key: str, model: str = "claude-opus-5-5", effort: str = "low", role: str = "verifier",
                 timeout_s: float = 60.0) -> None:
        import anthropic

        self._anthropic = anthropic
        self._client = anthropic.Anthropic(api_key=api_key, timeout=timeout_s, max_retries=2)
        self._model = model
        self._effort = effort
        self._info = ModelInfo(component=f"vlm_{role}", name=f"anthropic/{model}", version=anthropic.__version__,
                               external=True)

    @property
    def info(self) -> ModelInfo:
        return self._info

    def _create(self, content: list[dict], output_schema: dict | None, max_tokens: int):
        kwargs: dict = {
            "model": self._model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": content}],
            "output_config": {"effort": self._effort},
            # server-side refusal fallback (routes a declined request to a fallback model)
            "betas": ["server-side-fallback-2026-07-01"],
            "fallbacks": "default",
        }
        if output_schema is not None:
            kwargs["output_config"]["format"] = {"type": "json_schema", "schema": output_schema}
        return self._client.beta.messages.create(**kwargs)

    def judge(self, image: Image.Image, question: str) -> VLMJudgment:
        safe_q = question.replace("<", "(").replace(">", ")")
        content = [
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": _b64_jpeg(image)}},
            {"type": "text", "text": (
                f"{VERIFIER_PREAMBLE}Decide whether the statement is visibly true in this image. Answer 'uncertain' "
                f"if the image does not show enough to decide.\n<statement>{safe_q}</statement>"
            )},
        ]
        try:
            resp = self._create(content, _SCHEMA, max_tokens=4000)
        except self._anthropic.APIStatusError as exc:
            return VLMJudgment(question=question, answer="uncertain", raw={"error": f"api_status_{exc.status_code}"})
        except self._anthropic.APIConnectionError:
            return VLMJudgment(question=question, answer="uncertain", raw={"error": "connection_error"})
        if resp.stop_reason == "refusal":
            return VLMJudgment(question=question, answer="uncertain", raw={"error": "refusal"})
        text = next((b.text for b in resp.content if b.type == "text"), "")
        try:
            parsed = _StructuredJudgment.model_validate(json.loads(text))
        except Exception:  # noqa: BLE001 - malformed model output is treated as no evidence
            return VLMJudgment(question=question, answer="uncertain", raw={"error": "unparseable_output"})
        return VLMJudgment(
            question=question, answer=parsed.answer, p_yes=None, visible_evidence=parsed.visible_evidence,
            undetermined=parsed.undetermined, conflicts=parsed.conflicts, relevant_region=parsed.relevant_region,
            raw={"model": resp.model},
        )

    def describe(self, image: Image.Image, prompt: str | None = None, max_new_tokens: int = 300) -> str:
        content = [
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": _b64_jpeg(image)}},
            {"type": "text", "text": prompt or DESCRIBE_PROMPT},
        ]
        resp = self._create(content, None, max_tokens=max(2000, max_new_tokens))
        if resp.stop_reason == "refusal":
            return ""
        return next((b.text for b in resp.content if b.type == "text"), "").strip()
