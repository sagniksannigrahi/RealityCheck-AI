"""Local SmolVLM adapter (HuggingFaceTB/SmolVLM-*-Instruct, Apache-2.0) via transformers.

Yes/no verification reads the next-token distribution and reports P(yes) = p(Yes-tokens) /
(p(Yes-tokens) + p(No-tokens)). This is a model-native score, not a calibrated probability.
"""

from __future__ import annotations

from PIL import Image

from app.schemas.evidence import ModelInfo
from app.vlm.base import DESCRIBE_PROMPT, VLMJudgment


class SmolVLMAdapter:
    def __init__(self, model_id: str = "HuggingFaceTB/SmolVLM-500M-Instruct", device: str = "cpu",
                 cache_dir: str | None = None, max_image_side: int = 768, role: str = "verifier",
                 image_splitting: bool = False, revision: str | None = None) -> None:
        import torch
        import transformers
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self._torch = torch
        self._processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir, revision=revision)
        if hasattr(self._processor, "image_processor"):
            self._processor.image_processor.do_image_splitting = image_splitting
        self._model = AutoModelForImageTextToText.from_pretrained(
            model_id, dtype=torch.float32, cache_dir=cache_dir, revision=revision).eval()
        self._model.to(device)
        self._device = device
        self._max_side = max_image_side
        tok = self._processor.tokenizer
        self._yes_ids = sorted({tok.encode(w, add_special_tokens=False)[0] for w in ["Yes", " Yes", "yes", " yes"]})
        self._no_ids = sorted({tok.encode(w, add_special_tokens=False)[0] for w in ["No", " No", "no", " no"]})
        self._info = ModelInfo(component=f"vlm_{role}", name=model_id,
                               version=f"transformers-{transformers.__version__}"
                               f"+tiling={'on' if image_splitting else 'off'}", device=device)

    @property
    def info(self) -> ModelInfo:
        return self._info

    def _prep(self, image: Image.Image) -> Image.Image:
        if max(image.size) > self._max_side:
            image = image.copy()
            image.thumbnail((self._max_side, self._max_side))
        return image

    def _inputs(self, image: Image.Image, text: str):
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
        prompt = self._processor.apply_chat_template(msgs, add_generation_prompt=True)
        return self._processor(text=prompt, images=[self._prep(image)], return_tensors="pt").to(self._device)

    def judge(self, image: Image.Image, question: str) -> VLMJudgment:
        text = f"{question} Answer yes or no."
        x = self._inputs(image, text)
        with self._torch.no_grad():
            logits = self._model(**x).logits[0, -1].float()
        probs = self._torch.softmax(logits, dim=-1)
        py = float(probs[self._yes_ids].sum())
        pn = float(probs[self._no_ids].sum())
        mass = py + pn
        p_yes = py / mass if mass > 0 else 0.5
        answer = "yes" if p_yes >= 0.5 else "no"
        if mass < 0.5:
            answer = "uncertain"
        return VLMJudgment(question=question, answer=answer, p_yes=round(p_yes, 4),
                           raw={"yes_no_mass": round(mass, 4)})

    def describe(self, image: Image.Image, prompt: str | None = None, max_new_tokens: int = 120) -> str:
        x = self._inputs(image, prompt or DESCRIBE_PROMPT)
        with self._torch.no_grad():
            out = self._model.generate(**x, max_new_tokens=max_new_tokens, do_sample=False)
        return self._processor.batch_decode(out[:, x["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
