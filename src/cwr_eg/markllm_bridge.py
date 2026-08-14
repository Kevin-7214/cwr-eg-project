from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
from typing import Any


ALGORITHM_NAMES = {
    "kgw": "KGW",
    "unigram": "Unigram",
    "unbiased": "Unbiased",
    "synthid": "SynthID",
}


def _raw_secret(key_id: str, secret: Any | None = None) -> Any:
    environment_name = "CWR_EG_KEY_" + key_id.upper()
    raw = os.environ.get(environment_name) if secret is None else secret
    if raw is None:
        raise RuntimeError(
            f"Missing {environment_name}; watermark secrets must be supplied at runtime"
        )
    return raw


def _unbiased_key_bytes(raw: Any) -> bytes:
    value = int(raw)
    if not 0 < value < 2**1024:
        raise ValueError("Unbiased key must be a positive integer smaller than 2**1024")
    return value.to_bytes(128, "big")


def _key_override(key_id: str, secret: Any | None = None) -> dict[str, Any]:
    raw = _raw_secret(key_id, secret)
    family = key_id.rsplit("_key_", 1)[0]
    if family in {"kgw", "unigram"}:
        return {"hash_key": int(raw)}
    if family == "unbiased":
        return {"key": int(raw)}
    if family == "synthid":
        keys = json.loads(raw) if isinstance(raw, str) else raw
        if not isinstance(keys, list) or not keys or not all(isinstance(item, int) for item in keys):
            raise ValueError(f"{environment_name} must be a JSON integer list")
        return {"keys": keys}
    raise ValueError(f"Unsupported key family in {key_id}")


@dataclass(frozen=True, slots=True)
class MarkLlmSettings:
    repository: Path
    model_path: Path
    model_revision: str
    device: str
    max_new_tokens: int = 256
    do_sample: bool = True
    temperature: float = 0.8
    top_p: float = 0.95
    no_repeat_ngram_size: int = 4
    local_files_only: bool = True


class MarkLlmBridge:
    def __init__(self, settings: MarkLlmSettings) -> None:
        self.settings = settings
        if not (settings.repository / "watermark" / "auto_watermark.py").is_file():
            raise FileNotFoundError("Pinned MarkLLM checkout is missing")
        repository_text = str(settings.repository.resolve())
        if repository_text not in sys.path:
            sys.path.insert(0, repository_text)

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList
        from utils.transformers_config import TransformersConfig

        dtype = torch.bfloat16 if settings.device.startswith("cuda") else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(
            settings.model_path,
            revision=settings.model_revision,
            local_files_only=settings.local_files_only,
            use_fast=True,
            trust_remote_code=False,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            settings.model_path,
            revision=settings.model_revision,
            local_files_only=settings.local_files_only,
            torch_dtype=dtype,
            trust_remote_code=False,
            use_safetensors=True,
        ).to(settings.device)
        self.model.eval()
        output_vocab_size = int(self.model.get_output_embeddings().weight.shape[0])
        self.torch = torch
        self.logits_processor_list = LogitsProcessorList
        self.transformers_config = TransformersConfig(
            model=self.model,
            tokenizer=self.tokenizer,
            vocab_size=output_vocab_size,
            device=settings.device,
            max_new_tokens=settings.max_new_tokens,
            do_sample=settings.do_sample,
            temperature=settings.temperature,
            top_p=settings.top_p,
            no_repeat_ngram_size=settings.no_repeat_ngram_size,
            pad_token_id=self.tokenizer.eos_token_id,
        )

    def load_watermark(self, family: str, key_id: str | None, secret: Any | None = None):
        from watermark.auto_watermark import AutoWatermark

        algorithm_name = ALGORITHM_NAMES[family]
        config_path = self.settings.repository / "config" / f"{algorithm_name}.json"
        overrides = {} if key_id is None else _key_override(key_id, secret)
        watermark = AutoWatermark.load(
            algorithm_name,
            algorithm_config=str(config_path),
            transformers_config=self.transformers_config,
            **overrides,
        )
        if family == "unbiased" and key_id is not None:
            watermark.config.hash_key = _unbiased_key_bytes(_raw_secret(key_id, secret))
        return watermark

    def generate(
        self, prompt: str, family: str | None, key_id: str | None, *, seed: int
    ) -> str:
        self.torch.manual_seed(seed)
        if self.settings.device.startswith("cuda"):
            self.torch.cuda.manual_seed_all(seed)
        encoded = self.tokenizer(
            prompt, return_tensors="pt", add_special_tokens=True
        ).to(self.settings.device)
        generation_kwargs = dict(self.transformers_config.gen_kwargs)
        watermark = None
        if family is not None:
            if key_id is None:
                raise ValueError("Watermarked generation requires key_id")
            watermark = self.load_watermark(family, key_id)
            if family == "unbiased":
                watermark.utils.state_indicator = 0
            generation_kwargs["logits_processor"] = self.logits_processor_list(
                [watermark.logits_processor]
            )
        with self.torch.inference_mode():
            generated = self.model.generate(**encoded, **generation_kwargs)
        continuation = generated[0, encoded["input_ids"].shape[1] :]
        text = self.tokenizer.decode(continuation, skip_special_tokens=True).strip()
        if watermark is not None and family == "unbiased":
            watermark.utils.cc_history.clear()
        if not text:
            raise RuntimeError("Model returned an empty continuation")
        return text

    def detect(
        self, text: str, family: str, key_id: str, secret: Any | None = None
    ) -> dict[str, Any]:
        result = self.load_watermark(family, key_id, secret).detect_watermark(
            text, return_dict=True
        )
        if not isinstance(result, dict) or "score" not in result:
            raise RuntimeError("MarkLLM detector returned an unsupported result")
        return {"is_watermarked": bool(result["is_watermarked"]), "score": float(result["score"])}
