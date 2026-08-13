from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cwr_eg.contracts import CharacterInterval
from cwr_eg.features import ExtractedViews, FeatureView


@dataclass(frozen=True, slots=True)
class TransformerFeatureSettings:
    model_path: Path
    revision: str
    device: str
    maximum_tokens: int = 1024
    local_files_only: bool = True


class TransformersCausalFeatureExtractor:
    def __init__(self, settings: TransformerFeatureSettings) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.settings = settings
        self.torch = torch
        dtype = torch.bfloat16 if settings.device.startswith("cuda") else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(
            settings.model_path,
            revision=settings.revision,
            local_files_only=settings.local_files_only,
            use_fast=True,
        )
        if not self.tokenizer.is_fast:
            raise RuntimeError("A fast tokenizer with offset mapping is required")
        self.model = AutoModelForCausalLM.from_pretrained(
            settings.model_path,
            revision=settings.revision,
            local_files_only=settings.local_files_only,
            torch_dtype=dtype,
        ).to(settings.device)
        self.model.eval()

    def extract(self, document_id: str, raw_text: str, language: str) -> ExtractedViews:
        torch = self.torch
        encoded = self.tokenizer(
            raw_text,
            return_tensors="pt",
            return_offsets_mapping=True,
            add_special_tokens=False,
            truncation=True,
            max_length=self.settings.maximum_tokens,
        )
        offsets = encoded.pop("offset_mapping")[0].tolist()
        inputs = {name: value.to(self.settings.device) for name, value in encoded.items()}
        if inputs["input_ids"].shape[1] < 2:
            raise ValueError("At least two tokens are required for causal features")
        with torch.inference_mode():
            outputs = self.model(**inputs, output_hidden_states=True)
            logits = outputs.logits[:, :-1].float()
            targets = inputs["input_ids"][:, 1:]
            probabilities = torch.softmax(logits, dim=-1)
            target_probability = probabilities.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            nll = -torch.log(target_probability.clamp_min(1e-12))
            entropy = -(probabilities * torch.log(probabilities.clamp_min(1e-12))).sum(-1)
            top_two = torch.topk(probabilities, k=2, dim=-1).values
            proxy = torch.stack(
                (nll, entropy, top_two[..., 0], top_two[..., 0] - top_two[..., 1]),
                dim=-1,
            )[0]
            representation = outputs.hidden_states[-1][0, 1:].float()
        proxy_np = proxy.cpu().numpy().astype(np.float32)
        representation_np = representation.cpu().numpy().astype(np.float32)
        perturbation = np.zeros_like(proxy_np)
        perturbation[1:] = np.abs(proxy_np[1:] - proxy_np[:-1])
        used_offsets = offsets[1:]
        intervals = tuple(
            CharacterInterval(int(start), int(end)) if end > start else None
            for start, end in used_offsets
        )
        valid = np.asarray([item is not None for item in intervals], dtype=bool)
        lengths = np.asarray(
            [0 if item is None else item.length for item in intervals], dtype=np.float32
        )
        positions = np.linspace(0.0, 1.0, num=len(intervals), dtype=np.float32)
        language_code = np.full(len(intervals), 1.0 if language == "zh" else 0.0, dtype=np.float32)
        validity = np.stack(
            (lengths, positions, language_code, valid.astype(np.float32)), axis=-1
        )

        def make(name: str, values: np.ndarray) -> FeatureView:
            return FeatureView(
                name=name,
                values=values,
                raw_intervals=intervals,
                valid_mask=valid,
                metadata={
                    "model_path": str(self.settings.model_path),
                    "revision": self.settings.revision,
                    "language": language,
                },
            )

        return ExtractedViews(
            document_id=document_id,
            views={
                "proxy": make("proxy", proxy_np),
                "representation": make("representation", representation_np),
                "perturbation": make("perturbation", perturbation),
                "validity": make("validity", validity),
            },
            normalization_version="raw-unicode-codepoint-v1",
            extractor_version="transformers-causal-v1",
        )
