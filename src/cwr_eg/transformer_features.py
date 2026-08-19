from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

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
    trust_remote_code: bool = False


def _prepend_eos_for_single_token_rows(
    encoded: Any,
    offsets_by_row: list[list[list[int]]],
    *,
    eos_token_id: int | None,
) -> set[int]:
    lengths = [int(item) for item in encoded["attention_mask"].sum(dim=1).tolist()]
    if any(length < 1 for length in lengths):
        raise ValueError("At least one token is required for causal features")
    single_token_rows = {index for index, length in enumerate(lengths) if length == 1}
    if not single_token_rows:
        return set()
    if eos_token_id is None:
        raise ValueError("An EOS token is required for single-token causal features")
    if int(encoded["input_ids"].shape[1]) < 2:
        batch_size = int(encoded["input_ids"].shape[0])
        for name, value in list(encoded.items()):
            if getattr(value, "ndim", 0) != 2 or int(value.shape[1]) != 1:
                continue
            expanded = value.new_zeros((batch_size, 2))
            expanded[:, :1] = value
            encoded[name] = expanded
        for offsets in offsets_by_row:
            offsets.append([0, 0])
    for row_index in single_token_rows:
        original_token = encoded["input_ids"][row_index, 0].clone()
        original_offset = list(offsets_by_row[row_index][0])
        encoded["input_ids"][row_index, 0] = eos_token_id
        encoded["input_ids"][row_index, 1] = original_token
        encoded["attention_mask"][row_index, :2] = 1
        offsets_by_row[row_index][0] = [0, 0]
        offsets_by_row[row_index][1] = original_offset
    return single_token_rows


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
            trust_remote_code=settings.trust_remote_code,
        )
        if not self.tokenizer.is_fast:
            raise RuntimeError("A fast tokenizer with offset mapping is required")
        self.model = AutoModelForCausalLM.from_pretrained(
            settings.model_path,
            revision=settings.revision,
            local_files_only=settings.local_files_only,
            torch_dtype=dtype,
            trust_remote_code=settings.trust_remote_code,
            use_safetensors=True,
        ).to(settings.device)
        self.model.eval()

    def _build_extracted(
        self,
        *,
        document_id: str,
        language: str,
        offsets: list[list[int]],
        proxy: np.ndarray,
        representation: np.ndarray,
        extractor_version: str,
    ) -> ExtractedViews:
        proxy_np = np.asarray(proxy, dtype=np.float32)
        representation_np = np.asarray(representation, dtype=np.float32)
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
        language_code = np.full(
            len(intervals), 1.0 if language == "zh" else 0.0, dtype=np.float32
        )
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
            extractor_version=extractor_version,
        )

    def extract_many(
        self, documents: Iterable[tuple[str, str, str]]
    ) -> list[ExtractedViews]:
        torch = self.torch
        rows = list(documents)
        if not rows:
            return []
        texts = [raw_text for _, raw_text, _ in rows]
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        encoded = self.tokenizer(
            texts,
            return_tensors="pt",
            return_offsets_mapping=True,
            add_special_tokens=False,
            truncation=True,
            max_length=self.settings.maximum_tokens,
            padding=True,
        )
        offsets_by_row = encoded.pop("offset_mapping").tolist()
        single_token_rows = _prepend_eos_for_single_token_rows(
            encoded,
            offsets_by_row,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        inputs = {name: value.to(self.settings.device) for name, value in encoded.items()}
        lengths = inputs["attention_mask"].sum(dim=1).tolist()
        with torch.inference_mode():
            outputs = self.model(**inputs, output_hidden_states=True)
        extracted: list[ExtractedViews] = []
        for row_index, ((document_id, _, language), raw_length) in enumerate(
            zip(rows, lengths, strict=True)
        ):
            token_count = int(raw_length)
            token_logits = outputs.logits[row_index, : token_count - 1]
            targets = inputs["input_ids"][row_index, 1:token_count]
            proxy_parts = []
            for start in range(0, token_count - 1, 32):
                stop = min(start + 32, token_count - 1)
                logits = token_logits[start:stop].float()
                log_normalizer = torch.logsumexp(logits, dim=-1)
                target_logits = logits.gather(
                    -1, targets[start:stop].unsqueeze(-1)
                ).squeeze(-1)
                nll = log_normalizer - target_logits
                probabilities = torch.softmax(logits, dim=-1)
                entropy = (
                    probabilities * (log_normalizer.unsqueeze(-1) - logits)
                ).sum(-1)
                top_logits = torch.topk(logits, k=2, dim=-1).values
                top_probability = torch.exp(top_logits - log_normalizer.unsqueeze(-1))
                log_rank = torch.log1p(
                    (logits > target_logits.unsqueeze(-1)).sum(dim=-1).float()
                )
                proxy_parts.append(
                    torch.stack(
                        (
                            nll,
                            entropy,
                            log_rank,
                            top_probability[..., 0] - top_probability[..., 1],
                        ),
                        dim=-1,
                    ).cpu()
                )
            proxy = torch.cat(proxy_parts, dim=0).numpy()
            representation = (
                outputs.hidden_states[-1][row_index, 1:token_count]
                .float()
                .cpu()
                .numpy()
            )
            extracted.append(
                self._build_extracted(
                    document_id=document_id,
                    language=language,
                    offsets=offsets_by_row[row_index][:token_count],
                    proxy=proxy,
                    representation=representation,
                    extractor_version=(
                        "transformers-causal-batched-logrank-v3-short-eos-fallback-v1"
                        if row_index in single_token_rows
                        else "transformers-causal-batched-logrank-v3"
                    ),
                )
            )
        return extracted

    def extract(self, document_id: str, raw_text: str, language: str) -> ExtractedViews:
        return self.extract_many([(document_id, raw_text, language)])[0]
