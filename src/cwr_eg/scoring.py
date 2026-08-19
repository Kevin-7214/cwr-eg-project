from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import torch

from cwr_eg.hashing import sha256_file
from cwr_eg.manifest import read_jsonl, write_jsonl
from cwr_eg.modeling import CwrEgModel, CwrEgModelConfig


SCHEME_NAMES = ("kgw", "unigram", "unbiased", "synthid")


def nearest_valid_fill(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    scores = np.asarray(values, dtype=np.float64)
    mask = np.asarray(valid, dtype=bool)
    if scores.ndim != 1 or mask.shape != scores.shape:
        raise ValueError("Character values and valid mask must be aligned vectors")
    if not np.any(mask):
        raise ValueError("At least one character must have a valid token score")
    result = scores.copy()
    valid_indices = np.flatnonzero(mask)
    missing_indices = np.flatnonzero(~mask)
    insertion = np.searchsorted(valid_indices, missing_indices)
    right_positions = np.minimum(insertion, len(valid_indices) - 1)
    left_positions = np.maximum(insertion - 1, 0)
    left = valid_indices[left_positions]
    right = valid_indices[right_positions]
    use_right = np.abs(right - missing_indices) < np.abs(missing_indices - left)
    nearest = np.where(use_right, right, left)
    result[missing_indices] = result[nearest]
    return result


def map_token_logits_to_characters(
    token_logits: np.ndarray,
    token_offsets: np.ndarray,
    token_valid: np.ndarray,
    raw_length: int,
) -> tuple[np.ndarray, float, np.ndarray]:
    values = np.asarray(token_logits, dtype=np.float64)
    offsets = np.asarray(token_offsets, dtype=np.int64)
    valid_tokens = np.asarray(token_valid, dtype=bool)
    if values.ndim != 1 or offsets.shape != (len(values), 2):
        raise ValueError("Token logits and offsets are not aligned")
    if valid_tokens.shape != values.shape or raw_length < 1:
        raise ValueError("Invalid token mask or raw text length")
    sums = np.zeros(raw_length, dtype=np.float64)
    counts = np.zeros(raw_length, dtype=np.int64)
    for value, (start, end), is_valid in zip(
        values, offsets, valid_tokens, strict=True
    ):
        if not is_valid or start < 0 or end <= start:
            continue
        if end > raw_length:
            raise ValueError("Tokenizer offset exceeds raw Unicode code-point length")
        sums[start:end] += value
        counts[start:end] += 1
    covered = counts > 0
    coverage = float(np.mean(covered))
    sums[covered] /= counts[covered]
    return nearest_valid_fill(sums, covered), coverage, covered


def _window_starts(length: int, positions: int, stride: int) -> list[int]:
    if length < 1 or positions < 1 or stride < 1:
        raise ValueError("Window arguments must be positive")
    if length <= positions:
        return [0]
    starts = list(range(0, length - positions + 1, stride))
    final = length - positions
    if starts[-1] != final:
        starts.append(final)
    return starts


@dataclass(frozen=True, slots=True)
class FeatureDocument:
    views: dict[str, np.ndarray]
    valid_mask: np.ndarray
    offsets: np.ndarray


def load_feature_document(
    feature_path: str | Path,
    *,
    expected_sha256: str,
    view_names: Iterable[str],
) -> FeatureDocument:
    path = Path(feature_path)
    if sha256_file(path) != expected_sha256:
        raise RuntimeError("Feature SHA-256 does not match its manifest")
    names = tuple(view_names)
    with np.load(path, allow_pickle=False) as payload:
        views = {
            name: np.asarray(payload[f"{name}_values"], dtype=np.float32)
            for name in names
        }
        masks = [
            np.asarray(payload[f"{name}_mask"], dtype=bool) for name in names
        ]
        offsets = np.asarray(payload[f"{names[0]}_offsets"], dtype=np.int64)
        for name in names[1:]:
            if not np.array_equal(offsets, payload[f"{name}_offsets"]):
                raise RuntimeError("Feature views use inconsistent tokenizer offsets")
    lengths = {len(value) for value in views.values()} | {len(mask) for mask in masks}
    if lengths != {len(offsets)}:
        raise RuntimeError("Feature views, masks, and offsets are not aligned")
    if any(not np.all(np.isfinite(value)) for value in views.values()):
        raise RuntimeError("Feature document contains non-finite values")
    valid_mask = np.logical_and.reduce(masks)
    return FeatureDocument(views=views, valid_mask=valid_mask, offsets=offsets)


@torch.inference_mode()
def score_feature_document(
    models: list[CwrEgModel],
    feature: FeatureDocument,
    *,
    raw_length: int,
    positions: int,
    device: torch.device,
    stride: int | None = None,
    null_prototypes: list[torch.Tensor] | None = None,
) -> dict[str, Any]:
    if not models:
        raise ValueError("At least one checkpoint model is required")
    token_count = len(feature.offsets)
    window_stride = stride or max(1, positions // 2)
    starts = _window_starts(token_count, positions, window_stride)
    window_views: dict[str, torch.Tensor] = {}
    for name, values in feature.views.items():
        padded = np.zeros((len(starts), positions, values.shape[1]), dtype=np.float32)
        for window_index, start in enumerate(starts):
            stop = min(start + positions, token_count)
            padded[window_index, : stop - start] = values[start:stop]
        window_views[name] = torch.from_numpy(padded).to(device)
    padded_mask = np.zeros((len(starts), positions), dtype=bool)
    for window_index, start in enumerate(starts):
        stop = min(start + positions, token_count)
        padded_mask[window_index, : stop - start] = feature.valid_mask[start:stop]
    valid_mask = torch.from_numpy(padded_mask).to(device)

    token_sum = np.zeros(token_count, dtype=np.float64)
    token_count_by_position = np.zeros(token_count, dtype=np.int64)
    if null_prototypes is not None and len(null_prototypes) != len(models):
        raise ValueError("Each checkpoint model requires one null prototype")
    residual_by_model: list[np.ndarray] = []
    watermark_by_model: list[np.ndarray] = []
    private_scheme_values: list[np.ndarray] = []
    adversarial_scheme_values: list[np.ndarray] = []
    invariant_embeddings: list[np.ndarray] = []
    private_embeddings: list[np.ndarray] = []
    for model_index, model in enumerate(models):
        model.eval()
        outputs = model(window_views, valid_mask, grl_scale=0.0)
        character = outputs["character_logits"].float().cpu().numpy()
        for window_index, start in enumerate(starts):
            stop = min(start + positions, token_count)
            used = stop - start
            window_valid = padded_mask[window_index, :used]
            positions_index = np.flatnonzero(window_valid) + start
            token_sum[positions_index] += character[window_index, :used][window_valid]
            token_count_by_position[positions_index] += 1
        if null_prototypes is None:
            residual = outputs["residual_score"].float()
            residual_source = "legacy-untrained-head"
        else:
            prototype = null_prototypes[model_index].to(
                device=device, dtype=outputs["z_inv"].dtype
            )
            residual = torch.linalg.vector_norm(
                outputs["z_inv"] - prototype.unsqueeze(0), dim=-1
            ).float()
            residual_source = "train-null-prototype-euclidean-v1"
        residual_by_model.append(residual.cpu().numpy())
        watermark_by_model.append(outputs["watermark_logits"].float().cpu().numpy())
        private_scheme_values.append(
            outputs["private_scheme_logits"].float().cpu().numpy()
        )
        adversarial_scheme_values.append(
            outputs["scheme_adv_logits"].float().cpu().numpy()
        )
        invariant_embeddings.append(outputs["z_inv"].float().cpu().numpy())
        private_embeddings.append(outputs["z_priv"].float().cpu().numpy())
    token_valid = token_count_by_position > 0
    token_logits = np.zeros(token_count, dtype=np.float64)
    token_logits[token_valid] = token_sum[token_valid] / token_count_by_position[token_valid]
    character_logits, mapping_coverage, covered = map_token_logits_to_characters(
        token_logits, feature.offsets, token_valid, raw_length
    )
    private_logits = np.mean(np.concatenate(private_scheme_values, axis=0), axis=0)
    adversarial_logits = np.mean(
        np.concatenate(adversarial_scheme_values, axis=0), axis=0
    )
    invariant_embedding = np.mean(
        np.concatenate(invariant_embeddings, axis=0), axis=0
    )
    private_embedding = np.mean(np.concatenate(private_embeddings, axis=0), axis=0)
    proxy_values = feature.views["proxy"][feature.valid_mask]
    perturbation_values = feature.views["perturbation"][feature.valid_mask]
    if proxy_values.shape[1] < 3 or perturbation_values.shape[1] < 1:
        raise RuntimeError("Direct-statistics views have incompatible dimensions")
    direct_statistics = np.asarray(
        [
            proxy_values[:, 0].mean(),
            proxy_values[:, 1].mean(),
            proxy_values[:, 2].mean(),
            perturbation_values[:, 0].mean(),
        ],
        dtype=np.float64,
    )
    residual_values = np.mean(np.stack(residual_by_model, axis=0), axis=0)
    watermark_values = np.mean(np.stack(watermark_by_model, axis=0), axis=0)
    shifted = private_logits - np.max(private_logits)
    probabilities = np.exp(shifted) / np.exp(shifted).sum()
    if not (
        np.all(np.isfinite(character_logits))
        and np.all(np.isfinite(private_logits))
        and np.all(np.isfinite(adversarial_logits))
        and np.all(np.isfinite(invariant_embedding))
        and np.all(np.isfinite(private_embedding))
        and np.all(np.isfinite(direct_statistics))
        and np.all(np.isfinite(residual_values))
        and np.all(np.isfinite(watermark_values))
    ):
        raise RuntimeError("Checkpoint scoring produced a non-finite value")
    return {
        "character_logits": character_logits.tolist(),
        "generic_residual_score": float(np.max(residual_values)),
        "generic_residual_source": residual_source,
        "watermark_logit": float(np.max(watermark_values)),
        "mechanism_logits": {
            name: float(value) for name, value in zip(SCHEME_NAMES, private_logits, strict=True)
        },
        "mechanism_probabilities": {
            name: float(value) for name, value in zip(SCHEME_NAMES, probabilities, strict=True)
        },
        "invariant_scheme_adversary_logits": {
            name: float(value)
            for name, value in zip(SCHEME_NAMES, adversarial_logits, strict=True)
        },
        "invariant_embedding": invariant_embedding.tolist(),
        "private_embedding": private_embedding.tolist(),
        "direct_statistical_features": direct_statistics.tolist(),
        "effective_length": int(np.count_nonzero(covered)),
        "mapping_coverage": mapping_coverage,
        "window_count": len(starts),
        "checkpoint_count": len(models),
    }


def score_checkpoint_features(
    *,
    checkpoint_paths: Iterable[str | Path],
    feature_manifest_path: str | Path,
    documents_path: str | Path,
    output_path: str | Path,
    positions: int,
    device_name: str,
    minimum_mapping_coverage: float = 0.98,
    recipe_ids: Iterable[str] | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    checkpoints = [Path(path) for path in checkpoint_paths]
    if not checkpoints:
        raise ValueError("checkpoint_paths cannot be empty")
    checkpoint_payloads = [
        torch.load(path, map_location="cpu", weights_only=True) for path in checkpoints
    ]
    model_configs = [payload["model_config"] for payload in checkpoint_payloads]
    if any(config != model_configs[0] for config in model_configs[1:]):
        raise RuntimeError("Ensemble checkpoints use different model configurations")
    model_config = CwrEgModelConfig(**model_configs[0])
    device = torch.device(device_name)
    models: list[CwrEgModel] = []
    null_prototypes: list[torch.Tensor] = []
    for payload in checkpoint_payloads:
        model = CwrEgModel(model_config).to(device)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        models.append(model)
        prototype = torch.as_tensor(payload["null_prototype"], dtype=torch.float32)
        if prototype.shape != (model_config.invariant_dim,):
            raise RuntimeError("Checkpoint null prototype has an incompatible shape")
        null_prototypes.append(prototype)

    documents = {str(row["recipe_id"]): row for row in read_jsonl(documents_path)}
    features = read_jsonl(feature_manifest_path)
    allowed = set(recipe_ids or ())
    if allowed:
        features = [row for row in features if str(row["recipe_id"]) in allowed]
    if len({str(row["recipe_id"]) for row in features}) != len(features):
        raise ValueError("Feature manifest contains duplicate recipe ids")
    target = Path(output_path)
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite checkpoint scores: {target}")
    partial = target.with_suffix(target.suffix + ".partial")
    outputs = read_jsonl(partial) if partial.exists() else []
    completed = {str(row["recipe_id"]) for row in outputs}
    selected = {str(row["recipe_id"]) for row in features}
    if not completed.issubset(selected):
        raise RuntimeError("Partial score output contains an unapproved recipe id")
    if progress_callback is not None:
        progress_callback(len(outputs), len(features))
    for feature_row in features:
        recipe_id = str(feature_row["recipe_id"])
        if recipe_id in completed:
            continue
        if recipe_id not in documents:
            raise KeyError(f"Missing raw document for feature: {recipe_id}")
        document = documents[recipe_id]
        raw_text = str(document["text"])
        feature = load_feature_document(
            feature_row["feature_path"],
            expected_sha256=str(feature_row["feature_sha256"]),
            view_names=model_config.view_dims,
        )
        scores = score_feature_document(
            models,
            feature,
            raw_length=len(raw_text),
            positions=positions,
            device=device,
            null_prototypes=null_prototypes,
        )
        outputs.append(
            {
                "document_id": recipe_id,
                "recipe_id": recipe_id,
                "parent_ids": document["parent_ids"],
                "split": document["split"],
                "source": document.get("source"),
                "language": document["language"],
                "watermark_family": document.get("watermark_family"),
                "key_id": document.get("key_id"),
                "attack_id": document.get("attack_id"),
                "text": raw_text,
                **scores,
                "validity_override": "uncertain"
                if float(scores["mapping_coverage"]) < minimum_mapping_coverage
                else None,
                "mapping_policy": "unicode-codepoint-nearest-valid-fill-v1",
                "score_version": "checkpoint-sliding-window-v1",
            }
        )
        write_jsonl(partial, outputs)
        if progress_callback is not None:
            progress_callback(len(outputs), len(features))
    target.parent.mkdir(parents=True, exist_ok=True)
    partial.replace(target)
    return {
        "documents": len(outputs),
        "output_path": str(target),
        "output_sha256": sha256_file(target),
        "checkpoint_sha256": [sha256_file(path) for path in checkpoints],
        "ensemble_rule": "arithmetic_mean_character_logits",
        "positions": positions,
    }
