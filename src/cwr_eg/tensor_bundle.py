from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cwr_eg.counterfactual import build_pairing_masks
from cwr_eg.hashing import sha256_file
from cwr_eg.manifest import read_jsonl


SCHEME_LABELS = {"kgw": 0, "unigram": 1, "unbiased": 2, "synthid": 3}
LANGUAGE_LABELS = {"en": 0, "zh": 1}


def _pad(array: np.ndarray, positions: int) -> np.ndarray:
    result = np.zeros((positions, array.shape[1]), dtype=np.float32)
    used = min(positions, array.shape[0])
    result[:used] = array[:used]
    return result


def _load_example(row: dict[str, Any], positions: int) -> dict[str, Any]:
    if len(row["parent_ids"]) != 1:
        raise ValueError("Mixed-parent features are evaluation-only and cannot enter paired training")
    if sha256_file(row["feature_path"]) != str(row["feature_sha256"]):
        raise RuntimeError("Feature file SHA-256 does not match its manifest")
    with np.load(row["feature_path"], allow_pickle=False) as payload:
        view_names = tuple(
            key[: -len("_values")] for key in payload.files if key.endswith("_values")
        )
        views = {
            name: _pad(np.asarray(payload[f"{name}_values"], dtype=np.float32), positions)
            for name in view_names
        }
        if any(not np.all(np.isfinite(values)) for values in views.values()):
            raise RuntimeError("Feature file contains non-finite values")
        masks = [np.asarray(payload[f"{name}_mask"], dtype=bool)[:positions] for name in view_names]
    valid = np.zeros(positions, dtype=bool)
    if masks:
        used = min(positions, len(masks[0]))
        valid[:used] = np.logical_and.reduce([mask[:used] for mask in masks])
    family = row.get("watermark_family")
    return {
        "split": row["split"],
        "parent_id": row["parent_ids"][0],
        "intervention_id": row.get("intervention_id", family or "clean"),
        "views": views,
        "valid_mask": valid,
        "watermark_label": float(family is not None),
        "scheme_label": SCHEME_LABELS.get(family, 0),
        "language_label": LANGUAGE_LABELS[row["language"]],
        "boundary_quality_weight": 0.5 if row.get("boundary_quality") == "weak" else 1.0,
    }


def _pack_batch(examples: list[dict[str, Any]]) -> dict[str, Any]:
    view_names = tuple(examples[0]["views"])
    if any(tuple(example["views"]) != view_names for example in examples):
        raise ValueError("All examples in a batch must provide the same ordered views")
    pairing = build_pairing_masks(
        [example["parent_id"] for example in examples],
        [example["intervention_id"] for example in examples],
        [example["boundary_quality_weight"] for example in examples],
    )
    valid_mask = torch.tensor(
        np.stack([example["valid_mask"] for example in examples]).tolist(), dtype=torch.bool
    )
    watermark = torch.tensor(
        [example["watermark_label"] for example in examples], dtype=torch.float32
    )
    boundary_targets = watermark[:, None].expand_as(valid_mask).to(torch.float32)
    return {
        "split": examples[0]["split"],
        "views": {
            name: torch.tensor(
                np.stack([example["views"][name] for example in examples]).tolist(),
                dtype=torch.float32,
            )
            for name in view_names
        },
        "valid_mask": valid_mask,
        "watermark_labels": watermark,
        "scheme_labels": torch.tensor(
            [example["scheme_label"] for example in examples], dtype=torch.long
        ),
        "positive_mask": torch.tensor(pairing.positive_mask.tolist(), dtype=torch.bool),
        "negative_mask": torch.tensor(pairing.negative_mask.tolist(), dtype=torch.bool),
        "pair_weights": torch.tensor(pairing.pair_weights.tolist(), dtype=torch.float32),
        "nuisance_language_labels": torch.tensor(
            [example["language_label"] for example in examples], dtype=torch.long
        ),
        "boundary_targets": boundary_targets,
        "boundary_mask": valid_mask,
        "boundary_quality_weights": torch.tensor(
            [example["boundary_quality_weight"] for example in examples],
            dtype=torch.float32,
        ),
    }


def build_tensor_bundle(
    *,
    feature_manifest: str | Path,
    output_path: str | Path,
    positions: int = 256,
    maximum_batch_examples: int = 20,
) -> dict[str, Any]:
    rows = read_jsonl(feature_manifest)
    examples = [_load_example(row, positions) for row in rows]
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for example in examples:
        if example["split"] not in {"train", "dev"}:
            continue
        grouped[example["split"]][example["parent_id"]].append(example)
    result: dict[str, list[dict[str, Any]]] = {"train_batches": [], "dev_batches": []}
    for split in ("train", "dev"):
        pending: list[dict[str, Any]] = []
        for parent_id in sorted(grouped[split]):
            parent_examples = grouped[split][parent_id]
            if len(parent_examples) > maximum_batch_examples:
                raise ValueError("A parent intervention group exceeds the maximum batch size")
            if pending and len(pending) + len(parent_examples) > maximum_batch_examples:
                result[f"{split}_batches"].append(_pack_batch(pending))
                pending = []
            pending.extend(parent_examples)
        if pending:
            result[f"{split}_batches"].append(_pack_batch(pending))
    if not result["train_batches"] or not result["dev_batches"]:
        raise ValueError("Train and Dev feature groups are both required")
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, target)
    return {
        "output_path": str(target),
        "sha256": sha256_file(target),
        "train_batches": len(result["train_batches"]),
        "dev_batches": len(result["dev_batches"]),
        "positions": positions,
    }
