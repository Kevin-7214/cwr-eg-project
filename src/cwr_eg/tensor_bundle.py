from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch

from cwr_eg.counterfactual import build_pairing_masks
from cwr_eg.hashing import content_hash, sha256_file
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
        "recipe_id": row["recipe_id"],
        "base_recipe_id": row.get("base_recipe_id"),
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
    recipe_indices = {
        str(example["recipe_id"]): index for index, example in enumerate(examples)
    }
    consistency_pairs = [
        [recipe_indices[str(example["base_recipe_id"])], index]
        for index, example in enumerate(examples)
        if example.get("base_recipe_id") in recipe_indices
    ]
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
        "consistency_pairs": torch.tensor(consistency_pairs, dtype=torch.long).reshape(-1, 2),
        "consistency_preserves_label": torch.ones(
            len(consistency_pairs), dtype=torch.bool
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


def _atomic_torch_save(payload: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def build_sharded_tensor_bundle(
    *,
    feature_manifest: str | Path,
    output_dir: str | Path,
    positions: int = 256,
    maximum_batch_examples: int = 20,
    maximum_batches_per_shard: int = 16,
    excluded_watermark_families: tuple[str, ...] = (),
) -> dict[str, Any]:
    if positions < 1 or maximum_batch_examples < 1 or maximum_batches_per_shard < 1:
        raise ValueError("Tensor bundle limits must be positive")
    rows = read_jsonl(feature_manifest)
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    mixed_parent_features_excluded = 0
    excluded = set(excluded_watermark_families)
    if not excluded.issubset(SCHEME_LABELS):
        raise ValueError("Tensor bundle excludes an unsupported watermark family")
    for row in rows:
        if row.get("watermark_family") in excluded:
            continue
        if len(row["parent_ids"]) != 1:
            mixed_parent_features_excluded += 1
            continue
        split = str(row["split"])
        if split in {"train", "dev"}:
            grouped[split][str(row["parent_ids"][0])].append(row)

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=False)
    shard_entries: dict[str, list[dict[str, Any]]] = {"train": [], "dev": []}
    total_batches: dict[str, int] = {"train": 0, "dev": 0}
    total_examples: dict[str, int] = {"train": 0, "dev": 0}
    total_consistency_pairs: dict[str, int] = {"train": 0, "dev": 0}

    for split in ("train", "dev"):
        shard_batches: list[dict[str, Any]] = []
        pending_examples: list[dict[str, Any]] = []

        def flush_batch() -> None:
            nonlocal pending_examples, shard_batches
            if not pending_examples:
                return
            packed = _pack_batch(pending_examples)
            shard_batches.append(packed)
            total_batches[split] += 1
            total_examples[split] += len(pending_examples)
            total_consistency_pairs[split] += int(packed["consistency_pairs"].shape[0])
            pending_examples = []
            if len(shard_batches) == maximum_batches_per_shard:
                flush_shard()

        def flush_shard() -> None:
            nonlocal shard_batches
            if not shard_batches:
                return
            shard_index = len(shard_entries[split])
            path = target / f"{split}-{shard_index:04d}.pt"
            examples = sum(int(batch["watermark_labels"].shape[0]) for batch in shard_batches)
            _atomic_torch_save(
                {"format": "sharded-v1", "split": split, "batches": shard_batches},
                path,
            )
            shard_entries[split].append(
                {
                    "path": path.name,
                    "sha256": sha256_file(path),
                    "batches": len(shard_batches),
                    "examples": examples,
                }
            )
            shard_batches = []

        for parent_id in sorted(grouped[split]):
            parent_rows = grouped[split][parent_id]
            parent_examples = [_load_example(row, positions) for row in parent_rows]
            if len(parent_examples) > maximum_batch_examples:
                raise ValueError("A parent intervention group exceeds the maximum batch size")
            if pending_examples and (
                len(pending_examples) + len(parent_examples) > maximum_batch_examples
            ):
                flush_batch()
            pending_examples.extend(parent_examples)
        flush_batch()
        flush_shard()

    if not shard_entries["train"] or not shard_entries["dev"]:
        raise ValueError("Train and Dev feature groups are both required")
    index: dict[str, Any] = {
        "format": "sharded-v1",
        "positions": positions,
        "maximum_batch_examples": maximum_batch_examples,
        "maximum_batches_per_shard": maximum_batches_per_shard,
        "excluded_watermark_families": sorted(excluded),
        "mixed_parent_features_excluded": mixed_parent_features_excluded,
        "source_feature_manifest": str(feature_manifest),
        "source_feature_manifest_sha256": sha256_file(feature_manifest),
        "splits": {
            split: {
                "batches": total_batches[split],
                "examples": total_examples[split],
                "consistency_pairs": total_consistency_pairs[split],
                "shards": shard_entries[split],
            }
            for split in ("train", "dev")
        },
    }
    index["bundle_content_hash"] = content_hash(index)
    index_path = target / "bundle_index.json"
    temporary = index_path.with_suffix(index_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(index, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(index_path)
    return {
        "format": "sharded-v1",
        "output_path": str(target),
        "index_path": str(index_path),
        "index_sha256": sha256_file(index_path),
        "bundle_content_hash": index["bundle_content_hash"],
        "train_batches": total_batches["train"],
        "dev_batches": total_batches["dev"],
        "train_shards": len(shard_entries["train"]),
        "dev_shards": len(shard_entries["dev"]),
        "train_consistency_pairs": total_consistency_pairs["train"],
        "dev_consistency_pairs": total_consistency_pairs["dev"],
        "positions": positions,
        "mixed_parent_features_excluded": mixed_parent_features_excluded,
    }


def load_sharded_bundle_index(path: str | Path) -> tuple[Path, dict[str, Any]]:
    source = Path(path)
    index_path = source / "bundle_index.json" if source.is_dir() else source
    with index_path.open("r", encoding="utf-8") as handle:
        index = json.load(handle)
    if index.get("format") != "sharded-v1":
        raise ValueError("Unsupported tensor bundle format")
    expected_hash = str(index["bundle_content_hash"])
    without_hash = dict(index)
    without_hash.pop("bundle_content_hash", None)
    if content_hash(without_hash) != expected_hash:
        raise RuntimeError("Tensor bundle index content hash mismatch")
    return index_path, index


def iter_sharded_batches(path: str | Path, split: str) -> Iterator[dict[str, Any]]:
    index_path, index = load_sharded_bundle_index(path)
    if split not in {"train", "dev"}:
        raise ValueError("Only Train and Dev shards are loadable")
    for entry in index["splits"][split]["shards"]:
        shard_path = index_path.parent / str(entry["path"])
        if sha256_file(shard_path) != str(entry["sha256"]):
            raise RuntimeError(f"Tensor shard checksum mismatch: {shard_path.name}")
        payload = torch.load(shard_path, map_location="cpu", weights_only=True)
        if payload.get("format") != "sharded-v1" or payload.get("split") != split:
            raise RuntimeError(f"Tensor shard metadata mismatch: {shard_path.name}")
        batches = list(payload["batches"])
        if len(batches) != int(entry["batches"]):
            raise RuntimeError(f"Tensor shard batch count mismatch: {shard_path.name}")
        yield from batches
