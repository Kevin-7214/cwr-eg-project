from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from cwr_eg.hashing import sha256_file
from cwr_eg.losses import LossWeights
from cwr_eg.manifest import write_jsonl
from cwr_eg.modeling import CwrEgModelConfig
from cwr_eg.tensor_bundle import build_sharded_tensor_bundle, iter_sharded_batches
from cwr_eg.training import TrainingSettings, train_from_tensor_bundle


VIEW_DIMS = {"proxy": 2, "representation": 3, "perturbation": 2, "validity": 2}


def _feature(path: Path, value: float) -> None:
    payload = {}
    for name, dimension in VIEW_DIMS.items():
        payload[f"{name}_values"] = np.full((4, dimension), value, dtype=np.float32)
        payload[f"{name}_mask"] = np.ones(4, dtype=bool)
        payload[f"{name}_offsets"] = np.asarray([[0, 1], [1, 2], [2, 3], [3, 4]])
    np.savez_compressed(path, **payload)


def _bundle(tmp_path: Path) -> Path:
    rows = []
    recipe_index = 0
    for split, language in (("train", "en"), ("dev", "zh")):
        for parent_index in range(2):
            parent_id = f"{split}-{parent_index}"
            for family in (None, "kgw"):
                path = tmp_path / f"feature-{recipe_index}.npz"
                _feature(path, float(recipe_index + 1) / 10.0)
                rows.append(
                    {
                        "recipe_id": f"recipe-{recipe_index}",
                        "parent_ids": [parent_id],
                        "split": split,
                        "language": language,
                        "watermark_family": family,
                        "intervention_id": family or "clean",
                        "boundary_quality": "exact",
                        "feature_path": str(path),
                        "feature_sha256": sha256_file(path),
                    }
                )
                recipe_index += 1
    manifest = tmp_path / "features.jsonl"
    write_jsonl(manifest, rows)
    output = tmp_path / "bundle"
    result = build_sharded_tensor_bundle(
        feature_manifest=manifest,
        output_dir=output,
        positions=4,
        maximum_batch_examples=4,
        maximum_batches_per_shard=1,
    )
    assert result["train_shards"] == result["train_batches"] == 1
    assert result["dev_shards"] == result["dev_batches"] == 1
    assert len(list(iter_sharded_batches(output, "train"))) == 1
    return output


def _model_config() -> CwrEgModelConfig:
    return CwrEgModelConfig(
        view_dims=VIEW_DIMS,
        hidden_dim=8,
        invariant_dim=4,
        private_dim=4,
        scheme_classes=4,
        nuisance_classes={"language": 2},
        dropout=0.1,
    )


def test_sharded_resume_matches_uninterrupted_cpu_training(tmp_path) -> None:
    bundle = _bundle(tmp_path)
    uninterrupted = train_from_tensor_bundle(
        bundle_path=bundle,
        output_dir=tmp_path / "uninterrupted",
        model_config=_model_config(),
        settings=TrainingSettings(
            epochs=2, minimum_epochs=2, early_stopping_patience=0, seed=17
        ),
        loss_weights=LossWeights(),
        device_name="cpu",
    )
    staged = train_from_tensor_bundle(
        bundle_path=bundle,
        output_dir=tmp_path / "staged",
        model_config=_model_config(),
        settings=TrainingSettings(
            epochs=1, minimum_epochs=1, early_stopping_patience=0, seed=17
        ),
        loss_weights=LossWeights(),
        device_name="cpu",
    )
    resumed = train_from_tensor_bundle(
        bundle_path=bundle,
        output_dir=tmp_path / "staged",
        model_config=_model_config(),
        settings=TrainingSettings(
            epochs=2, minimum_epochs=1, early_stopping_patience=0, seed=17
        ),
        loss_weights=LossWeights(),
        device_name="cpu",
        resume_checkpoint=staged["latest_checkpoint"],
    )
    uninterrupted_state = torch.load(
        uninterrupted["latest_checkpoint"], map_location="cpu", weights_only=True
    )["state_dict"]
    resumed_state = torch.load(
        resumed["latest_checkpoint"], map_location="cpu", weights_only=True
    )["state_dict"]
    assert uninterrupted_state.keys() == resumed_state.keys()
    assert all(
        torch.equal(uninterrupted_state[name], resumed_state[name])
        for name in uninterrupted_state
    )


def test_sharded_bundle_serializes_base_attack_consistency_pairs(tmp_path) -> None:
    rows = []
    for index, (recipe_id, base_recipe_id) in enumerate(
        (("base-clean", None), ("attack-base-clean", "base-clean"))
    ):
        feature_path = tmp_path / f"pair-{index}.npz"
        _feature(feature_path, float(index + 1))
        rows.append(
            {
                "recipe_id": recipe_id,
                "base_recipe_id": base_recipe_id,
                "parent_ids": ["parent-1"],
                "split": "train",
                "language": "en",
                "watermark_family": None,
                "intervention_id": "clean" if base_recipe_id is None else "copy_edit",
                "boundary_quality": "exact",
                "feature_path": str(feature_path),
                "feature_sha256": sha256_file(feature_path),
            }
        )
    for index, (recipe_id, base_recipe_id) in enumerate(
        (("dev-base-clean", None), ("dev-attack-base-clean", "dev-base-clean")), start=2
    ):
        feature_path = tmp_path / f"pair-{index}.npz"
        _feature(feature_path, float(index + 1))
        rows.append(
            {
                "recipe_id": recipe_id,
                "base_recipe_id": base_recipe_id,
                "parent_ids": ["parent-2"],
                "split": "dev",
                "language": "zh",
                "watermark_family": None,
                "intervention_id": "clean" if base_recipe_id is None else "copy_edit",
                "boundary_quality": "exact",
                "feature_path": str(feature_path),
                "feature_sha256": sha256_file(feature_path),
            }
        )
    manifest = tmp_path / "paired-features.jsonl"
    write_jsonl(manifest, rows)
    output = tmp_path / "paired-bundle"
    result = build_sharded_tensor_bundle(
        feature_manifest=manifest,
        output_dir=output,
        positions=4,
        maximum_batch_examples=4,
        maximum_batches_per_shard=1,
    )
    train_batch = next(iter_sharded_batches(output, "train"))
    assert result["train_consistency_pairs"] == 1
    assert result["dev_consistency_pairs"] == 1
    assert train_batch["consistency_pairs"].tolist() == [[0, 1]]
    assert train_batch["consistency_preserves_label"].tolist() == [True]
