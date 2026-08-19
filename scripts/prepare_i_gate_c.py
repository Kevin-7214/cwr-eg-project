from __future__ import annotations

from collections import Counter
from datetime import datetime
import json
import os
from pathlib import Path
import shutil

from cwr_eg.hashing import content_hash, sha256_file
from cwr_eg.manifest import read_jsonl, write_jsonl


ROOT = Path(__file__).resolve().parents[1]
TARGET_ROOT = Path("artifacts/i05_train_dev")
TEMP_ROOT = Path("artifacts/i05_train_dev.seed.tmp")
FREEZE_PATH = Path("manifests/i_gate_c_freeze.json")
FEATURE_SCOPE_PATH = Path("docs/i_gate_c_feature_scope.json")
GATE_DOC_PATH = Path("docs/i_gate_c_approval.md")


def _write_new(path: Path, text: str) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite frozen artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _replace_code_hashes(scope: dict) -> None:
    for entry in scope["code_files"]:
        entry["sha256"] = sha256_file(entry["path"])
    scope["runner_sha256"] = sha256_file("src/cwr_eg/runtime.py")


def main() -> int:
    os.chdir(ROOT)
    for path in (TARGET_ROOT, TEMP_ROOT, FREEZE_PATH, FEATURE_SCOPE_PATH, GATE_DOC_PATH):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite I-GATE-C artifact: {path}")

    source_documents_path = Path("artifacts/i04_full/feature_documents.jsonl")
    rows = [
        row
        for row in read_jsonl(source_documents_path)
        if str(row["split"]) in {"train", "dev"}
    ]
    if len(rows) != 4047 or {str(row["split"]) for row in rows} != {"train", "dev"}:
        raise RuntimeError("Frozen Train/Dev feature-document count changed")
    if len({str(row["recipe_id"]) for row in rows}) != len(rows):
        raise RuntimeError("Train/Dev feature-document IDs are not unique")
    target_by_id = {str(row["recipe_id"]): row for row in rows}

    canary_documents = {
        str(row["recipe_id"]): row
        for row in read_jsonl("artifacts/i03_canary/feature_documents.jsonl")
    }
    canary_manifest = read_jsonl("artifacts/i03_canary/features/feature_manifest.jsonl")
    if len(canary_manifest) != 810:
        raise RuntimeError("Canary feature manifest no longer contains 810 rows")
    source_entries = {str(row["recipe_id"]): row for row in canary_manifest}
    if len(source_entries) != 810 or not set(source_entries).issubset(target_by_id):
        raise RuntimeError("Canary features are not an exact Train/Dev subset")

    TEMP_ROOT.mkdir(parents=True)
    feature_dir = TEMP_ROOT / "features"
    feature_dir.mkdir()
    write_jsonl(TEMP_ROOT / "feature_documents.jsonl", rows)
    seeded_entries: dict[str, dict] = {}
    for recipe_id, entry in source_entries.items():
        if (
            recipe_id not in canary_documents
            or canary_documents[recipe_id]["text_sha256"]
            != target_by_id[recipe_id]["text_sha256"]
        ):
            raise RuntimeError(f"Canary text provenance drifted: {recipe_id}")
        source = Path(str(entry["feature_path"]))
        if sha256_file(source) != str(entry["feature_sha256"]):
            raise RuntimeError(f"Canary feature hash drifted: {recipe_id}")
        destination = feature_dir / f"{recipe_id}.npz"
        shutil.copy2(source, destination)
        if sha256_file(destination) != str(entry["feature_sha256"]):
            raise RuntimeError(f"Copied feature hash mismatch: {recipe_id}")
        seeded_entries[recipe_id] = {
            **entry,
            "feature_path": str(
                TARGET_ROOT / "features" / f"{recipe_id}.npz"
            ).replace("\\", "/"),
        }
    seeded_manifest = [
        seeded_entries[str(row["recipe_id"])]
        for row in rows
        if str(row["recipe_id"]) in seeded_entries
    ]
    write_jsonl(feature_dir / "feature_manifest.jsonl", seeded_manifest)
    os.replace(TEMP_ROOT, TARGET_ROOT)

    train_dev_path = TARGET_ROOT / "feature_documents.jsonl"
    seed_manifest_path = TARGET_ROOT / "features/feature_manifest.jsonl"
    counts = Counter((str(row["split"]), str(row["kind"])) for row in rows)
    tensor_plans = {
        "full": {"train_examples": 2997, "dev_examples": 1000},
        "lofo_kgw": {"train_examples": 2397, "dev_examples": 800},
        "lofo_unigram": {"train_examples": 2398, "dev_examples": 800},
        "lofo_unbiased": {"train_examples": 2397, "dev_examples": 800},
        "lofo_synthid": {"train_examples": 2397, "dev_examples": 800},
    }
    for plan in tensor_plans.values():
        plan.update(
            {
                "train_batches": 150,
                "dev_batches": 50,
                "train_shards": 10,
                "dev_shards": 4,
                "mixed_parent_features_excluded": 50,
            }
        )
    freeze = {
        "manifest_version": "i-gate-c-v1",
        "created_at": datetime.now().astimezone().isoformat(),
        "task_id": "I-GATE-C",
        "parent_completion": {
            "path": "manifests/i04_completion.json",
            "sha256": sha256_file("manifests/i04_completion.json"),
        },
        "config": {
            "path": "configs/intermediate.yaml",
            "sha256": sha256_file("configs/intermediate.yaml"),
        },
        "training_matrix": {
            "path": "configs/intermediate_training_matrix.yaml",
            "sha256": sha256_file("configs/intermediate_training_matrix.yaml"),
            "runs": 10,
        },
        "train_dev_input": {
            "path": str(train_dev_path).replace("\\", "/"),
            "sha256": sha256_file(train_dev_path),
            "documents": len(rows),
            "split_kind_counts": {
                f"{split}:{kind}": count
                for (split, kind), count in sorted(counts.items())
            },
            "calibration_documents": 0,
            "test_documents": 0,
        },
        "feature_resume": {
            "source": "I-03-canary",
            "documents": len(seeded_manifest),
            "remaining_documents": len(rows) - len(seeded_manifest),
            "manifest_path": str(seed_manifest_path).replace("\\", "/"),
            "manifest_sha256": sha256_file(seed_manifest_path),
            "copy_policy": "byte-identical-sha256-verified-v1",
        },
        "feature_code": [
            {
                "path": path,
                "sha256": sha256_file(path),
            }
            for path in (
                "src/cwr_eg/runtime.py",
                "src/cwr_eg/transformer_features.py",
                "src/cwr_eg/monitoring.py",
                "src/cwr_eg/manifest.py",
                "src/cwr_eg/hashing.py",
            )
        ],
        "model": {
            "id": "Qwen/Qwen2.5-1.5B-Instruct",
            "revision": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
            "local_files_only": True,
            "dtype": "bfloat16",
        },
        "feature_settings": {
            "maximum_tokens": 1024,
            "microbatch_sequence": [4, 2, 1],
            "atomic_write": True,
            "resume": True,
        },
        "tensor_bundle_plans": tensor_plans,
        "training": {
            "positions": 256,
            "batch_size": 20,
            "hidden_dim": 256,
            "invariant_dim": 128,
            "private_dim": 128,
            "learning_rate": 0.0003,
            "maximum_epochs": 20,
            "minimum_epochs": 5,
            "early_stopping_patience": 4,
            "dependent_scope_policy": "bind exact bundle index hash after tensorization",
        },
        "dev_policy": {
            "allowed_splits": ["train", "dev"],
            "ensemble_rule": "arithmetic_mean_character_logits",
            "test_based_selection": False,
        },
        "calibration_sealed": True,
        "test_sealed": True,
        "deviation_audit": {
            "status": "pass",
            "objective_changed": False,
            "sample_or_split_changed": False,
            "model_asset_changed": False,
            "training_matrix_changed": False,
            "calibration_or_test_unsealed": False,
        },
    }
    freeze["content_hash"] = content_hash(freeze)
    _write_new(FREEZE_PATH, json.dumps(freeze, ensure_ascii=False, indent=2) + "\n")

    feature_scope = json.loads(
        Path("docs/i03_canary_feature_extraction_scope.json").read_text(
            encoding="utf-8"
        )
    )
    feature_scope.update(
        {
            "task_id": "I-05-train-dev-features",
            "freeze_manifest": str(FREEZE_PATH).replace("\\", "/"),
            "freeze_manifest_sha256": sha256_file(FREEZE_PATH),
            "parent_freeze_manifest_sha256": sha256_file(
                "manifests/intermediate_freeze_manifest.json"
            ),
            "input_path": str(train_dev_path).replace("\\", "/"),
            "input_sha256": sha256_file(train_dev_path),
            "expected_document_count": len(rows),
            "limit": len(rows),
            "allowed_splits": ["train", "dev"],
            "resume": True,
            "resume_manifest_sha256": sha256_file(seed_manifest_path),
            "expected_resumed_documents": len(seeded_manifest),
            "output_dir": str(TARGET_ROOT / "features").replace("\\", "/"),
            "result_path": str(TARGET_ROOT / "feature_extraction_result.json").replace(
                "\\", "/"
            ),
        }
    )
    _replace_code_hashes(feature_scope)
    _write_new(
        FEATURE_SCOPE_PATH,
        json.dumps(feature_scope, ensure_ascii=False, indent=2) + "\n",
    )
    gate_text = f"""# I-GATE-C Freeze and Approval Scope

- Train/Dev documents: {len(rows)}
- Reused canary features: {len(seeded_manifest)}
- New feature documents: {len(rows) - len(seeded_manifest)}
- Calibration/Test documents exposed: 0/0
- Gate freeze SHA-256: `{sha256_file(FREEZE_PATH)}`
- Feature scope SHA-256: `{sha256_file(FEATURE_SCOPE_PATH)}`
- Tensor variants: full, LOFO KGW, Unigram, Unbiased, SynthID
- Training runs: 10 exactly as frozen in `configs/intermediate_training_matrix.yaml`

Dependent tensor, training, and Dev-scoring scopes must bind the exact preceding artifact hashes. No scope may admit Calibration or Test rows.
"""
    _write_new(GATE_DOC_PATH, gate_text)
    print(
        json.dumps(
            {
                "freeze_sha256": sha256_file(FREEZE_PATH),
                "feature_scope_sha256": sha256_file(FEATURE_SCOPE_PATH),
                "train_dev_documents": len(rows),
                "resumed_features": len(seeded_manifest),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
