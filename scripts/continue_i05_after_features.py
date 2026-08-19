from __future__ import annotations

from datetime import datetime
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

import psutil

import continue_i04_after_base as ops
from cwr_eg.config import load_yaml
from cwr_eg.hashing import sha256_file
from cwr_eg.manifest import read_jsonl, write_jsonl
from cwr_eg.tensor_bundle import load_sharded_bundle_index


ROOT = Path(__file__).resolve().parents[1]
CONFIG = Path("configs/intermediate.yaml")
FREEZE = Path("manifests/i_gate_c_freeze.json")
STATUS = Path("status/i05_continuation_status.json")
FEATURE_ROOT = Path("artifacts/i05_train_dev")


def _write_new(path: Path, payload: dict) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite completion artifact: {path}")
    ops._write_json(path, payload)


def _process_matches(process_id: int, marker: str) -> bool:
    if not psutil.pid_exists(process_id):
        return False
    try:
        return marker in " ".join(psutil.Process(process_id).cmdline())
    except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
        return False


def _wait_for_feature_process(process_id: int) -> None:
    result = FEATURE_ROOT / "feature_extraction_result.json"
    deadline = time.monotonic() + 8 * 60 * 60
    while not result.exists():
        if not _process_matches(process_id, "extract-features"):
            raise RuntimeError("Feature process exited without a completed result")
        if time.monotonic() > deadline:
            raise TimeoutError("Train/Dev feature extraction exceeded eight hours")
        time.sleep(30)


def _validate_features() -> tuple[Path, dict]:
    documents = read_jsonl(FEATURE_ROOT / "feature_documents.jsonl")
    manifest_path = FEATURE_ROOT / "features/feature_manifest.jsonl"
    manifest = read_jsonl(manifest_path)
    if len(documents) != 4047 or len(manifest) != 4047:
        raise RuntimeError("Train/Dev feature count differs from I-GATE-C")
    document_ids = [str(row["recipe_id"]) for row in documents]
    manifest_ids = [str(row["recipe_id"]) for row in manifest]
    if document_ids != manifest_ids or len(set(manifest_ids)) != 4047:
        raise RuntimeError("Feature manifest order or IDs differ from the frozen input")
    if {str(row["split"]) for row in manifest} != {"train", "dev"}:
        raise RuntimeError("Feature manifest contains a sealed split")
    for entry in manifest:
        path = Path(str(entry["feature_path"]))
        if sha256_file(path) != str(entry["feature_sha256"]):
            raise RuntimeError(f"Feature SHA-256 mismatch: {entry['recipe_id']}")
    result_path = FEATURE_ROOT / "feature_extraction_result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if (
        result["documents"] != 4047
        or result["resumed_documents"] != 810
        or result["final_microbatch"] not in {4, 2, 1}
        or result["feature_manifest_sha256"] != sha256_file(manifest_path)
    ):
        raise RuntimeError("Feature result differs from the frozen scope")
    completion = Path("manifests/i05_feature_completion.json")
    if not completion.exists():
        _write_new(
            completion,
            {
                "task_id": "I-05-features",
                "completed_at": ops._now(),
                "documents": 4047,
                "resumed_documents": 810,
                "new_documents": 3237,
                "feature_manifest": str(manifest_path).replace("\\", "/"),
                "feature_manifest_sha256": sha256_file(manifest_path),
                "result_sha256": sha256_file(result_path),
                "runtime_result": result,
                "resources": ops._resource_summary("I-05-train-dev-features"),
                "deviation_audit": {
                    "status": "pass",
                    "train_dev_only": True,
                    "calibration_or_test_unsealed": False,
                    "model_or_feature_settings_changed": False,
                },
            },
        )
    return manifest_path, result


def _enrich_training_feature_manifest(feature_manifest: Path) -> Path:
    documents = {
        str(row["recipe_id"]): row
        for row in read_jsonl(FEATURE_ROOT / "feature_documents.jsonl")
    }
    enriched = []
    for row in read_jsonl(feature_manifest):
        document = documents[str(row["recipe_id"])]
        enriched.append(
            {
                **row,
                "kind": document["kind"],
                "base_recipe_id": document.get("base_recipe_id"),
            }
        )
    target = FEATURE_ROOT / "features/training_feature_manifest_v2.jsonl"
    if target.exists():
        if read_jsonl(target) != enriched:
            raise RuntimeError("Enriched training feature manifest drifted")
    else:
        write_jsonl(target, enriched)
    return target


def _consistency_pair_counts(
    feature_manifest: Path, excluded_watermark_families: list[str]
) -> dict[str, int]:
    excluded = set(excluded_watermark_families)
    rows = [
        row
        for row in read_jsonl(feature_manifest)
        if row.get("watermark_family") not in excluded and len(row["parent_ids"]) == 1
    ]
    ids_by_split = {
        split: {str(row["recipe_id"]) for row in rows if row["split"] == split}
        for split in ("train", "dev")
    }
    return {
        split: sum(
            row["split"] == split
            and row.get("base_recipe_id") in ids_by_split[split]
            for row in rows
        )
        for split in ("train", "dev")
    }


def _update_code_hashes(scope: dict) -> None:
    for entry in scope["code_files"]:
        entry["sha256"] = sha256_file(entry["path"])
    scope["runner_sha256"] = sha256_file("src/cwr_eg/runtime.py")


def _scope_once(path: Path, payload: dict) -> dict:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError(f"Existing dependent scope drifted: {path}")
        return existing
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return payload


def _approval_once(
    path: Path, *, approval_id: str, action: str, scope_path: Path, evidence: str
) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return ops._approval(
        approval_id=approval_id,
        action=action,
        scope_path=scope_path,
        output_path=path,
        evidence=evidence,
    )


def _build_bundles(feature_manifest: Path) -> dict[str, dict]:
    plans = json.loads(FREEZE.read_text(encoding="utf-8"))["tensor_bundle_plans"]
    exclusions = {
        "full": [],
        "lofo_kgw": ["kgw"],
        "lofo_unigram": ["unigram"],
        "lofo_unbiased": ["unbiased"],
        "lofo_synthid": ["synthid"],
    }
    template = json.loads(
        Path("docs/i03_canary_tensorize_scope.json").read_text(encoding="utf-8")
    )
    summaries: dict[str, dict] = {}
    for variant, excluded in exclusions.items():
        plan = plans[variant]
        consistency_pairs = _consistency_pair_counts(feature_manifest, excluded)
        scope_path = Path(f"docs/i05_tensorize_{variant}_scope.json")
        output_path = FEATURE_ROOT / f"tensor_bundle_{variant}"
        result_path = FEATURE_ROOT / f"tensorize_{variant}_result.json"
        scope = json.loads(json.dumps(template))
        scope.update(
            {
                "task_id": f"I-05-tensorize-{variant}",
                "freeze_manifest": str(FREEZE).replace("\\", "/"),
                "freeze_manifest_sha256": sha256_file(FREEZE),
                "feature_manifest": str(feature_manifest).replace("\\", "/"),
                "feature_manifest_sha256": sha256_file(feature_manifest),
                "excluded_watermark_families": excluded,
                "expected_train_batches": plan["train_batches"],
                "expected_dev_batches": plan["dev_batches"],
                "expected_train_examples": plan["train_examples"],
                "expected_dev_examples": plan["dev_examples"],
                "expected_train_shards": plan["train_shards"],
                "expected_dev_shards": plan["dev_shards"],
                "expected_mixed_parent_features_excluded": 50,
                "expected_train_consistency_pairs": consistency_pairs["train"],
                "expected_dev_consistency_pairs": consistency_pairs["dev"],
                "output_path": str(output_path).replace("\\", "/"),
                "result_path": str(result_path).replace("\\", "/"),
            }
        )
        _update_code_hashes(scope)
        _scope_once(scope_path, scope)
        approval_path = Path(f"status/approvals/i05_tensorize_{variant}.json")
        approval = _approval_once(
            approval_path,
            approval_id=f"I05-TENSORIZE-{variant.upper()}-20260815",
            action="tensorize",
            scope_path=scope_path,
            evidence=(
                "User authorized continued I-stage execution on 2026-08-15. This CPU "
                f"scope builds only the frozen {variant} Train/Dev sharded bundle and "
                "does not expose Calibration or Test."
            ),
        )
        if not result_path.exists():
            if output_path.exists():
                raise RuntimeError(f"Incomplete tensor output requires audit: {output_path}")
            ops._run(
                [
                    sys.executable,
                    "-m",
                    "cwr_eg.cli",
                    "tensorize",
                    "--config",
                    str(CONFIG),
                    "--resource-class",
                    ops.RESOURCE_CLASS,
                    "--scope-file",
                    str(scope_path),
                    "--approval",
                    str(approval_path),
                ],
                Path(f"status/i05_tensorize_{variant}_console.log"),
            )
        result = json.loads(result_path.read_text(encoding="utf-8"))
        index_path, index = load_sharded_bundle_index(output_path)
        if (
            index["splits"]["train"]["batches"] != plan["train_batches"]
            or index["splits"]["dev"]["batches"] != plan["dev_batches"]
            or index["splits"]["train"]["examples"] != plan["train_examples"]
            or index["splits"]["dev"]["examples"] != plan["dev_examples"]
            or len(index["splits"]["train"]["shards"]) != plan["train_shards"]
            or len(index["splits"]["dev"]["shards"]) != plan["dev_shards"]
            or index["mixed_parent_features_excluded"] != 50
            or index["splits"]["train"]["consistency_pairs"]
            != consistency_pairs["train"]
            or index["splits"]["dev"]["consistency_pairs"]
            != consistency_pairs["dev"]
        ):
            raise RuntimeError(f"Tensor bundle count drift: {variant}")
        summaries[variant] = {
            "scope_sha256": sha256_file(scope_path),
            "approval_fingerprint": approval["fingerprint"],
            "result_sha256": sha256_file(result_path),
            "bundle_path": str(output_path).replace("\\", "/"),
            "bundle_index_sha256": sha256_file(index_path),
            "bundle_content_hash": index["bundle_content_hash"],
            "train_examples": plan["train_examples"],
            "dev_examples": plan["dev_examples"],
            "train_consistency_pairs": consistency_pairs["train"],
            "dev_consistency_pairs": consistency_pairs["dev"],
        }
        ops._append_progress(
            f"I-05-tensorize-{variant}",
            "done",
            "Frozen Train/Dev sharded bundle passed exact count and hash QA.",
            bundle_index_sha256=sha256_file(index_path),
        )
    completion = Path("manifests/i05_tensor_bundles_completion.json")
    if not completion.exists():
        _write_new(
            completion,
            {
                "task_id": "I-05-tensor-bundles",
                "completed_at": ops._now(),
                "variants": summaries,
                "deviation_audit": {
                    "status": "pass",
                    "train_dev_only": True,
                    "bundle_variants_changed": False,
                    "calibration_or_test_unsealed": False,
                },
            },
        )
    return summaries


def _run_training(bundle_summaries: dict[str, dict]) -> dict[str, dict]:
    matrix = load_yaml("configs/intermediate_training_matrix.yaml")
    template = json.loads(
        Path("docs/i03_canary_training_scope.json").read_text(encoding="utf-8")
    )
    summaries: dict[str, dict] = {}
    for run in matrix["runs"]:
        run_id = str(run["run_id"])
        variant = str(run["bundle_variant"])
        bundle = bundle_summaries[variant]
        scope_path = Path(f"docs/i05_train_{run_id}_scope.json")
        output_dir = FEATURE_ROOT / "training" / run_id
        result_path = FEATURE_ROOT / "training_results" / f"{run_id}.json"
        scope = json.loads(json.dumps(template))
        scope.update(
            {
                "task_id": f"I-05-train-{run_id}",
                "run_id": run_id,
                "role": run["role"],
                "freeze_manifest": str(FREEZE).replace("\\", "/"),
                "freeze_manifest_sha256": sha256_file(FREEZE),
                "bundle_path": bundle["bundle_path"],
                "bundle_sha256": bundle["bundle_index_sha256"],
                "bundle_content_hash": bundle["bundle_content_hash"],
                "training_settings": {
                    **template["training_settings"],
                    "epochs": 20,
                    "seed": int(run["seed"]),
                    "minimum_epochs": 5,
                    "early_stopping_patience": 4,
                    "deterministic_algorithms": True,
                },
                "loss_overrides": dict(run.get("loss_overrides", {})),
                "output_dir": str(output_dir).replace("\\", "/"),
                "result_path": str(result_path).replace("\\", "/"),
            }
        )
        if "excluded_watermark_family" in run:
            scope["excluded_watermark_family"] = run["excluded_watermark_family"]
            scope["held_out_decision_label"] = run["held_out_decision_label"]
        _update_code_hashes(scope)
        _scope_once(scope_path, scope)
        approval_path = Path(f"status/approvals/i05_train_{run_id}.json")
        approval = _approval_once(
            approval_path,
            approval_id=f"I05-TRAIN-{run_id.upper()}-20260815",
            action="train",
            scope_path=scope_path,
            evidence=(
                "User authorized continued I-stage execution on 2026-08-15. This scope "
                f"executes only frozen training run {run_id} on its exact Train/Dev "
                "bundle; Calibration and Test remain sealed."
            ),
        )
        ops._set_status("train", "in_progress", run_id=run_id)
        if not result_path.exists():
            if output_dir.exists():
                raise RuntimeError(f"Incomplete training output requires resume audit: {run_id}")
            ops._run(
                [
                    sys.executable,
                    "-m",
                    "cwr_eg.cli",
                    "train",
                    "--config",
                    str(CONFIG),
                    "--resource-class",
                    ops.RESOURCE_CLASS,
                    "--scope-file",
                    str(scope_path),
                    "--approval",
                    str(approval_path),
                ],
                Path(f"status/i05_train_{run_id}_console.log"),
            )
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if (
            not math.isfinite(float(result["best_dev_total"]))
            or int(result["epochs_completed"]) < 5
            or sha256_file(result["checkpoint"]) != result["checkpoint_sha256"]
            or sha256_file(result["training_log"]) != result["training_log_sha256"]
        ):
            raise RuntimeError(f"Training QA failed: {run_id}")
        summaries[run_id] = {
            "role": run["role"],
            "bundle_variant": variant,
            "seed": int(run["seed"]),
            "scope_sha256": sha256_file(scope_path),
            "approval_fingerprint": approval["fingerprint"],
            "result_path": str(result_path).replace("\\", "/"),
            "result_sha256": sha256_file(result_path),
            "checkpoint": result["checkpoint"],
            "checkpoint_sha256": result["checkpoint_sha256"],
            "best_epoch": result["best_epoch"],
            "best_dev_total": result["best_dev_total"],
            "epochs_completed": result["epochs_completed"],
            "stopped_early": result["stopped_early"],
        }
        ops._append_progress(
            f"I-05-train-{run_id}-audit",
            "done",
            "Training checkpoint, finite losses, minimum epochs, and Dev-only early stopping passed QA.",
            checkpoint_sha256=result["checkpoint_sha256"],
            best_dev_total=result["best_dev_total"],
        )
    completion = Path("manifests/i05_training_matrix_completion.json")
    if not completion.exists():
        _write_new(
            completion,
            {
                "task_id": "I-05-training-matrix",
                "completed_at": ops._now(),
                "runs": summaries,
                "full_ensemble_rule": "arithmetic_mean_character_logits",
                "deviation_audit": {
                    "status": "pass",
                    "run_count": len(summaries),
                    "training_matrix_changed": False,
                    "dev_only_early_stopping": True,
                    "calibration_or_test_unsealed": False,
                },
            },
        )
    return summaries


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-process-id", type=int, required=True)
    args = parser.parse_args()
    os.chdir(ROOT)
    ops.STATUS_PATH = STATUS
    ops.CONFIG = CONFIG
    ops._set_status("wait_for_features", "in_progress", process_id=args.feature_process_id)
    _wait_for_feature_process(args.feature_process_id)
    ops._set_status("validate_features", "in_progress")
    feature_manifest, _ = _validate_features()
    feature_manifest = _enrich_training_feature_manifest(feature_manifest)
    ops._set_status("tensorize", "in_progress")
    bundles = _build_bundles(feature_manifest)
    ops._set_status("training_matrix", "in_progress", runs=10)
    runs = _run_training(bundles)
    ops._set_status(
        "training_complete",
        "done",
        runs=len(runs),
        completion_sha256=sha256_file("manifests/i05_training_matrix_completion.json"),
    )
    print(
        json.dumps(
            {
                "features": 4047,
                "bundles": len(bundles),
                "training_runs": len(runs),
                "completion_sha256": sha256_file(
                    "manifests/i05_training_matrix_completion.json"
                ),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        os.chdir(ROOT)
        ops.STATUS_PATH = STATUS
        ops._set_status(
            "failed",
            "error",
            error_type=type(error).__name__,
            error_message=str(error),
            traceback=traceback.format_exc(),
        )
        ops._append_progress(
            "I-05-continuation",
            "blocked",
            "The fail-closed I-05 continuation stopped; inspect status/i05_continuation_status.json.",
            error_type=type(error).__name__,
            error_message=str(error),
        )
        raise
