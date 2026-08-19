from __future__ import annotations

import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Any

import psutil

import continue_i04_after_base as ops
import continue_i05_after_features as stage
from cwr_eg.hashing import sha256_file
from cwr_eg.manifest import read_jsonl


ROOT = Path(__file__).resolve().parents[1]
CONFIG = Path("configs/intermediate.yaml")
FREEZE = Path("manifests/i_gate_c_freeze.json")
FEATURE_ROOT = Path("artifacts/i05_train_dev")
TRAINING_COMPLETION = Path("manifests/i05_training_matrix_completion.json")
STATUS = Path("status/i05_post_training_status.json")


def _wait_for_training(process_id: int) -> None:
    deadline = time.monotonic() + 18 * 60 * 60
    while not TRAINING_COMPLETION.exists():
        if not psutil.pid_exists(process_id):
            raise RuntimeError("I-05 continuation exited without a training completion manifest")
        if time.monotonic() > deadline:
            raise TimeoutError("I-05 training matrix exceeded eighteen hours")
        time.sleep(30)


def _checkpoints() -> dict[str, dict[str, Any]]:
    payload = json.loads(TRAINING_COMPLETION.read_text(encoding="utf-8"))
    runs = payload["runs"]
    if len(runs) != 10:
        raise RuntimeError("Training completion does not contain ten runs")
    for run_id, row in runs.items():
        if sha256_file(row["checkpoint"]) != row["checkpoint_sha256"]:
            raise RuntimeError(f"Checkpoint drift before scoring: {run_id}")
    return runs


def _score_specs(runs: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    full = [runs[f"full_seed_{seed}"] for seed in (20260815, 20260816, 20260817)]
    specs: list[dict[str, Any]] = [
        {
            "name": "full_ensemble",
            "checkpoints": full,
            "recipe_ids": None,
            "expected": 4047,
            "rule": "arithmetic_mean_character_logits_for_three_full_models",
        }
    ]
    for run_id, row in runs.items():
        specs.append(
            {
                "name": run_id,
                "checkpoints": [row],
                "recipe_ids": "dev_single_parent",
                "expected": 1000,
                "rule": "single_model_dev_stability_or_ablation",
            }
        )
    return specs


def _scope_for(spec: dict[str, Any], dev_ids: list[str]) -> tuple[Path, Path, Path]:
    template = json.loads(Path("docs/i03_canary_checkpoint_scoring_scope.json").read_text(encoding="utf-8"))
    name = spec["name"]
    scope_path = Path(f"docs/i05_score_{name}_scope.json")
    output_path = FEATURE_ROOT / "checkpoint_scores" / f"{name}.jsonl"
    result_path = FEATURE_ROOT / "checkpoint_score_results" / f"{name}.json"
    feature_manifest = FEATURE_ROOT / "features/feature_manifest.jsonl"
    documents = FEATURE_ROOT / "feature_documents.jsonl"
    scope = json.loads(json.dumps(template))
    scope.update(
        {
            "task_id": f"I-05-score-{name}",
            "role": "train_dev_checkpoint_scoring",
            "freeze_manifest": str(FREEZE).replace("\\", "/"),
            "freeze_manifest_sha256": sha256_file(FREEZE),
            "checkpoints": [
                {"path": row["checkpoint"], "sha256": row["checkpoint_sha256"]}
                for row in spec["checkpoints"]
            ],
            "ensemble_rule": spec["rule"],
            "feature_manifest": str(feature_manifest).replace("\\", "/"),
            "feature_manifest_sha256": sha256_file(feature_manifest),
            "documents_path": str(documents).replace("\\", "/"),
            "documents_sha256": sha256_file(documents),
            "expected_document_count": spec["expected"],
            "allowed_splits": ["train", "dev"],
            "output_path": str(output_path).replace("\\", "/"),
            "result_path": str(result_path).replace("\\", "/"),
        }
    )
    if spec["recipe_ids"] == "dev_single_parent":
        scope["recipe_ids"] = dev_ids
    else:
        scope.pop("recipe_ids", None)
    stage._update_code_hashes(scope)
    stage._scope_once(scope_path, scope)
    return scope_path, output_path, result_path


def _validate_scores(path: Path, expected_ids: set[str], documents: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows = read_jsonl(path)
    ids = [str(row["recipe_id"]) for row in rows]
    if len(rows) != len(expected_ids) or set(ids) != expected_ids or len(set(ids)) != len(ids):
        raise RuntimeError(f"Checkpoint score IDs differ from scope: {path}")
    minimum_coverage = 1.0
    uncertain = 0
    for row in rows:
        document = documents[str(row["recipe_id"])]
        coverage = float(row["mapping_coverage"])
        minimum_coverage = min(minimum_coverage, coverage)
        uncertain += row.get("validity_override") == "uncertain"
        scalar_values = [row["generic_residual_score"], row["watermark_logit"], coverage]
        vector_values = row["character_logits"] + row["invariant_embedding"] + row["private_embedding"]
        if len(row["character_logits"]) != len(document["text"]):
            raise RuntimeError(f"Character score length mismatch: {row['recipe_id']}")
        if not all(math.isfinite(float(value)) for value in scalar_values + vector_values):
            raise RuntimeError(f"Non-finite checkpoint score: {row['recipe_id']}")
        if (coverage < 0.98) != (row.get("validity_override") == "uncertain"):
            raise RuntimeError(f"Coverage uncertainty mismatch: {row['recipe_id']}")
    return {
        "documents": len(rows),
        "output_sha256": sha256_file(path),
        "mapping_coverage_minimum": minimum_coverage,
        "uncertain": uncertain,
    }


def _run_scores(runs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    feature_rows = read_jsonl(FEATURE_ROOT / "features/feature_manifest.jsonl")
    all_ids = {str(row["recipe_id"]) for row in feature_rows}
    dev_ids = [
        str(row["recipe_id"])
        for row in feature_rows
        if row["split"] == "dev" and len(row["parent_ids"]) == 1
    ]
    if len(all_ids) != 4047 or len(dev_ids) != 1000:
        raise RuntimeError("Train/Dev scoring population drifted")
    documents = {
        str(row["recipe_id"]): row
        for row in read_jsonl(FEATURE_ROOT / "feature_documents.jsonl")
    }
    summaries: dict[str, Any] = {}
    for spec in _score_specs(runs):
        name = str(spec["name"])
        scope_path, output_path, result_path = _scope_for(spec, dev_ids)
        approval_path = Path(f"status/approvals/i05_score_{name}.json")
        approval = stage._approval_once(
            approval_path,
            approval_id=f"I05-SCORE-{name.upper()}-20260815",
            action="score-checkpoint",
            scope_path=scope_path,
            evidence=(
                "User authorized continued I-stage GPU execution on 2026-08-15. "
                f"This exact scope scores {name} only on frozen Train/Dev inputs; "
                "Calibration and Test remain sealed."
            ),
        )
        ops._set_status("checkpoint_scoring", "in_progress", score_name=name)
        if not result_path.exists():
            if output_path.exists() and not output_path.with_suffix(output_path.suffix + ".partial").exists():
                raise RuntimeError(f"Unpaired score output requires audit: {output_path}")
            ops._run(
                [
                    sys.executable,
                    "-m",
                    "cwr_eg.cli",
                    "score-checkpoint",
                    "--config",
                    str(CONFIG),
                    "--resource-class",
                    ops.RESOURCE_CLASS,
                    "--scope-file",
                    str(scope_path),
                    "--approval",
                    str(approval_path),
                ],
                Path(f"status/i05_score_{name}_console.log"),
            )
        expected_ids = all_ids if name == "full_ensemble" else set(dev_ids)
        qa = _validate_scores(output_path, expected_ids, documents)
        summaries[name] = {
            **qa,
            "scope_sha256": sha256_file(scope_path),
            "approval_fingerprint": approval["fingerprint"],
            "result_sha256": sha256_file(result_path),
            "checkpoint_sha256": [row["checkpoint_sha256"] for row in spec["checkpoints"]],
        }
        ops._append_progress(
            f"I-05-score-{name}-audit",
            "done",
            "Real checkpoint scores passed exact-ID, finite-value, mapping, and split QA.",
            output_sha256=qa["output_sha256"],
        )
    completion = Path("manifests/i05_checkpoint_scoring_completion.json")
    if not completion.exists():
        stage._write_new(
            completion,
            {
                "task_id": "I-05-checkpoint-scoring",
                "completed_at": ops._now(),
                "scores": summaries,
                "deviation_audit": {
                    "status": "pass",
                    "primary_ensemble_rule_changed": False,
                    "individual_models_used_for_stability_only": True,
                    "calibration_or_test_unsealed": False,
                },
            },
        )
    return summaries


def _run_analysis() -> dict[str, Any]:
    output_json = FEATURE_ROOT / "dev_analysis.json"
    output_md = Path("reports/i05_train_dev_analysis.md")
    if not output_json.exists():
        ops._run(
            [
                sys.executable,
                "scripts/analyze_i05_dev.py",
                "--ensemble",
                str(FEATURE_ROOT / "checkpoint_scores/full_ensemble.jsonl"),
                "--model-scores-dir",
                str(FEATURE_ROOT / "checkpoint_scores"),
                "--matrix",
                str(TRAINING_COMPLETION),
                "--output-json",
                str(output_json),
                "--output-md",
                str(output_md),
            ],
            Path("status/i05_dev_analysis_console.log"),
        )
    result = json.loads(output_json.read_text(encoding="utf-8"))
    if result["deviation_audit"]["calibration_or_test_read"]:
        raise RuntimeError("Dev analysis crossed the sealed split boundary")
    completion = Path("manifests/i05_dev_analysis_completion.json")
    if not completion.exists():
        stage._write_new(
            completion,
            {
                "task_id": "I-05-dev-analysis",
                "completed_at": ops._now(),
                "analysis_sha256": sha256_file(output_json),
                "report_sha256": sha256_file(output_md),
                "deviation_audit": result["deviation_audit"],
            },
        )
    return result


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--training-process-id", type=int, required=True)
    args = parser.parse_args()
    os.chdir(ROOT)
    ops.STATUS_PATH = STATUS
    ops.CONFIG = CONFIG
    ops._set_status("wait_for_training", "in_progress", process_id=args.training_process_id)
    _wait_for_training(args.training_process_id)
    runs = _checkpoints()
    ops._set_status("checkpoint_scoring", "in_progress", scopes=11)
    scores = _run_scores(runs)
    ops._set_status("dev_analysis", "in_progress")
    analysis = _run_analysis()
    ops._set_status(
        "i05_compute_complete",
        "done",
        score_scopes=len(scores),
        primary_generic_auc=analysis["primary_dev"]["generic_only"]["watermark_auc"],
    )
    print(json.dumps({"score_scopes": len(scores), "status": "done"}, sort_keys=True))
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
            "I-05-post-training",
            "blocked",
            "Fail-closed post-training continuation stopped; inspect status/i05_post_training_status.json.",
            error_type=type(error).__name__,
            error_message=str(error),
        )
        raise
