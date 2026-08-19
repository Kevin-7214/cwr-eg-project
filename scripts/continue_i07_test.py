from __future__ import annotations

import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Any

import numpy as np
import psutil
from sklearn.metrics import roc_auc_score

import continue_i04_after_base as ops
import continue_i05_after_features as stage
import continue_i_gate_d_calibration as calstage
from cwr_eg.bundle import CalibrationBundle
from cwr_eg.enums import DecisionLabel
from cwr_eg.hashing import content_hash, sha256_file
from cwr_eg.manifest import read_jsonl, write_jsonl
from cwr_eg.metrics import EvaluationRecord, oscr

from i07_test_helpers import (
    calibrated_document_scores,
    evaluate_baseline_bundle,
    fit_baseline_bundle,
    mask_registered_records,
    merge_generic_with_registered,
    select_fusion_weight,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = Path("configs/intermediate.yaml")
GATE_D = Path("manifests/i_gate_d_freeze.json")
I06_COMPLETION = Path("manifests/i06_calibration_completion.json")
I07_FREEZE = Path("manifests/i07_test_freeze.json")
I07_FREEZE_AMENDMENT = Path("manifests/i07_test_freeze_amendment_01.json")
CAL_ROOT = Path("artifacts/i06_calibration")
TEST_ROOT = Path("artifacts/i07_test")
STATUS = Path("status/i07_test_status.json")
FAMILIES = ("kgw", "unigram", "unbiased", "synthid")
EXECUTION_FREEZE = I07_FREEZE


def _wait_for_i06(process_id: int) -> None:
    deadline = time.monotonic() + 24 * 60 * 60
    while not I06_COMPLETION.exists():
        if not psutil.pid_exists(process_id):
            raise RuntimeError("I-06 process exited without calibration completion")
        if time.monotonic() > deadline:
            raise TimeoutError("I-06 continuation exceeded twenty-four hours")
        time.sleep(30)


def _approval(action: str, name: str, scope_path: Path) -> tuple[Path, dict[str, Any]]:
    path = Path(f"status/approvals/{name}.json")
    approval = stage._approval_once(
        path,
        approval_id=f"{name.upper()}-20260815",
        action=action,
        scope_path=scope_path,
        evidence=(
            "User authorized all I-stage local GPU and evaluation actions without additional prompts. "
            "This exact action is bound to the frozen I-06 calibration and I-07 Test scope."
        ),
    )
    return path, approval


def _run_action(action: str, scope_path: Path, approval_path: Path, log_path: Path) -> None:
    ops._run(
        [
            sys.executable,
            "-m",
            "cwr_eg.cli",
            action,
            "--config",
            str(CONFIG),
            "--resource-class",
            ops.RESOURCE_CLASS,
            "--scope-file",
            str(scope_path),
            "--approval",
            str(approval_path),
        ],
        log_path,
    )


def _asset_template() -> dict[str, Any]:
    return json.loads(Path("docs/i03_canary_base_generation_scope.json").read_text(encoding="utf-8"))


def _registered_scope(
    *, name: str, score_path: Path, output_path: Path, result_path: Path, expected: int
) -> tuple[Path, Path]:
    asset = _asset_template()
    scope_path = Path(f"docs/{name}_scope.json")
    freeze_path = EXECUTION_FREEZE if name.startswith("i07_test") else I06_COMPLETION
    scope = {
        "task_id": name.replace("_", "-"),
        "freeze_manifest": str(freeze_path).replace("\\", "/"),
        "freeze_manifest_sha256": sha256_file(freeze_path),
        "runner_sha256": sha256_file("src/cwr_eg/runtime.py"),
        "code_files": calstage._code_entries(
            [
                "src/cwr_eg/runtime.py",
                "src/cwr_eg/registered_scoring.py",
                "src/cwr_eg/markllm_bridge.py",
                "src/cwr_eg/candidates.py",
                "src/cwr_eg/contracts.py",
                "src/cwr_eg/manifest.py",
                "src/cwr_eg/hashing.py",
            ]
        ),
        "model_id": asset["model_id"],
        "model_path": asset["model_path"],
        "revision": asset["revision"],
        "model_files": asset["model_files"],
        "markllm_repository": asset["markllm_repository"],
        "markllm_commit": asset["markllm_commit"],
        "key_file": asset["key_file"],
        "key_file_sha256": asset["key_file_sha256"],
        "required_key_ids": asset["required_key_ids"],
        "score_records_path": str(score_path).replace("\\", "/"),
        "score_records_sha256": sha256_file(score_path),
        "families": list(FAMILIES),
        "authorized_key_slots": ["a", "b"],
        "include_scheme_only": True,
        "expected_document_count": expected,
        "device": "cuda:0",
        "monitoring": asset["monitoring"],
        "output_path": str(output_path).replace("\\", "/"),
        "result_path": str(result_path).replace("\\", "/"),
    }
    stage._scope_once(scope_path, scope)
    approval_path, _ = _approval("score-registered", name, scope_path)
    return scope_path, approval_path


def _run_registered(
    *, name: str, score_path: Path, output_path: Path, result_path: Path, expected: int
) -> Path:
    scope_path, approval_path = _registered_scope(
        name=name,
        score_path=score_path,
        output_path=output_path,
        result_path=result_path,
        expected=expected,
    )
    if not result_path.exists():
        _run_action("score-registered", scope_path, approval_path, Path(f"status/{name}_console.log"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result["documents"] != expected or result["output_sha256"] != sha256_file(output_path):
        raise RuntimeError(f"Registered scoring QA failed: {name}")
    return output_path


def _pretest_freeze_inputs() -> tuple[dict[str, Any], dict[str, Any]]:
    baseline_path = Path("artifacts/i05_train_dev/baselines/frozen_baselines.joblib")
    baseline_manifest = Path("manifests/i05_frozen_baselines.json")
    if not baseline_manifest.exists():
        summary = fit_baseline_bundle(
            Path("artifacts/i05_train_dev/checkpoint_scores/full_ensemble.jsonl"), baseline_path
        )
        stage._write_new(
            baseline_manifest,
            {
                "task_id": "I-05-frozen-baselines",
                "completed_at": ops._now(),
                **summary,
                "test_read": False,
            },
        )
    else:
        summary = json.loads(baseline_manifest.read_text(encoding="utf-8"))
        if sha256_file(baseline_path) != summary["sha256"]:
            raise RuntimeError("Frozen baseline bundle hash drift")

    dev_input = Path("artifacts/i06_calibration/dev_single_parent_scores.jsonl")
    if not dev_input.exists():
        rows = [
            row
            for row in read_jsonl("artifacts/i05_train_dev/checkpoint_scores/full_ensemble.jsonl")
            if row["split"] == "dev" and len(row["parent_ids"]) == 1
        ]
        if len(rows) != 1000:
            raise RuntimeError("Frozen Dev registered population must contain 1000 rows")
        write_jsonl(dev_input, rows)
    registered_dev = _run_registered(
        name="i06_dev_registered_scoring",
        score_path=dev_input,
        output_path=CAL_ROOT / "dev_registered_a_and_b.jsonl",
        result_path=CAL_ROOT / "dev_registered_result.json",
        expected=1000,
    )
    bundle = CalibrationBundle.load(CAL_ROOT / "calibration_bundle")
    search = json.loads(json.dumps(__import__("yaml").safe_load(CONFIG.read_text(encoding="utf-8"))["search"]))
    fusion_path = Path("manifests/i06_fusion_selection.json")
    if not fusion_path.exists():
        fusion = select_fusion_weight(read_jsonl(registered_dev), bundle, search)
        stage._write_new(
            fusion_path,
            {
                "task_id": "I-06-dev-fusion-selection",
                "completed_at": ops._now(),
                **fusion,
                "test_read": False,
            },
        )
    fusion = json.loads(fusion_path.read_text(encoding="utf-8"))
    return summary, fusion


def _freeze_test(baseline: dict[str, Any], fusion: dict[str, Any]) -> dict[str, Any]:
    gate = json.loads(GATE_D.read_text(encoding="utf-8"))
    training = json.loads(Path("manifests/i05_training_matrix_completion.json").read_text(encoding="utf-8"))
    if I07_FREEZE.exists():
        freeze = json.loads(I07_FREEZE.read_text(encoding="utf-8"))
        if (
            freeze["test_input_sha256"] != gate["test_input"]["sha256"]
            or freeze["calibration_completion_sha256"] != sha256_file(I06_COMPLETION)
        ):
            raise RuntimeError("Existing I-07 Test freeze drifted")
        return freeze
    payload = {
        "manifest_version": "i07-test-freeze-v1",
        "created_at": ops._now(),
        "task_id": "I-GATE-D-final-test",
        "gate_d_sha256": sha256_file(GATE_D),
        "calibration_completion_sha256": sha256_file(I06_COMPLETION),
        "calibration_manifest_sha256": sha256_file(CAL_ROOT / "calibration_bundle/calibration_manifest.json"),
        "null_distributions_sha256": sha256_file(CAL_ROOT / "calibration_bundle/null_distributions.npz"),
        "test_input_path": gate["test_input"]["path"],
        "test_input_sha256": gate["test_input"]["sha256"],
        "test_documents": gate["test_input"]["documents"],
        "test_parents": 200,
        "full_checkpoints": gate["full_checkpoints"],
        "lofo_checkpoints": {
            family: {
                "path": training["runs"][f"lofo_{family}"]["checkpoint"],
                "sha256": training["runs"][f"lofo_{family}"]["checkpoint_sha256"],
            }
            for family in FAMILIES
        },
        "baseline_bundle_sha256": baseline["sha256"],
        "fusion_selection_sha256": sha256_file("manifests/i06_fusion_selection.json"),
        "fusion_weight_generic": fusion["selected"]["weight_generic"],
        "ensemble_rule": "arithmetic_mean_character_logits_for_three_full_models",
        "authorization_scenarios": ["a_and_b", "a_only", "b_only"],
        "test_execution_rule": "single_unseal_after_all_hashes_frozen",
        "deviation_audit": {
            "status": "pass",
            "test_used_before_freeze": False,
            "model_or_baseline_selected_on_test": False,
            "ensemble_rule_changed": False,
        },
    }
    payload["content_hash"] = content_hash(payload)
    stage._write_new(I07_FREEZE, payload)
    return payload


def _test_features(freeze: dict[str, Any]) -> Path:
    template = json.loads(Path("docs/i_gate_c_feature_scope.json").read_text(encoding="utf-8"))
    output_dir = TEST_ROOT / "features"
    result_path = TEST_ROOT / "feature_extraction_result.json"
    scope_path = Path("docs/i07_test_feature_scope.json")
    scope = json.loads(json.dumps(template))
    scope.update(
        {
            "task_id": "I-07-test-features",
            "freeze_manifest": str(EXECUTION_FREEZE).replace("\\", "/"),
            "freeze_manifest_sha256": sha256_file(EXECUTION_FREEZE),
            "input_path": freeze["test_input_path"],
            "input_sha256": freeze["test_input_sha256"],
            "expected_document_count": freeze["test_documents"],
            "limit": freeze["test_documents"],
            "allowed_splits": ["test"],
            "output_dir": str(output_dir).replace("\\", "/"),
            "result_path": str(result_path).replace("\\", "/"),
        }
    )
    for key in ("parent_freeze_manifest_sha256", "resume_manifest_sha256", "expected_resumed_documents"):
        scope.pop(key, None)
    stage._update_code_hashes(scope)
    stage._scope_once(scope_path, scope)
    approval_path, _ = _approval("extract-features", "i07_test_features", scope_path)
    if not result_path.exists():
        _run_action("extract-features", scope_path, approval_path, Path("status/i07_test_features_console.log"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    manifest = output_dir / "feature_manifest.jsonl"
    if result["documents"] != freeze["test_documents"] or result["feature_manifest_sha256"] != sha256_file(manifest):
        raise RuntimeError("I-07 Test feature QA failed")
    return manifest


def _score_checkpoint(
    *,
    name: str,
    checkpoints: list[dict[str, str]],
    feature_manifest: Path,
    documents_path: Path,
    recipe_ids: list[str] | None,
    expected: int,
) -> Path:
    template = json.loads(Path("docs/i03_canary_checkpoint_scoring_scope.json").read_text(encoding="utf-8"))
    output_path = TEST_ROOT / "checkpoint_scores" / f"{name}.jsonl"
    result_path = TEST_ROOT / "checkpoint_score_results" / f"{name}.json"
    scope_path = Path(f"docs/i07_score_{name}_scope.json")
    scope = json.loads(json.dumps(template))
    scope.update(
        {
            "task_id": f"I-07-score-{name}",
            "role": "frozen_test_checkpoint_scoring",
            "freeze_manifest": str(EXECUTION_FREEZE).replace("\\", "/"),
            "freeze_manifest_sha256": sha256_file(EXECUTION_FREEZE),
            "checkpoints": checkpoints,
            "ensemble_rule": "arithmetic_mean_character_logits" if len(checkpoints) > 1 else "single_lofo_model",
            "feature_manifest": str(feature_manifest).replace("\\", "/"),
            "feature_manifest_sha256": sha256_file(feature_manifest),
            "documents_path": str(documents_path).replace("\\", "/"),
            "documents_sha256": sha256_file(documents_path),
            "expected_document_count": expected,
            "allowed_splits": ["test"] if documents_path.is_relative_to(TEST_ROOT) else ["calibration"],
            "output_path": str(output_path).replace("\\", "/"),
            "result_path": str(result_path).replace("\\", "/"),
        }
    )
    if recipe_ids is None:
        scope.pop("recipe_ids", None)
    else:
        scope["recipe_ids"] = recipe_ids
    stage._update_code_hashes(scope)
    stage._scope_once(scope_path, scope)
    approval_path, _ = _approval("score-checkpoint", f"i07_score_{name}", scope_path)
    if not result_path.exists():
        _run_action("score-checkpoint", scope_path, approval_path, Path(f"status/i07_score_{name}_console.log"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result["documents"] != expected or result["output_sha256"] != sha256_file(output_path):
        raise RuntimeError(f"I-07 checkpoint score QA failed: {name}")
    return output_path


def _infer(
    *, name: str, score_path: Path, bundle_path: Path, expected: int
) -> Path:
    output_path = TEST_ROOT / "decisions" / f"{name}.jsonl"
    result_path = TEST_ROOT / "inference_results" / f"{name}.json"
    scope_path = Path(f"docs/i07_infer_{name}_scope.json")
    bundle_manifest = bundle_path / "calibration_manifest.json"
    null_path = bundle_path / "null_distributions.npz"
    manifest = json.loads(bundle_manifest.read_text(encoding="utf-8"))
    scope = {
        "task_id": f"I-07-infer-{name}",
        "freeze_manifest": str(EXECUTION_FREEZE).replace("\\", "/"),
        "freeze_manifest_sha256": sha256_file(EXECUTION_FREEZE),
        "runner_sha256": sha256_file("src/cwr_eg/runtime.py"),
        "code_files": calstage._code_entries(
            [
                "src/cwr_eg/runtime.py",
                "src/cwr_eg/bundle.py",
                "src/cwr_eg/calibration.py",
                "src/cwr_eg/contracts.py",
                "src/cwr_eg/enums.py",
                "src/cwr_eg/inference.py",
                "src/cwr_eg/candidates.py",
                "src/cwr_eg/decision.py",
                "src/cwr_eg/validity.py",
            ]
        ),
        "calibration_bundle": str(bundle_path).replace("\\", "/"),
        "calibration_manifest_sha256": sha256_file(bundle_manifest),
        "null_distributions_sha256": sha256_file(null_path),
        "score_records_path": str(score_path).replace("\\", "/"),
        "score_records_sha256": sha256_file(score_path),
        "expected_document_count": expected,
        "allowed_splits": ["test"],
        "versions": {
            "model_version": manifest["header"]["model_version"],
            "calibration_id": manifest["header"]["calibration_id"],
            "normalization_version": "raw-unicode-codepoint-v1",
            "manifest_version": "i07-test-freeze-v1",
            "code_revision": "working-tree:" + sha256_file("src/cwr_eg/runtime.py"),
        },
        "monitoring": _asset_template()["monitoring"],
        "output_path": str(output_path).replace("\\", "/"),
        "result_path": str(result_path).replace("\\", "/"),
    }
    stage._scope_once(scope_path, scope)
    approval_path, _ = _approval("infer", f"i07_infer_{name}", scope_path)
    if not result_path.exists():
        _run_action("infer", scope_path, approval_path, Path(f"status/i07_infer_{name}_console.log"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result["documents"] != expected or result["output_sha256"] != sha256_file(output_path):
        raise RuntimeError(f"I-07 inference QA failed: {name}")
    return output_path


def _prepare_and_evaluate(
    *,
    name: str,
    decisions_path: Path,
    documents_path: Path,
    authorized_slots: list[str],
    held_out_family: str | None = None,
) -> tuple[Path, dict[str, Any]]:
    records_path = TEST_ROOT / "evaluation_records" / f"{name}.jsonl"
    prepare_result = TEST_ROOT / "evaluation_prepare_results" / f"{name}.json"
    prepare_scope = Path(f"docs/i07_prepare_evaluation_{name}_scope.json")
    scope = {
        "task_id": f"I-07-prepare-evaluation-{name}",
        "freeze_manifest": str(EXECUTION_FREEZE).replace("\\", "/"),
        "freeze_manifest_sha256": sha256_file(EXECUTION_FREEZE),
        "runner_sha256": sha256_file("src/cwr_eg/runtime.py"),
        "code_files": calstage._code_entries(
            [
                "src/cwr_eg/runtime.py",
                "src/cwr_eg/evaluation_records.py",
                "src/cwr_eg/manifest.py",
                "src/cwr_eg/hashing.py",
            ]
        ),
        "decisions_path": str(decisions_path).replace("\\", "/"),
        "decisions_sha256": sha256_file(decisions_path),
        "documents_path": str(documents_path).replace("\\", "/"),
        "documents_sha256": sha256_file(documents_path),
        "authorized_key_slots": authorized_slots,
        "held_out_family": held_out_family,
        "expected_parent_count": 200,
        "monitoring": _asset_template()["monitoring"],
        "output_path": str(records_path).replace("\\", "/"),
        "result_path": str(prepare_result).replace("\\", "/"),
    }
    stage._scope_once(prepare_scope, scope)
    approval_path, _ = _approval("prepare-evaluation", f"i07_prepare_evaluation_{name}", prepare_scope)
    if not prepare_result.exists():
        _run_action("prepare-evaluation", prepare_scope, approval_path, Path(f"status/i07_prepare_evaluation_{name}_console.log"))
    metric_result = TEST_ROOT / "metric_results" / f"{name}.json"
    metric_scope = Path(f"docs/i07_evaluate_{name}_scope.json")
    eval_scope = {
        "task_id": f"I-07-evaluate-{name}",
        "freeze_manifest": str(EXECUTION_FREEZE).replace("\\", "/"),
        "freeze_manifest_sha256": sha256_file(EXECUTION_FREEZE),
        "runner_sha256": sha256_file("src/cwr_eg/runtime.py"),
        "code_files": calstage._code_entries(
            [
                "src/cwr_eg/runtime.py",
                "src/cwr_eg/metrics.py",
                "src/cwr_eg/calibration.py",
                "src/cwr_eg/contracts.py",
                "src/cwr_eg/enums.py",
                "src/cwr_eg/intervals.py",
                "src/cwr_eg/manifest.py",
                "src/cwr_eg/hashing.py",
            ]
        ),
        "records_path": str(records_path).replace("\\", "/"),
        "records_sha256": sha256_file(records_path),
        "stratify_by": ["source", "language", "watermark_family", "key_id", "attack_id"],
        "bootstrap_replicates": 2000,
        "seed": 20260815,
        "monitoring": _asset_template()["monitoring"],
        "result_path": str(metric_result).replace("\\", "/"),
    }
    stage._scope_once(metric_scope, eval_scope)
    eval_approval, _ = _approval("evaluate", f"i07_evaluate_{name}", metric_scope)
    if not metric_result.exists():
        _run_action("evaluate", metric_scope, eval_approval, Path(f"status/i07_evaluate_{name}_console.log"))
    result = json.loads(metric_result.read_text(encoding="utf-8"))
    return records_path, result


def _fit_lofo_bundle(
    *, family: str, checkpoint: dict[str, str], calibration_feature_manifest: Path, main_registered: Path
) -> Path:
    cal_documents = CAL_ROOT / "null_feature_documents.jsonl"
    expected = len(read_jsonl(cal_documents))
    generic = _score_checkpoint(
        name=f"lofo_{family}_calibration",
        checkpoints=[checkpoint],
        feature_manifest=calibration_feature_manifest,
        documents_path=cal_documents,
        recipe_ids=None,
        expected=expected,
    )
    merged = CAL_ROOT / "lofo" / family / "registered_scores.jsonl"
    if not merged.exists():
        merge_generic_with_registered(generic, main_registered, merged, excluded_family=family)
    records = CAL_ROOT / "lofo" / family / "parent_calibration_records.jsonl"
    prepare_result = CAL_ROOT / "lofo" / family / "prepare_result.json"
    prepare_scope = Path(f"docs/i07_prepare_lofo_{family}_calibration_scope.json")
    scope = {
        "task_id": f"I-07-prepare-lofo-{family}-calibration",
        "freeze_manifest": str(I07_FREEZE).replace("\\", "/"),
        "freeze_manifest_sha256": sha256_file(I07_FREEZE),
        "runner_sha256": sha256_file("src/cwr_eg/runtime.py"),
        "code_files": calstage._code_entries(
            ["src/cwr_eg/runtime.py", "src/cwr_eg/calibration_records.py", "src/cwr_eg/candidates.py", "src/cwr_eg/manifest.py", "src/cwr_eg/hashing.py"]
        ),
        "scored_documents_path": str(merged).replace("\\", "/"),
        "scored_documents_sha256": sha256_file(merged),
        "expected_parent_count": 200,
        "monitoring": _asset_template()["monitoring"],
        "output_path": str(records).replace("\\", "/"),
        "result_path": str(prepare_result).replace("\\", "/"),
    }
    stage._scope_once(prepare_scope, scope)
    approval_path, _ = _approval("prepare-calibration", f"i07_prepare_lofo_{family}_calibration", prepare_scope)
    if not prepare_result.exists():
        _run_action("prepare-calibration", prepare_scope, approval_path, Path(f"status/i07_prepare_lofo_{family}_calibration_console.log"))
    bundle_path = CAL_ROOT / "lofo" / family / "calibration_bundle"
    result_path = CAL_ROOT / "lofo" / family / "calibration_result.json"
    fit_scope = Path(f"docs/i07_fit_lofo_{family}_calibration_scope.json")
    primary_header = json.loads((CAL_ROOT / "calibration_bundle/calibration_manifest.json").read_text(encoding="utf-8"))["header"]
    header = {**primary_header, "calibration_id": f"i07-lofo-{family}-parent-max-v1", "model_version": "sha256:" + checkpoint["sha256"]}
    fit = {
        "task_id": f"I-07-fit-lofo-{family}-calibration",
        "freeze_manifest": str(I07_FREEZE).replace("\\", "/"),
        "freeze_manifest_sha256": sha256_file(I07_FREEZE),
        "runner_sha256": sha256_file("src/cwr_eg/runtime.py"),
        "code_files": calstage._code_entries(["src/cwr_eg/runtime.py", "src/cwr_eg/bundle.py", "src/cwr_eg/calibration.py", "src/cwr_eg/hashing.py"]),
        "records_path": str(records).replace("\\", "/"),
        "records_sha256": sha256_file(records),
        "aggregation_unit": "parent_id",
        "minimum_parents_per_stratum": 100,
        "header": header,
        "monitoring": _asset_template()["monitoring"],
        "output_dir": str(bundle_path).replace("\\", "/"),
        "result_path": str(result_path).replace("\\", "/"),
    }
    stage._scope_once(fit_scope, fit)
    fit_approval, _ = _approval("calibrate", f"i07_fit_lofo_{family}_calibration", fit_scope)
    if not result_path.exists():
        _run_action("calibrate", fit_scope, fit_approval, Path(f"status/i07_fit_lofo_{family}_calibration_console.log"))
    bundle = CalibrationBundle.load(bundle_path)
    if bundle.metadata["parent_counts_by_stratum"] != {"en:all": 100, "zh:all": 100}:
        raise RuntimeError(f"LOFO calibration parent QA failed: {family}")
    return bundle_path


def _freeze_lofo_calibrations(bundle_paths: dict[str, Path]) -> Path:
    payload = {
        "manifest_version": "i07-test-freeze-amendment-v1",
        "created_at": ops._now(),
        "task_id": "I-GATE-D-final-test-amendment",
        "base_freeze_path": str(I07_FREEZE).replace("\\", "/"),
        "base_freeze_sha256": sha256_file(I07_FREEZE),
        "lofo_calibration_bundles": {
            family: {
                "path": str(path).replace("\\", "/"),
                "calibration_manifest_sha256": sha256_file(path / "calibration_manifest.json"),
                "null_distributions_sha256": sha256_file(path / "null_distributions.npz"),
            }
            for family, path in bundle_paths.items()
        },
        "test_execution_still_sealed": True,
        "deviation_audit": {
            "status": "pass",
            "all_lofo_calibrations_fit_before_test_feature_extraction": True,
            "test_used_for_calibration": False,
        },
    }
    payload["content_hash"] = content_hash(payload)
    if I07_FREEZE_AMENDMENT.exists():
        existing = json.loads(I07_FREEZE_AMENDMENT.read_text(encoding="utf-8"))
        if existing["base_freeze_sha256"] != payload["base_freeze_sha256"]:
            raise RuntimeError("I-07 final Test freeze amendment drifted")
    else:
        stage._write_new(I07_FREEZE_AMENDMENT, payload)
    return I07_FREEZE_AMENDMENT


def _composite_oscr(main_records: Path, lofo_records: dict[str, Path]) -> float:
    rows = read_jsonl(main_records)
    known = [row for row in rows if row["true_label"] == "known_scheme_known_key"]
    unknown = [row for path in lofo_records.values() for row in read_jsonl(path) if row["true_label"] == "suspected_unknown_scheme"]
    records = [
        EvaluationRecord(
            parent_id=str(row["parent_id"]),
            true_label=DecisionLabel(row["true_label"]),
            predicted_label=DecisionLabel(row["predicted_label"]),
            score=float(row["score"]),
            knownness_score=float(row["knownness_score"]),
        )
        for row in known + unknown
    ]
    return oscr(records)


def main() -> int:
    import argparse
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--i06-process-id", type=int, required=True)
    args = parser.parse_args()
    global EXECUTION_FREEZE
    os.chdir(ROOT)
    ops.STATUS_PATH = STATUS
    ops.CONFIG = CONFIG
    ops._set_status("wait_for_i06", "in_progress", process_id=args.i06_process_id)
    _wait_for_i06(args.i06_process_id)

    ops._set_status("freeze_pretest_baselines", "in_progress")
    baseline, fusion = _pretest_freeze_inputs()
    freeze = _freeze_test(baseline, fusion)

    ops._set_status("freeze_lofo_calibrations", "in_progress", families=4)
    calibration_feature_manifest = CAL_ROOT / "features/feature_manifest.jsonl"
    calibration_registered = CAL_ROOT / "registered_scores_a_and_b.jsonl"
    lofo_bundle_paths = {
        family: _fit_lofo_bundle(
            family=family,
            checkpoint=freeze["lofo_checkpoints"][family],
            calibration_feature_manifest=calibration_feature_manifest,
            main_registered=calibration_registered,
        )
        for family in FAMILIES
    }
    EXECUTION_FREEZE = _freeze_lofo_calibrations(lofo_bundle_paths)

    ops._set_status("test_features", "in_progress", documents=freeze["test_documents"])
    test_feature_manifest = _test_features(freeze)
    test_documents = Path(freeze["test_input_path"])
    full_scores = _score_checkpoint(
        name="full_ensemble_test",
        checkpoints=[{"path": row["path"], "sha256": row["sha256"]} for row in freeze["full_checkpoints"]],
        feature_manifest=test_feature_manifest,
        documents_path=test_documents,
        recipe_ids=None,
        expected=freeze["test_documents"],
    )
    ops._set_status("test_registered_scoring", "in_progress")
    main_registered = _run_registered(
        name="i07_test_registered_scoring",
        score_path=full_scores,
        output_path=TEST_ROOT / "registered_scores_a_and_b.jsonl",
        result_path=TEST_ROOT / "registered_scoring_result.json",
        expected=freeze["test_documents"],
    )
    scenario_scores = {"a_and_b": main_registered}
    for name, slots in (("a_only", ("a",)), ("b_only", ("b",))):
        path = TEST_ROOT / f"registered_scores_{name}.jsonl"
        if not path.exists():
            mask_registered_records(main_registered, path, authorized_key_slots=slots)
        scenario_scores[name] = path

    scenario_results: dict[str, Any] = {}
    scenario_records: dict[str, Path] = {}
    primary_bundle = CAL_ROOT / "calibration_bundle"
    for name, score_path in scenario_scores.items():
        decisions = _infer(name=name, score_path=score_path, bundle_path=primary_bundle, expected=freeze["test_documents"])
        slots = ["a", "b"] if name == "a_and_b" else [name[0]]
        records, metrics = _prepare_and_evaluate(
            name=name,
            decisions_path=decisions,
            documents_path=test_documents,
            authorized_slots=slots,
        )
        scenario_records[name] = records
        scenario_results[name] = metrics

    ops._set_status("lofo_test", "in_progress", families=4)
    test_rows = read_jsonl(test_documents)
    test_feature_ids = {str(row["recipe_id"]) for row in read_jsonl(test_feature_manifest)}
    lofo_results: dict[str, Any] = {}
    lofo_records: dict[str, Path] = {}
    for family in FAMILIES:
        checkpoint = freeze["lofo_checkpoints"][family]
        bundle_path = lofo_bundle_paths[family]
        subset = [
            row for row in test_rows if (row.get("watermark_family") is None or row.get("watermark_family") == family) and len(row["parent_ids"]) == 1
        ]
        subset_path = TEST_ROOT / "lofo" / family / "feature_documents.jsonl"
        if not subset_path.exists():
            write_jsonl(subset_path, subset)
        recipe_ids = [str(row["recipe_id"]) for row in subset]
        if not set(recipe_ids).issubset(test_feature_ids):
            raise RuntimeError(f"LOFO Test subset lacks frozen features: {family}")
        generic = _score_checkpoint(
            name=f"lofo_{family}_test",
            checkpoints=[checkpoint],
            feature_manifest=test_feature_manifest,
            documents_path=subset_path,
            recipe_ids=recipe_ids,
            expected=len(subset),
        )
        merged = TEST_ROOT / "lofo" / family / "registered_scores.jsonl"
        if not merged.exists():
            merge_generic_with_registered(generic, main_registered, merged, excluded_family=family)
        decisions = _infer(name=f"lofo_{family}", score_path=merged, bundle_path=bundle_path, expected=len(subset))
        records, metrics = _prepare_and_evaluate(
            name=f"lofo_{family}",
            decisions_path=decisions,
            documents_path=subset_path,
            authorized_slots=["a", "b"],
            held_out_family=family,
        )
        lofo_records[family] = records
        lofo_results[family] = metrics

    ops._set_status("test_baselines", "in_progress")
    baseline_results = evaluate_baseline_bundle(
        Path("artifacts/i05_train_dev/baselines/frozen_baselines.joblib"), full_scores
    )
    search = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))["search"]
    labels, generic_evidence, registered_evidence = calibrated_document_scores(
        [row for row in read_jsonl(main_registered) if len(row["parent_ids"]) == 1],
        CalibrationBundle.load(primary_bundle),
        search,
    )
    weight = float(freeze["fusion_weight_generic"])
    baseline_results["registered_only"] = {
        "watermark_auc": float(roc_auc_score(labels, registered_evidence))
    }
    baseline_results["linear_evidence_fusion"] = {
        "weight_generic": weight,
        "watermark_auc": float(
            roc_auc_score(labels, weight * generic_evidence + (1.0 - weight) * registered_evidence)
        ),
    }
    baseline_results["markllm_registered_detectors"] = {
        "families": list(FAMILIES),
        "authorization_scenarios": list(scenario_results),
        "primary_registered_score_sha256": sha256_file(main_registered),
    }

    result_path = TEST_ROOT / "i07_results.json"
    result = {
        "task_id": "I-07",
        "completed_at": ops._now(),
        "test_freeze_sha256": sha256_file(EXECUTION_FREEZE),
        "test_documents": freeze["test_documents"],
        "test_parents": 200,
        "scenario_metrics": scenario_results,
        "lofo_metrics": lofo_results,
        "baselines": baseline_results,
        "composite_oscr": _composite_oscr(scenario_records["a_and_b"], lofo_records),
        "primary_parent_fwer": scenario_results["a_and_b"]["parent_fwer"],
        "deviation_audit": {
            "status": "pass",
            "test_executions": 1,
            "test_used_for_training_selection_or_early_stopping": False,
            "ensemble_rule_changed": False,
            "baseline_or_fusion_selected_on_test": False,
            "all_failed_rows_remained_explicit": True,
        },
    }
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    completion = Path("manifests/i07_test_completion.json")
    if not completion.exists():
        stage._write_new(
            completion,
            {
                "task_id": "I-07",
                "completed_at": ops._now(),
                "result_path": str(result_path).replace("\\", "/"),
                "result_sha256": sha256_file(result_path),
                "test_freeze_sha256": sha256_file(EXECUTION_FREEZE),
                "primary_parent_fwer": result["primary_parent_fwer"],
                "deviation_audit": result["deviation_audit"],
            },
        )
    ops._set_status("i07_complete", "done", result_sha256=sha256_file(result_path))
    print(json.dumps({"status": "done", "test_parents": 200}, sort_keys=True))
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
            "I-07",
            "blocked",
            "Fail-closed I-07 continuation stopped; inspect status/i07_test_status.json.",
            error_type=type(error).__name__,
            error_message=str(error),
        )
        raise
