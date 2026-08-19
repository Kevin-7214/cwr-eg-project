from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Any

import psutil

import continue_i04_after_base as ops
import continue_i05_after_features as stage
from cwr_eg.bundle import CalibrationBundle
from cwr_eg.hashing import content_hash, sha256_file
from cwr_eg.manifest import read_jsonl, write_jsonl


ROOT = Path(__file__).resolve().parents[1]
CONFIG = Path("configs/intermediate.yaml")
I04_DOCUMENTS = Path("artifacts/i04_full/feature_documents.jsonl")
I05_COMPLETION = Path("manifests/i05_dev_analysis_completion.json")
GATE_FREEZE = Path("manifests/i_gate_d_freeze.json")
CAL_ROOT = Path("artifacts/i06_calibration")
TEST_ROOT = Path("artifacts/i07_test")
STATUS = Path("status/i_gate_d_calibration_status.json")


def _wait_for_i05(process_id: int) -> None:
    deadline = time.monotonic() + 24 * 60 * 60
    while not I05_COMPLETION.exists():
        if not psutil.pid_exists(process_id):
            raise RuntimeError("I-05 post-training process exited without Dev analysis completion")
        if time.monotonic() > deadline:
            raise TimeoutError("I-05 post-training continuation exceeded twenty-four hours")
        time.sleep(30)


def _write_split_inputs() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = read_jsonl(I04_DOCUMENTS)
    calibration = [
        row
        for row in rows
        if row["split"] == "calibration"
        and row.get("watermark_family") is None
        and len(row["parent_ids"]) == 1
    ]
    test = [row for row in rows if row["split"] == "test"]
    calibration_parents = {str(row["parent_ids"][0]) for row in calibration}
    test_parents = {str(parent) for row in test for parent in row["parent_ids"]}
    language_parents = {
        language: {
            str(row["parent_ids"][0])
            for row in calibration
            if row["language"] == language
        }
        for language in ("en", "zh")
    }
    if len(calibration_parents) != 200 or len(test_parents) != 200:
        raise RuntimeError("I-GATE-D parent population differs from the frozen split")
    if {language: len(parents) for language, parents in language_parents.items()} != {"en": 100, "zh": 100}:
        raise RuntimeError("Calibration language strata are not 100 independent parents each")
    calibration_path = CAL_ROOT / "null_feature_documents.jsonl"
    test_path = TEST_ROOT / "feature_documents.jsonl"
    if calibration_path.exists() or test_path.exists():
        existing_cal = read_jsonl(calibration_path)
        existing_test = read_jsonl(test_path)
        if existing_cal != calibration or existing_test != test:
            raise RuntimeError("Existing I-GATE-D split input drifted")
    else:
        write_jsonl(calibration_path, calibration)
        write_jsonl(test_path, test)
    return calibration, test


def _training_runs() -> dict[str, Any]:
    path = Path("manifests/i05_training_matrix_completion.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    runs = payload["runs"]
    for run_id, row in runs.items():
        if sha256_file(row["checkpoint"]) != row["checkpoint_sha256"]:
            raise RuntimeError(f"Checkpoint drift at I-GATE-D: {run_id}")
    return runs


def _freeze(calibration: list[dict[str, Any]], test: list[dict[str, Any]], runs: dict[str, Any]) -> dict[str, Any]:
    calibration_path = CAL_ROOT / "null_feature_documents.jsonl"
    test_path = TEST_ROOT / "feature_documents.jsonl"
    if GATE_FREEZE.exists():
        existing = json.loads(GATE_FREEZE.read_text(encoding="utf-8"))
        if (
            existing["calibration_null_input"]["sha256"] != sha256_file(calibration_path)
            or existing["test_input"]["sha256"] != sha256_file(test_path)
            or existing["training_completion"]["sha256"]
            != sha256_file("manifests/i05_training_matrix_completion.json")
        ):
            raise RuntimeError("Existing I-GATE-D freeze drifted")
        return existing
    payload = {
        "manifest_version": "i-gate-d-v1",
        "created_at": ops._now(),
        "task_id": "I-GATE-D",
        "i05_dev_analysis": {
            "path": str(I05_COMPLETION).replace("\\", "/"),
            "sha256": sha256_file(I05_COMPLETION),
        },
        "training_completion": {
            "path": "manifests/i05_training_matrix_completion.json",
            "sha256": sha256_file("manifests/i05_training_matrix_completion.json"),
        },
        "full_checkpoints": [
            {
                "run_id": run_id,
                "path": runs[run_id]["checkpoint"],
                "sha256": runs[run_id]["checkpoint_sha256"],
            }
            for run_id in ("full_seed_20260815", "full_seed_20260816", "full_seed_20260817")
        ],
        "calibration_null_input": {
            "path": str(calibration_path).replace("\\", "/"),
            "sha256": sha256_file(calibration_path),
            "documents": len(calibration),
            "parents": 200,
            "parent_counts_by_language": {"en": 100, "zh": 100},
            "descendants": ["clean", "attacked-clean"],
        },
        "test_input": {
            "path": str(test_path).replace("\\", "/"),
            "sha256": sha256_file(test_path),
            "documents": len(test),
            "parents": 200,
            "execution_sealed": True,
        },
        "ensemble_rule": "arithmetic_mean_character_logits_for_three_full_models",
        "calibration_action_approved_next": True,
        "test_action_approval_deferred_until_i06_frozen": True,
        "deviation_audit": {
            "status": "pass",
            "sample_or_split_changed": False,
            "test_content_used_for_selection": False,
            "test_features_or_scores_computed": False,
            "calibration_parents_per_language": 100,
        },
    }
    payload["content_hash"] = content_hash(payload)
    stage._write_new(GATE_FREEZE, payload)
    return payload


def _approval(action: str, name: str, scope_path: Path) -> tuple[Path, dict[str, Any]]:
    path = Path(f"status/approvals/{name}.json")
    approval = stage._approval_once(
        path,
        approval_id=f"{name.upper()}-20260815",
        action=action,
        scope_path=scope_path,
        evidence=(
            "User authorized all I-stage local GPU actions without additional prompts. "
            "This exact I-GATE-D scope is hash-bound; Test execution remains sealed until I-06 completes."
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


def _calibration_features(freeze: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    template = json.loads(Path("docs/i_gate_c_feature_scope.json").read_text(encoding="utf-8"))
    input_path = Path(freeze["calibration_null_input"]["path"])
    output_dir = CAL_ROOT / "features"
    result_path = CAL_ROOT / "feature_extraction_result.json"
    scope_path = Path("docs/i06_calibration_feature_scope_v2.json")
    scope = json.loads(json.dumps(template))
    scope.update(
        {
            "task_id": "I-06-calibration-features",
            "freeze_manifest": str(GATE_FREEZE).replace("\\", "/"),
            "freeze_manifest_sha256": sha256_file(GATE_FREEZE),
            "input_path": str(input_path).replace("\\", "/"),
            "input_sha256": sha256_file(input_path),
            "expected_document_count": len(read_jsonl(input_path)),
            "limit": len(read_jsonl(input_path)),
            "allowed_splits": ["calibration"],
            "output_dir": str(output_dir).replace("\\", "/"),
            "result_path": str(result_path).replace("\\", "/"),
        }
    )
    for key in ("parent_freeze_manifest_sha256", "resume_manifest_sha256", "expected_resumed_documents"):
        scope.pop(key, None)
    stage._update_code_hashes(scope)
    stage._scope_once(scope_path, scope)
    approval_path, approval = _approval("extract-features", "i06_calibration_features_v2", scope_path)
    _record_feature_scope_correction(scope_path, approval_path)
    if not result_path.exists():
        _run_action("extract-features", scope_path, approval_path, Path("status/i06_calibration_features_console.log"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    manifest = output_dir / "feature_manifest.jsonl"
    if result["documents"] != scope["expected_document_count"] or result["feature_manifest_sha256"] != sha256_file(manifest):
        raise RuntimeError("Calibration feature extraction QA failed")
    return manifest, approval


def _record_feature_scope_correction(scope_path: Path, approval_path: Path) -> None:
    correction_path = Path("manifests/i06_calibration_feature_scope_correction_01.json")
    old_scope = Path("docs/i06_calibration_feature_scope.json")
    old_approval = Path("status/approvals/i06_calibration_features.json")
    payload = {
        "manifest_version": "i06-scope-correction-v1",
        "created_at": ops._now(),
        "task_id": "I-06-calibration-features-scope-correction",
        "trigger": {
            "error": "Feature resumed-document count does not match the approved scope",
            "stale_expected_resumed_documents": 810,
            "actual_calibration_documents": 400,
        },
        "root_cause": (
            "The I-06 Calibration feature scope was cloned from the I-GATE-C Train/Dev "
            "feature scope but did not remove expected_resumed_documents=810. "
            "The I-07 Test feature path already removes that stale template field."
        ),
        "correction": {
            "old_scope": str(old_scope).replace("\\", "/"),
            "old_scope_sha256": sha256_file(old_scope),
            "old_approval": str(old_approval).replace("\\", "/"),
            "old_approval_sha256": sha256_file(old_approval),
            "new_scope": str(scope_path).replace("\\", "/"),
            "new_scope_sha256": sha256_file(scope_path),
            "new_approval": str(approval_path).replace("\\", "/"),
            "new_approval_sha256": sha256_file(approval_path),
            "removed_field": "expected_resumed_documents",
            "expected_document_count": 400,
        },
        "isolation_audit": {
            "calibration_input_changed": False,
            "calibration_parent_population_changed": False,
            "train_or_dev_changed": False,
            "test_input_changed": False,
            "test_features_or_scores_computed": False,
            "test_remains_sealed": True,
            "model_or_checkpoint_selection_changed": False,
        },
    }
    payload["content_hash"] = content_hash(payload)
    if correction_path.exists():
        existing = json.loads(correction_path.read_text(encoding="utf-8"))
        for key in ("trigger", "root_cause", "correction", "isolation_audit"):
            if existing.get(key) != payload.get(key):
                raise RuntimeError("Existing I-06 feature-scope correction attestation drifted")
        return
    stage._write_new(correction_path, payload)


def _calibration_checkpoint_scores(freeze: dict[str, Any], feature_manifest: Path) -> Path:
    template = json.loads(Path("docs/i03_canary_checkpoint_scoring_scope.json").read_text(encoding="utf-8"))
    output_path = CAL_ROOT / "checkpoint_scores.jsonl"
    result_path = CAL_ROOT / "checkpoint_scoring_result.json"
    scope_path = Path("docs/i06_calibration_checkpoint_scoring_scope.json")
    scope = json.loads(json.dumps(template))
    scope.update(
        {
            "task_id": "I-06-calibration-checkpoint-scoring",
            "role": "parent_null_calibration_ensemble",
            "freeze_manifest": str(GATE_FREEZE).replace("\\", "/"),
            "freeze_manifest_sha256": sha256_file(GATE_FREEZE),
            "checkpoints": [
                {"path": row["path"], "sha256": row["sha256"]}
                for row in freeze["full_checkpoints"]
            ],
            "ensemble_rule": freeze["ensemble_rule"],
            "feature_manifest": str(feature_manifest).replace("\\", "/"),
            "feature_manifest_sha256": sha256_file(feature_manifest),
            "documents_path": freeze["calibration_null_input"]["path"],
            "documents_sha256": freeze["calibration_null_input"]["sha256"],
            "expected_document_count": freeze["calibration_null_input"]["documents"],
            "allowed_splits": ["calibration"],
            "output_path": str(output_path).replace("\\", "/"),
            "result_path": str(result_path).replace("\\", "/"),
        }
    )
    scope.pop("recipe_ids", None)
    stage._update_code_hashes(scope)
    stage._scope_once(scope_path, scope)
    approval_path, _ = _approval("score-checkpoint", "i06_calibration_checkpoint_scoring", scope_path)
    if not result_path.exists():
        _run_action("score-checkpoint", scope_path, approval_path, Path("status/i06_calibration_checkpoint_scoring_console.log"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result["documents"] != freeze["calibration_null_input"]["documents"] or result["output_sha256"] != sha256_file(output_path):
        raise RuntimeError("Calibration checkpoint score QA failed")
    return output_path


def _code_entries(paths: list[str]) -> list[dict[str, str]]:
    return [{"path": path, "sha256": sha256_file(path)} for path in paths]


def _registered_scores(freeze: dict[str, Any], checkpoint_scores: Path) -> Path:
    asset = json.loads(Path("docs/i03_canary_base_generation_scope.json").read_text(encoding="utf-8"))
    output_path = CAL_ROOT / "registered_scores_a_and_b.jsonl"
    result_path = CAL_ROOT / "registered_scoring_result.json"
    scope_path = Path("docs/i06_calibration_registered_scoring_scope_v3.json")
    scope = {
        "task_id": "I-06-calibration-registered-scoring",
        "freeze_manifest": str(GATE_FREEZE).replace("\\", "/"),
        "freeze_manifest_sha256": sha256_file(GATE_FREEZE),
        "runner_sha256": sha256_file("src/cwr_eg/runtime.py"),
        "code_files": _code_entries(
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
        "score_records_path": str(checkpoint_scores).replace("\\", "/"),
        "score_records_sha256": sha256_file(checkpoint_scores),
        "families": ["kgw", "unigram", "unbiased", "synthid"],
        "authorized_key_slots": ["a", "b"],
        "include_scheme_only": True,
        "expected_document_count": freeze["calibration_null_input"]["documents"],
        "device": "cuda:0",
        "monitoring": asset["monitoring"],
        "output_path": str(output_path).replace("\\", "/"),
        "result_path": str(result_path).replace("\\", "/"),
        "resume_policy": "existing-partial-exact-recipe-id-prefix",
        "short_segment_policy": "zero-token-precheck-plus-observed-kgw-min-prefix-inapplicable",
    }
    stage._scope_once(scope_path, scope)
    approval_path, _ = _approval("score-registered", "i06_calibration_registered_scoring_v3", scope_path)
    _record_registered_zero_token_correction(scope_path, approval_path, output_path)
    if not result_path.exists():
        _run_action("score-registered", scope_path, approval_path, Path("status/i06_calibration_registered_scoring_console.log"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result["documents"] != freeze["calibration_null_input"]["documents"] or result["output_sha256"] != sha256_file(output_path):
        raise RuntimeError("Calibration registered scoring QA failed")
    return output_path


def _record_registered_short_segment_correction(
    scope_path: Path, approval_path: Path, output_path: Path
) -> None:
    correction_path = Path("manifests/i06_registered_short_segment_correction_01.json")
    partial_path = output_path.with_suffix(output_path.suffix + ".partial")
    partial_rows = read_jsonl(partial_path) if partial_path.exists() else []
    old_scope = Path("docs/i06_calibration_registered_scoring_scope.json")
    old_approval = Path("status/approvals/i06_calibration_registered_scoring.json")
    payload = {
        "manifest_version": "i06-registered-short-segment-correction-v1",
        "created_at": ops._now(),
        "task_id": "I-06-registered-short-segment-correction",
        "trigger": {
            "error_type": "ValueError",
            "error_message_prefix": (
                "Must have at least 1 token to score after the first min_prefix_len="
            ),
            "observed_completed_documents_before_failure": len(partial_rows),
            "expected_documents": 400,
        },
        "root_cause": (
            "A refined candidate interval can be too short for MarkLLM KGW's "
            "min_prefix_len seeding requirement. The registered scorer previously treated "
            "this detector-level non-applicability as a fatal pipeline error."
        ),
        "correction": {
            "behavior": (
                "Catch only the observed MarkLLM min-prefix short-segment ValueError; "
                "record that exact registered detector test as applicability=invalid and "
                "omit it from registered maxima. Re-raise every other detector error."
            ),
            "old_registered_scoring_sha256": (
                "8d041447e32f4ea773f39b39c3b35f1e2882e0cff9a267307d2a1f9866ab1462"
            ),
            "new_registered_scoring_sha256": (
                "90d8963aafcdc6ce29e4f922db668fb8d01766d519246fbe1b29a0ce8e31019a"
            ),
            "old_scope": str(old_scope).replace("\\", "/"),
            "old_scope_sha256": sha256_file(old_scope),
            "old_approval": str(old_approval).replace("\\", "/"),
            "old_approval_sha256": sha256_file(old_approval),
            "new_scope": str(scope_path).replace("\\", "/"),
            "new_scope_sha256": sha256_file(scope_path),
            "new_approval": str(approval_path).replace("\\", "/"),
            "new_approval_sha256": sha256_file(approval_path),
            "partial_path": str(partial_path).replace("\\", "/"),
            "partial_sha256_at_correction": (
                sha256_file(partial_path) if partial_path.exists() else None
            ),
            "partial_documents_at_correction": len(partial_rows),
            "resume_preserves_completed_rows": True,
        },
        "isolation_audit": {
            "train_or_dev_changed": False,
            "calibration_input_changed": False,
            "checkpoint_scores_changed": False,
            "test_input_changed": False,
            "test_features_or_scores_computed": False,
            "test_remains_sealed": True,
            "model_or_checkpoint_selection_changed": False,
        },
    }
    payload["content_hash"] = content_hash(payload)
    if correction_path.exists():
        existing = json.loads(correction_path.read_text(encoding="utf-8"))
        for key in ("root_cause", "isolation_audit"):
            if existing.get(key) != payload.get(key):
                raise RuntimeError("Existing registered short-segment correction drifted")
        return
    stage._write_new(correction_path, payload)


def _record_registered_zero_token_correction(
    scope_path: Path, approval_path: Path, output_path: Path
) -> None:
    correction_path = Path("manifests/i06_registered_zero_token_correction_02.json")
    partial_path = output_path.with_suffix(output_path.suffix + ".partial")
    partial_rows = read_jsonl(partial_path) if partial_path.exists() else []
    previous_correction = Path("manifests/i06_registered_short_segment_correction_01.json")
    payload = {
        "manifest_version": "i06-registered-zero-token-correction-v1",
        "created_at": ops._now(),
        "task_id": "I-06-registered-zero-token-correction",
        "trigger": {
            "error_type": "RuntimeError",
            "error_message": (
                "cannot reshape tensor of 0 elements into shape [1, 0, -1, 128] "
                "because the unspecified dimension size -1 can be any value and is ambiguous"
            ),
            "detector_family": "unbiased",
            "observed_completed_documents_before_failure": len(partial_rows),
            "expected_documents": 400,
        },
        "root_cause": (
            "A refined character interval can contain only tokenizer-ignored/whitespace "
            "characters and therefore tokenize to zero detector tokens. Calling a MarkLLM "
            "detector on such an interval is outside detector applicability; Unbiased reached "
            "Qwen2 with sequence length zero and failed during attention projection reshape."
        ),
        "correction": {
            "behavior": (
                "Before registered detector calls, tokenize the exact candidate segment with "
                "the detector tokenizer and add_special_tokens=False. If token count is zero, "
                "record each family/key detector test for that interval as applicability=invalid "
                "with reason zero_detector_tokens and do not call MarkLLM. Preserve the prior "
                "narrow KGW min_prefix_len applicability handling. All other exceptions re-raise."
            ),
            "previous_registered_scoring_sha256": "90d8963aafcdc6ce29e4f922db668fb8d01766d519246fbe1b29a0ce8e31019a",
            "new_registered_scoring_sha256": "bdb369a69f99b076a180d2a3e00159a46bc7431787740f5a17bfff77ed44868c",
            "previous_correction_path": (
                str(previous_correction).replace("\\", "/")
                if previous_correction.exists() else None
            ),
            "previous_correction_sha256": (
                sha256_file(previous_correction) if previous_correction.exists() else None
            ),
            "new_scope": str(scope_path).replace("\\", "/"),
            "new_scope_sha256": sha256_file(scope_path),
            "new_approval": str(approval_path).replace("\\", "/"),
            "new_approval_sha256": sha256_file(approval_path),
            "partial_path": str(partial_path).replace("\\", "/"),
            "partial_sha256_at_correction": (
                sha256_file(partial_path) if partial_path.exists() else None
            ),
            "partial_documents_at_correction": len(partial_rows),
            "resume_preserves_completed_rows": True,
        },
        "isolation_audit": {
            "train_or_dev_changed": False,
            "calibration_input_changed": False,
            "checkpoint_scores_changed": False,
            "test_input_changed": False,
            "test_features_or_scores_computed": False,
            "test_remains_sealed": True,
            "model_or_checkpoint_selection_changed": False,
        },
    }
    payload["content_hash"] = content_hash(payload)
    if correction_path.exists():
        existing = json.loads(correction_path.read_text(encoding="utf-8"))
        for key in ("root_cause", "isolation_audit"):
            if existing.get(key) != payload.get(key):
                raise RuntimeError("Existing zero-token registered correction drifted")
        return
    stage._write_new(correction_path, payload)


def _prepare_records(registered_scores: Path) -> Path:
    output_path = CAL_ROOT / "parent_calibration_records.jsonl"
    result_path = CAL_ROOT / "prepare_calibration_result.json"
    scope_path = Path("docs/i06_prepare_parent_calibration_scope.json")
    scope = {
        "task_id": "I-06-prepare-parent-calibration",
        "freeze_manifest": str(GATE_FREEZE).replace("\\", "/"),
        "freeze_manifest_sha256": sha256_file(GATE_FREEZE),
        "runner_sha256": sha256_file("src/cwr_eg/runtime.py"),
        "code_files": _code_entries(
            [
                "src/cwr_eg/runtime.py",
                "src/cwr_eg/calibration_records.py",
                "src/cwr_eg/candidates.py",
                "src/cwr_eg/manifest.py",
                "src/cwr_eg/hashing.py",
            ]
        ),
        "scored_documents_path": str(registered_scores).replace("\\", "/"),
        "scored_documents_sha256": sha256_file(registered_scores),
        "expected_parent_count": 200,
        "monitoring": json.loads(Path("docs/i_gate_c_feature_scope.json").read_text(encoding="utf-8"))["monitoring"],
        "output_path": str(output_path).replace("\\", "/"),
        "result_path": str(result_path).replace("\\", "/"),
    }
    stage._scope_once(scope_path, scope)
    approval_path, _ = _approval("prepare-calibration", "i06_prepare_parent_calibration", scope_path)
    if not result_path.exists():
        _run_action("prepare-calibration", scope_path, approval_path, Path("status/i06_prepare_parent_calibration_console.log"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result["parents"] != 200 or result["output_sha256"] != sha256_file(output_path):
        raise RuntimeError("Parent calibration record QA failed")
    return output_path


def _fit_calibration(records_path: Path, freeze: dict[str, Any]) -> Path:
    output_dir = CAL_ROOT / "calibration_bundle"
    result_path = CAL_ROOT / "calibration_result.json"
    scope_path = Path("docs/i06_fit_parent_calibration_scope.json")
    scope = {
        "task_id": "I-06-fit-parent-calibration",
        "freeze_manifest": str(GATE_FREEZE).replace("\\", "/"),
        "freeze_manifest_sha256": sha256_file(GATE_FREEZE),
        "runner_sha256": sha256_file("src/cwr_eg/runtime.py"),
        "code_files": _code_entries(
            [
                "src/cwr_eg/runtime.py",
                "src/cwr_eg/bundle.py",
                "src/cwr_eg/calibration.py",
                "src/cwr_eg/hashing.py",
            ]
        ),
        "records_path": str(records_path).replace("\\", "/"),
        "records_sha256": sha256_file(records_path),
        "aggregation_unit": "parent_id",
        "minimum_parents_per_stratum": 100,
        "header": {
            "calibration_id": "i06-parent-max-v1",
            "protocol_version": "0.2.0-intermediate",
            "data_manifest_hash": sha256_file(GATE_FREEZE),
            "search_config_hash": sha256_file(CONFIG),
            "registered_registry_hash": sha256_file("configs/registered_registry.yaml"),
            "normalization_version": "raw-unicode-codepoint-v1",
            "model_version": "sha256:" + content_hash([row["sha256"] for row in freeze["full_checkpoints"]]),
            "languages": ["en", "zh"],
            "strata": ["en:all", "zh:all"],
        },
        "monitoring": json.loads(Path("docs/i_gate_c_feature_scope.json").read_text(encoding="utf-8"))["monitoring"],
        "output_dir": str(output_dir).replace("\\", "/"),
        "result_path": str(result_path).replace("\\", "/"),
    }
    stage._scope_once(scope_path, scope)
    approval_path, _ = _approval("calibrate", "i06_fit_parent_calibration", scope_path)
    if not result_path.exists():
        _run_action("calibrate", scope_path, approval_path, Path("status/i06_fit_parent_calibration_console.log"))
    bundle = CalibrationBundle.load(output_dir)
    counts = bundle.metadata["parent_counts_by_stratum"]
    minimum_p = bundle.metadata["minimum_empirical_p_by_stratum"]
    if counts != {"en:all": 100, "zh:all": 100} or any(abs(value - 1 / 101) > 1e-12 for value in minimum_p.values()):
        raise RuntimeError("Calibration pool size or minimum empirical p-value QA failed")
    return output_dir


def _complete(freeze: dict[str, Any], bundle_path: Path) -> None:
    completion = Path("manifests/i06_calibration_completion.json")
    if not completion.exists():
        bundle_manifest = bundle_path / "calibration_manifest.json"
        null_path = bundle_path / "null_distributions.npz"
        stage._write_new(
            completion,
            {
                "task_id": "I-06",
                "completed_at": ops._now(),
                "gate_freeze_sha256": sha256_file(GATE_FREEZE),
                "calibration_documents": freeze["calibration_null_input"]["documents"],
                "calibration_parents": 200,
                "parent_counts_by_stratum": {"en:all": 100, "zh:all": 100},
                "minimum_empirical_p": 1 / 101,
                "calibration_manifest_sha256": sha256_file(bundle_manifest),
                "null_distributions_sha256": sha256_file(null_path),
                "test_input_sha256": freeze["test_input"]["sha256"],
                "test_execution_remained_sealed": True,
                "deviation_audit": {
                    "status": "pass",
                    "aggregation_unit": "parent_id",
                    "clean_and_attacked_clean_merged_by_parent_maximum": True,
                    "calibration_strata_changed": False,
                    "test_features_or_scores_computed": False,
                },
            },
        )


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--i05-process-id", type=int, required=True)
    args = parser.parse_args()
    os.chdir(ROOT)
    ops.STATUS_PATH = STATUS
    ops.CONFIG = CONFIG
    ops._set_status("wait_for_i05", "in_progress", process_id=args.i05_process_id)
    _wait_for_i05(args.i05_process_id)
    ops._set_status("i_gate_d_freeze", "in_progress")
    calibration, test = _write_split_inputs()
    runs = _training_runs()
    freeze = _freeze(calibration, test, runs)
    ops._set_status("calibration_features", "in_progress", documents=len(calibration))
    feature_manifest, _ = _calibration_features(freeze)
    ops._set_status("calibration_checkpoint_scoring", "in_progress")
    checkpoint_scores = _calibration_checkpoint_scores(freeze, feature_manifest)
    ops._set_status("calibration_registered_scoring", "in_progress")
    registered_scores = _registered_scores(freeze, checkpoint_scores)
    ops._set_status("parent_calibration", "in_progress")
    records = _prepare_records(registered_scores)
    bundle = _fit_calibration(records, freeze)
    _complete(freeze, bundle)
    ops._set_status("i06_complete", "done", bundle=str(bundle).replace("\\", "/"))
    print(json.dumps({"calibration_parents": 200, "status": "done"}, sort_keys=True))
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
            "I-GATE-D-I-06",
            "blocked",
            "Fail-closed I-GATE-D/I-06 continuation stopped; inspect status/i_gate_d_calibration_status.json.",
            error_type=type(error).__name__,
            error_message=str(error),
        )
        raise
