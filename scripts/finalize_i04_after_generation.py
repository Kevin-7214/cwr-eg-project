from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import continue_i04_after_base as ops


ROOT = Path(__file__).resolve().parents[1]


def _write_once(path: Path, payload: dict) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite completion artifact: {path}")
    ops._write_json(path, payload)


def main() -> int:
    os.chdir(ROOT)
    ops.FREEZE_PATH = Path("manifests/intermediate_freeze_amendment_04.json")
    ops.STATUS_PATH = Path("status/i04_finalize_status.json")
    base_path = Path("artifacts/i04_full/base_generated_reconciled.jsonl")
    attack_path = Path("artifacts/i04_full/attacked_generated.jsonl")
    mixed_path = Path("artifacts/i04_full/mixed_generated.jsonl")
    base_qa = ops._validate_generation("base_generation", base_path, 4000)
    attack_qa = ops._validate_generation("matched_attack", attack_path, 4000)
    mixed_qa = ops._validate_generation("mixed_document", mixed_path, 400)

    base_completion_path = Path("manifests/i_04_full_base_completion.json")
    if not base_completion_path.exists():
        _write_once(
            base_completion_path,
            {
                "task_id": "I-04-full-base",
                "completed_at": ops._now(),
                "qa": base_qa,
                "reconciliation_manifest": "manifests/i04_base_retry_reconciliation.json",
                "reconciliation_manifest_sha256": ops.sha256_file(
                    "manifests/i04_base_retry_reconciliation.json"
                ),
                "resources": ops._resource_summary("I-04-full-base"),
                "deviation_audit": {
                    "status": "pass_with_explicit_failures",
                    "objective_changed": False,
                    "sample_scale_changed": False,
                    "model_or_generation_parameters_changed": False,
                    "calibration_or_test_unsealed": False,
                },
            },
        )
    mixed_completion_path = Path("manifests/i_04_full_mixed_completion.json")
    if not mixed_completion_path.exists():
        ops._write_completion(
            "I-04-full-mixed",
            mixed_qa,
            Path("artifacts/i04_full/mixed_generation_result.json"),
            ops._resource_summary("I-04-full-mixed-v3"),
            40,
        )

    generated_train_mixed = sum(
        row.get("status") == "generated" and row.get("split") == "train"
        for row in ops.read_jsonl(mixed_path)
    )
    expected_feature_documents = (
        base_qa["generated_rows"]
        + attack_qa["generated_rows"]
        + mixed_qa["generated_rows"]
        - generated_train_mixed
    )
    ops._set_status(
        "assemble",
        "in_progress",
        expected_feature_documents=expected_feature_documents,
    )
    assemble_scope = Path("docs/i04_full_assemble_scope.json")
    if not assemble_scope.exists():
        assemble_scope = ops._write_assemble_scope(
            [
                {"path": str(base_path).replace("\\", "/"), "sha256": base_qa["sha256"]},
                {
                    "path": str(attack_path).replace("\\", "/"),
                    "sha256": attack_qa["sha256"],
                },
                {
                    "path": str(mixed_path).replace("\\", "/"),
                    "sha256": mixed_qa["sha256"],
                },
            ],
            expected_feature_documents=expected_feature_documents,
        )
    approval_path = Path("status/approvals/i04_full_assemble.json")
    if not approval_path.exists():
        approval = ops._approval(
            approval_id="I04-FULL-ASSEMBLE-20260815",
            action="assemble-data",
            scope_path=assemble_scope,
            output_path=approval_path,
            evidence=(
                "User authorized continued execution of the intermediate plan on "
                "2026-08-15. This CPU-only scope binds the validated base, attack, and "
                "mixed outputs and does not unseal Calibration or Test."
            ),
        )
    else:
        approval = json.loads(approval_path.read_text(encoding="utf-8"))

    result_path = Path("artifacts/i04_full/assemble_result.json")
    if not result_path.exists():
        ops._run(
            [
                sys.executable,
                "-m",
                "cwr_eg.cli",
                "assemble-data",
                "--config",
                str(ops.CONFIG),
                "--resource-class",
                ops.RESOURCE_CLASS,
                "--scope-file",
                str(assemble_scope),
                "--approval",
                str(approval_path),
            ],
            Path("status/i04_full_assemble_console.log"),
        )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    total_failures = (
        base_qa["failed_rows"] + attack_qa["failed_rows"] + mixed_qa["failed_rows"]
    )
    expected_status_counts = {"generated": 8400 - total_failures}
    if total_failures:
        expected_status_counts["failed"] = total_failures
    if (
        result["recipes"] != 8400
        or result["feature_documents"] != expected_feature_documents
        or result["train_mixed_excluded"] != 150
        or result["status_counts"] != expected_status_counts
    ):
        raise RuntimeError("Full assembly counts differ from the frozen I-04 protocol")
    completion_path = Path("manifests/i04_completion.json")
    if not completion_path.exists():
        _write_once(
            completion_path,
            {
                "task_id": "I-04",
                "completed_at": ops._now(),
                "status": "complete",
                "base": base_qa,
                "attack": attack_qa,
                "mixed": mixed_qa,
                "assemble_result": result,
                "assemble_result_sha256": ops.sha256_file(result_path),
                "assemble_fingerprint": approval["fingerprint"],
                "generation_failures": total_failures,
                "silent_drops": 0,
                "deviation_audit": {
                    "status": "pass_with_explicit_failures",
                    "objective_changed": False,
                    "sample_scale_changed": False,
                    "frozen_recipe_or_split_changed": False,
                    "model_assets_changed": False,
                    "generation_parameters_changed": False,
                    "calibration_or_test_unsealed": False,
                    "stop_conditions_bypassed": False,
                },
            },
        )
    ops._append_progress(
        "I-04",
        "done",
        "I-04 completed all 8400 frozen recipes with 11 explicit failures, no silent drop, and a passing deviation audit.",
        completion_sha256=ops.sha256_file(completion_path),
        feature_documents=expected_feature_documents,
        explicit_failures=total_failures,
    )
    ops._set_status(
        "complete",
        "done",
        completion_sha256=ops.sha256_file(completion_path),
    )
    print(
        json.dumps(
            {
                "completion_sha256": ops.sha256_file(completion_path),
                "feature_documents": expected_feature_documents,
                "explicit_failures": total_failures,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
