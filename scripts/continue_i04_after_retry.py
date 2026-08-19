from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import continue_i04_after_base as ops


ROOT = Path(__file__).resolve().parents[1]
STATUS_PATH = Path("status/i04_post_retry_continuation_status.json")


def _process_exists(process_id: int) -> bool:
    completed = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            f"if (Get-Process -Id {process_id} -ErrorAction SilentlyContinue) "
            "{ exit 0 } else { exit 1 }",
        ],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return completed.returncode == 0


def _wait_for_attack(process_id: int) -> None:
    attack_path = Path("artifacts/i04_full/attacked_generated.jsonl")
    result_path = Path("artifacts/i04_full/attack_generation_result.json")
    deadline = time.monotonic() + 8 * 60 * 60
    while not (attack_path.exists() and result_path.exists()):
        if not _process_exists(process_id):
            raise RuntimeError("Attack process exited without complete output and result")
        if time.monotonic() > deadline:
            raise TimeoutError("Matched-attack generation exceeded eight hours")
        time.sleep(30)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attack-process-id", type=int, required=True)
    args = parser.parse_args()
    os.chdir(ROOT)
    ops.STATUS_PATH = STATUS_PATH
    ops.FREEZE_PATH = Path("manifests/intermediate_freeze_amendment_04.json")
    ops._set_status("wait_for_attack", "in_progress", process_id=args.attack_process_id)
    _wait_for_attack(args.attack_process_id)

    base_path = Path("artifacts/i04_full/base_generated_reconciled.jsonl")
    attack_path = Path("artifacts/i04_full/attacked_generated.jsonl")
    attack_result = Path("artifacts/i04_full/attack_generation_result.json")
    base_qa = ops._validate_generation("base_generation", base_path, 4000)
    attack_qa = ops._validate_generation("matched_attack", attack_path, 4000)
    ops._write_completion(
        "I-04-full-attack",
        attack_qa,
        attack_result,
        ops._resource_summary("I-04-full-attack"),
        400,
    )
    ops._append_progress(
        "I-04-full-attack-audit",
        "done",
        "Full attack output passed exact count, hash, frozen-recipe, text, interval, and deviation checks.",
        output_sha256=attack_qa["sha256"],
        rows=4000,
        explicit_failures=attack_qa["failed_rows"],
    )

    mixed_scope = Path("docs/i04_full_mixed_generation_scope_v3.json")
    mixed_approval = Path("status/approvals/i04_full_mixed_generation_v3.json")
    mixed_fingerprint = json.loads(mixed_approval.read_text(encoding="utf-8"))[
        "fingerprint"
    ]
    ops._set_status("run_mixed", "in_progress", fingerprint=mixed_fingerprint)
    ops._run(
        [
            sys.executable,
            "-m",
            "cwr_eg.cli",
            "generate",
            "--config",
            str(ops.CONFIG),
            "--resource-class",
            ops.RESOURCE_CLASS,
            "--scope-file",
            str(mixed_scope),
            "--approval",
            str(mixed_approval),
        ],
        Path("status/i04_full_mixed_v2_console.log"),
    )
    mixed_path = Path("artifacts/i04_full/mixed_generated.jsonl")
    mixed_result = Path("artifacts/i04_full/mixed_generation_result.json")
    mixed_qa = ops._validate_generation("mixed_document", mixed_path, 400)
    ops._write_completion(
        "I-04-full-mixed",
        mixed_qa,
        mixed_result,
        ops._resource_summary("I-04-full-mixed-v3"),
        40,
    )
    ops._append_progress(
        "I-04-full-mixed-audit",
        "done",
        "Full mixed output passed exact count, hash, frozen-recipe, text, interval, and deviation checks.",
        output_sha256=mixed_qa["sha256"],
        rows=400,
        explicit_failures=mixed_qa["failed_rows"],
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
    ops._set_status("assemble", "in_progress")
    assemble_scope = ops._write_assemble_scope(
        [
            {"path": str(base_path).replace("\\", "/"), "sha256": base_qa["sha256"]},
            {
                "path": str(attack_path).replace("\\", "/"),
                "sha256": attack_qa["sha256"],
            },
            {"path": str(mixed_path).replace("\\", "/"), "sha256": mixed_qa["sha256"]},
        ],
        expected_feature_documents=expected_feature_documents,
    )
    assemble_approval_path = Path("status/approvals/i04_full_assemble.json")
    assemble_approval = ops._approval(
        approval_id="I04-FULL-ASSEMBLE-20260814-R1",
        action="assemble-data",
        scope_path=assemble_scope,
        output_path=assemble_approval_path,
        evidence=(
            "User authorized direct I-stage execution. This CPU assembly scope binds the "
            "QA-validated reconciled base, matched-attack, and mixed outputs after the "
            "single deterministic retry decision."
        ),
    )
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
            str(assemble_approval_path),
        ],
        Path("status/i04_full_assemble_console.log"),
    )
    assemble_result_path = Path("artifacts/i04_full/assemble_result.json")
    assemble_result = json.loads(assemble_result_path.read_text(encoding="utf-8"))
    total_failures = (
        base_qa["failed_rows"] + attack_qa["failed_rows"] + mixed_qa["failed_rows"]
    )
    expected_status_counts = {"generated": 8400 - total_failures}
    if total_failures:
        expected_status_counts["failed"] = total_failures
    if (
        assemble_result["recipes"] != 8400
        or assemble_result["feature_documents"] != expected_feature_documents
        or assemble_result["train_mixed_excluded"] != 150
        or assemble_result["status_counts"] != expected_status_counts
    ):
        raise RuntimeError("Full assembly counts differ from the frozen I-04 protocol")

    completion = {
        "task_id": "I-04",
        "completed_at": ops._now(),
        "status": "complete",
        "base": base_qa,
        "attack": attack_qa,
        "mixed": mixed_qa,
        "assemble_result": assemble_result,
        "assemble_result_sha256": ops.sha256_file(assemble_result_path),
        "assemble_fingerprint": assemble_approval["fingerprint"],
        "generation_failures": total_failures,
        "silent_drops": 0,
        "deviation_audit": {
            "status": "pass_with_explicit_failures" if total_failures else "pass",
            "objective_changed": False,
            "sample_scale_changed": False,
            "frozen_recipe_or_split_changed": False,
            "model_assets_changed": False,
            "generation_parameters_changed": False,
            "calibration_or_test_unsealed": False,
            "stop_conditions_bypassed": False,
            "recursive_retry_performed": False,
        },
    }
    ops._write_json(Path("manifests/i04_completion.json"), completion)
    ops._append_progress(
        "I-04",
        "done",
        "I-04 completed 8400/8400 frozen recipes with every residual failure explicit and no silent drop; deviation audit passed.",
        completion_sha256=ops.sha256_file("manifests/i04_completion.json"),
        feature_documents=expected_feature_documents,
        explicit_failures=total_failures,
    )
    ops._set_status(
        "complete",
        "done",
        completion_sha256=ops.sha256_file("manifests/i04_completion.json"),
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        os.chdir(ROOT)
        ops.STATUS_PATH = STATUS_PATH
        ops._set_status(
            "failed",
            "error",
            error_type=type(error).__name__,
            error_message=str(error),
            traceback=traceback.format_exc(),
        )
        ops._append_progress(
            "I-04-post-retry-continuation",
            "blocked",
            "The fail-closed post-retry continuation stopped; inspect the dedicated status file.",
            error_type=type(error).__name__,
            error_message=str(error),
        )
        raise
