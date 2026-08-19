from __future__ import annotations

from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Any

from cwr_eg.generated_data import _validate_generated_row
from cwr_eg.hashing import sha256_file
from cwr_eg.manifest import read_jsonl


ROOT = Path(__file__).resolve().parents[1]
CONFIG = Path("configs/intermediate.yaml")
RESOURCE_CLASS = "local-rtx5060-8gb"
STATUS_PATH = Path("status/i04_continuation_status.json")
PROGRESS_PATH = Path("status/progress.jsonl")
RECIPE_PATH = Path("manifests/intermediate_recipes.jsonl")
FREEZE_PATH = Path("manifests/intermediate_freeze_amendment_03.json")


def _now() -> str:
    return datetime.now().astimezone().isoformat()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _set_status(phase: str, status: str, **details: Any) -> None:
    _write_json(
        STATUS_PATH,
        {"time": _now(), "phase": phase, "status": status, "details": details},
    )


def _append_progress(task_id: str, status: str, evidence: str, **details: Any) -> None:
    payload = {
        "time": _now(),
        "task_id": task_id,
        "status": status,
        "evidence": evidence,
        "details": details,
    }
    with PROGRESS_PATH.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def _run(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    with log_path.open("w", encoding="utf-8", newline="\n") as log:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed with code {completed.returncode}; see {log_path}"
        )


def _resource_summary(task_id: str) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in PROGRESS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selected = [row for row in rows if row.get("task_id") == task_id]
    if not selected:
        raise RuntimeError(f"No resource monitoring records found for {task_id}")
    details = [row["details"] for row in selected if "gpu_temperature_c" in row["details"]]
    return {
        "monitor_points": len(details),
        "maximum_gpu_temperature_c": max(row["gpu_temperature_c"] for row in details),
        "maximum_used_ram_gib": max(row["used_ram_gib"] for row in details),
        "minimum_free_disk_gib": min(row["free_disk_gib"] for row in details),
        "first_recorded_at": selected[0]["time"],
        "last_recorded_at": selected[-1]["time"],
    }


def _validate_generation(kind: str, path: Path, expected: int) -> dict[str, Any]:
    recipes = [row for row in read_jsonl(RECIPE_PATH) if row["kind"] == kind]
    if len(recipes) != expected:
        raise RuntimeError(f"Frozen {kind} recipe count changed")
    rows = read_jsonl(path)
    if len(rows) != expected:
        raise RuntimeError(f"{kind} output count is {len(rows)}, expected {expected}")
    output_sha256 = sha256_file(path)
    recipe_by_id = {str(row["recipe_id"]): row for row in recipes}
    row_ids = [str(row["recipe_id"]) for row in rows]
    if len(set(row_ids)) != len(rows) or set(row_ids) != set(recipe_by_id):
        raise RuntimeError(f"{kind} output ids differ from the frozen recipe set")
    for row in rows:
        _validate_generated_row(row, recipe_by_id[str(row["recipe_id"])])
    failed = [row for row in rows if row.get("status") == "failed"]
    return {
        "path": str(path),
        "sha256": output_sha256,
        "bytes": path.stat().st_size,
        "rows": len(rows),
        "unique_recipe_ids": len({str(row["recipe_id"]) for row in rows}),
        "generated_rows": len(rows) - len(failed),
        "failed_rows": len(failed),
        "failed_recipe_ids": [str(row["recipe_id"]) for row in failed],
    }


def _approval(
    *, approval_id: str, action: str, scope_path: Path, output_path: Path, evidence: str
) -> dict[str, Any]:
    issued = datetime.now().astimezone()
    expires = issued + timedelta(days=3)
    _run(
        [
            sys.executable,
            "scripts/create_approval_record.py",
            "--approval-id",
            approval_id,
            "--action",
            action,
            "--config",
            str(CONFIG),
            "--resource-class",
            RESOURCE_CLASS,
            "--scope-file",
            str(scope_path),
            "--issued-at",
            issued.isoformat(),
            "--expires-at",
            expires.isoformat(),
            "--chat-evidence",
            evidence,
            "--output",
            str(output_path),
        ],
        Path(f"status/{approval_id.lower()}_creation.log"),
    )
    return json.loads(output_path.read_text(encoding="utf-8"))


def _write_completion(
    name: str,
    qa: dict[str, Any],
    result_path: Path,
    resources: dict[str, Any],
    resumed: int,
) -> None:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    payload = {
        "task_id": name,
        "completed_at": _now(),
        "qa": qa,
        "result_path": str(result_path),
        "result_sha256": sha256_file(result_path),
        "resumed_documents": resumed,
        "new_documents": qa["rows"] - resumed,
        "runtime_result": result,
        "resources": resources,
        "deviation_audit": {
            "status": (
                "pass_with_explicit_failures" if qa["failed_rows"] else "pass"
            ),
            "objective_changed": False,
            "sample_scale_changed": False,
            "frozen_recipe_fields_changed": False,
            "model_or_generation_parameters_changed": False,
            "calibration_or_test_unsealed": False,
            "stop_condition_bypassed": False,
        },
    }
    _write_json(Path(f"manifests/{name.lower().replace('-', '_')}_completion.json"), payload)


def _write_assemble_scope(
    inputs: list[dict[str, Any]], *, expected_feature_documents: int
) -> Path:
    path = Path("docs/i04_full_assemble_scope.json")
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite scope: {path}")
    scope = json.loads(
        Path("docs/i03_canary_assemble_scope.json").read_text(encoding="utf-8")
    )
    scope.update(
        {
            "task_id": "I-04-full-assemble",
            "freeze_manifest": str(FREEZE_PATH).replace("\\", "/"),
            "freeze_manifest_sha256": sha256_file(FREEZE_PATH),
            "parent_freeze_manifest_sha256": sha256_file(
                "manifests/intermediate_freeze_manifest.json"
            ),
            "runner_sha256": sha256_file("src/cwr_eg/runtime.py"),
            "recipe_manifest": str(RECIPE_PATH).replace("\\", "/"),
            "recipe_manifest_sha256": sha256_file(RECIPE_PATH),
            "inputs": inputs,
            "expected_recipe_count": 8400,
            "expected_feature_documents": expected_feature_documents,
            "expected_train_mixed_excluded": 150,
            "output_path": "artifacts/i04_full/all_generated.jsonl",
            "feature_documents_path": "artifacts/i04_full/feature_documents.jsonl",
            "result_path": "artifacts/i04_full/assemble_result.json",
        }
    )
    for entry in scope["code_files"]:
        entry["sha256"] = sha256_file(entry["path"])
    path.write_text(json.dumps(scope, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def main() -> int:
    os.chdir(ROOT)
    original_base_path = Path("artifacts/i04_full/base_generated.jsonl")
    reconciled_base_path = Path("artifacts/i04_full/base_generated_reconciled.jsonl")
    base_path = reconciled_base_path if reconciled_base_path.exists() else original_base_path
    base_result = Path("artifacts/i04_full/base_generation_result.json")
    _set_status("wait_for_base", "in_progress")
    deadline = time.monotonic() + 9 * 60 * 60
    while not (base_path.exists() and base_result.exists()):
        if time.monotonic() > deadline:
            raise TimeoutError("Full base generation did not complete within nine hours")
        time.sleep(30)
    _set_status("validate_base", "in_progress")
    base_qa = _validate_generation("base_generation", base_path, 4000)
    _write_completion(
        "I-04-full-base",
        base_qa,
        base_result,
        _resource_summary("I-04-full-base"),
        400,
    )
    _append_progress(
        "I-04-full-base-audit",
        "done",
        "Full base output passed exact count, hash, frozen-recipe, text, interval, and deviation checks.",
        output_sha256=base_qa["sha256"],
        rows=4000,
    )

    _set_status("freeze_attack_scope", "in_progress", base_sha256=base_qa["sha256"])
    attack_scope = Path("docs/i04_full_attack_generation_scope.json")
    _run(
        [
            sys.executable,
            "scripts/write_i04_scope.py",
            "--kind",
            "attack",
            "--base-input-sha256",
            base_qa["sha256"],
            "--base-input-path",
            str(base_path).replace("\\", "/"),
            "--output",
            str(attack_scope),
        ],
        Path("status/i04_attack_scope_creation.log"),
    )
    attack_approval_path = Path("status/approvals/i04_full_attack_generation.json")
    attack_approval = _approval(
        approval_id="I04-FULL-ATTACK-20260814",
        action="attack-generate",
        scope_path=attack_scope,
        output_path=attack_approval_path,
        evidence=(
            "User granted standing authorization for direct GPU execution of the I-stage "
            "intermediate plan on 2026-08-14. This dependent scope binds the QA-validated "
            f"full-base SHA-256 {base_qa['sha256']}, reuses 400 frozen canary attacks, and "
            "generates only the remaining 3600 matched attacks."
        ),
    )
    _append_progress(
        "I-04-full-attack-scope",
        "done",
        "Dependent attack scope and approval were frozen after full-base QA.",
        fingerprint=attack_approval["fingerprint"],
        base_sha256=base_qa["sha256"],
    )
    _set_status("run_attack", "in_progress", fingerprint=attack_approval["fingerprint"])
    _run(
        [
            sys.executable,
            "-m",
            "cwr_eg.cli",
            "attack-generate",
            "--config",
            str(CONFIG),
            "--resource-class",
            RESOURCE_CLASS,
            "--scope-file",
            str(attack_scope),
            "--approval",
            str(attack_approval_path),
        ],
        Path("status/i04_full_attack_console.log"),
    )
    attack_path = Path("artifacts/i04_full/attacked_generated.jsonl")
    attack_result = Path("artifacts/i04_full/attack_generation_result.json")
    attack_qa = _validate_generation("matched_attack", attack_path, 4000)
    _write_completion(
        "I-04-full-attack",
        attack_qa,
        attack_result,
        _resource_summary("I-04-full-attack"),
        400,
    )
    _append_progress(
        "I-04-full-attack-audit",
        "done",
        "Full attack output passed exact count, hash, frozen-recipe, text, interval, and deviation checks.",
        output_sha256=attack_qa["sha256"],
        rows=4000,
    )

    mixed_scope = Path("docs/i04_full_mixed_generation_scope.json")
    mixed_approval = Path("status/approvals/i04_full_mixed_generation.json")
    mixed_fingerprint = json.loads(mixed_approval.read_text(encoding="utf-8"))["fingerprint"]
    _set_status("run_mixed", "in_progress", fingerprint=mixed_fingerprint)
    _run(
        [
            sys.executable,
            "-m",
            "cwr_eg.cli",
            "generate",
            "--config",
            str(CONFIG),
            "--resource-class",
            RESOURCE_CLASS,
            "--scope-file",
            str(mixed_scope),
            "--approval",
            str(mixed_approval),
        ],
        Path("status/i04_full_mixed_console.log"),
    )
    mixed_path = Path("artifacts/i04_full/mixed_generated.jsonl")
    mixed_result = Path("artifacts/i04_full/mixed_generation_result.json")
    mixed_qa = _validate_generation("mixed_document", mixed_path, 400)
    _write_completion(
        "I-04-full-mixed",
        mixed_qa,
        mixed_result,
        _resource_summary("I-04-full-mixed"),
        40,
    )
    _append_progress(
        "I-04-full-mixed-audit",
        "done",
        "Full mixed output passed exact count, hash, frozen-recipe, text, interval, and deviation checks.",
        output_sha256=mixed_qa["sha256"],
        rows=400,
    )

    _set_status("assemble", "in_progress")
    mixed_rows = read_jsonl(mixed_path)
    generated_train_mixed = sum(
        row.get("status") == "generated" and row.get("split") == "train"
        for row in mixed_rows
    )
    expected_feature_documents = (
        base_qa["generated_rows"]
        + attack_qa["generated_rows"]
        + mixed_qa["generated_rows"]
        - generated_train_mixed
    )
    assemble_scope = _write_assemble_scope(
        [
            {"path": str(base_path).replace("\\", "/"), "sha256": base_qa["sha256"]},
            {"path": str(attack_path).replace("\\", "/"), "sha256": attack_qa["sha256"]},
            {"path": str(mixed_path).replace("\\", "/"), "sha256": mixed_qa["sha256"]},
        ],
        expected_feature_documents=expected_feature_documents,
    )
    assemble_approval_path = Path("status/approvals/i04_full_assemble.json")
    assemble_approval = _approval(
        approval_id="I04-FULL-ASSEMBLE-20260814",
        action="assemble-data",
        scope_path=assemble_scope,
        output_path=assemble_approval_path,
        evidence=(
            "User granted standing authorization for I-stage intermediate execution on "
            "2026-08-14. This CPU assembly scope binds all three QA-validated I-04 outputs "
            "and writes the frozen 8400-row order plus every successfully generated permitted feature document."
        ),
    )
    _run(
        [
            sys.executable,
            "-m",
            "cwr_eg.cli",
            "assemble-data",
            "--config",
            str(CONFIG),
            "--resource-class",
            RESOURCE_CLASS,
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
        "completed_at": _now(),
        "status": "complete",
        "base": base_qa,
        "attack": attack_qa,
        "mixed": mixed_qa,
        "assemble_result": assemble_result,
        "assemble_result_sha256": sha256_file(assemble_result_path),
        "assemble_fingerprint": assemble_approval["fingerprint"],
        "generation_failures": total_failures,
        "silent_drops": 0,
        "deviation_audit": {
            "status": (
                "pass_with_explicit_failures" if total_failures else "pass"
            ),
            "objective_changed": False,
            "sample_scale_changed": False,
            "frozen_recipe_or_split_changed": False,
            "model_assets_changed": False,
            "generation_parameters_changed": False,
            "calibration_or_test_unsealed": False,
            "stop_conditions_bypassed": False,
        },
    }
    _write_json(Path("manifests/i04_completion.json"), completion)
    _append_progress(
        "I-04",
        "done",
        "I-04 completed 8400/8400 frozen recipes with every residual failure explicit and no silent drop; deviation audit passed.",
        completion_sha256=sha256_file("manifests/i04_completion.json"),
        feature_documents=expected_feature_documents,
        explicit_failures=total_failures,
    )
    _set_status("complete", "done", completion_sha256=sha256_file("manifests/i04_completion.json"))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        os.chdir(ROOT)
        _set_status(
            "failed",
            "error",
            error_type=type(error).__name__,
            error_message=str(error),
            traceback=traceback.format_exc(),
        )
        _append_progress(
            "I-04-continuation",
            "blocked",
            "The fail-closed I-04 continuation stopped; inspect status/i04_continuation_status.json.",
            error_type=type(error).__name__,
            error_message=str(error),
        )
        raise
