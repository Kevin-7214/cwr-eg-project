from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path

from cwr_eg.hashing import content_hash, sha256_file
from cwr_eg.manifest import read_jsonl
from cwr_eg.runtime import _load_approved_generation_partial


AMENDMENT_PATH = Path("manifests/intermediate_freeze_amendment_04.json")
ATTACK_SCOPE_PATH = Path("docs/i04_full_attack_resume_scope.json")
MIXED_SCOPE_PATH = Path("docs/i04_full_mixed_generation_scope_v3.json")
PARTIAL_PATH = Path("artifacts/i04_full/attacked_generated.jsonl.partial")
RECIPE_PATH = Path("manifests/intermediate_recipes.jsonl")
RUNTIME_PATH = Path("src/cwr_eg/runtime.py")


def _replace_runtime_hash(scope: dict) -> None:
    runtime_sha256 = sha256_file(RUNTIME_PATH)
    scope["runner_sha256"] = runtime_sha256
    for entry in scope["code_files"]:
        if entry["path"] == "src/cwr_eg/runtime.py":
            entry["sha256"] = runtime_sha256
            return
    raise RuntimeError("Frozen scope does not list the runtime")


def _write_new(path: Path, payload: dict) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite frozen artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> int:
    recipes = [
        row for row in read_jsonl(RECIPE_PATH) if row["kind"] == "matched_attack"
    ]
    source_scope_path = Path("docs/i04_full_attack_generation_scope.json")
    source_scope = json.loads(source_scope_path.read_text(encoding="utf-8"))
    recipes = [
        row for row in recipes if row["attack_id"] in set(source_scope["attack_ids"])
    ][: int(source_scope["limit"])]
    partial_sha256 = sha256_file(PARTIAL_PATH)
    rows = _load_approved_generation_partial(
        PARTIAL_PATH,
        expected_sha256=partial_sha256,
        expected_count=727,
        recipes=recipes,
        allow_failed_rows=True,
    )
    failed_ids = [str(row["recipe_id"]) for row in rows if row["status"] == "failed"]
    if len(rows) != 727 or len({row["recipe_id"] for row in rows}) != 727:
        raise RuntimeError("Attack resume partial does not contain 727 unique rows")
    if failed_ids != ["attack-base-clean-52812039e5522400"]:
        raise RuntimeError("Attack resume partial failure set drifted")

    parent_path = Path("manifests/intermediate_freeze_amendment_03.json")
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    amendment = {
        "manifest_version": "intermediate-freeze-amendment-v1",
        "amendment_id": "I-FIX-04-explicit-attack-empty-output-resume",
        "created_at": datetime.now().astimezone().isoformat(),
        "applies_from": "I-04-attack-resume",
        "parent_amendment": {
            "path": str(parent_path).replace("\\", "/"),
            "sha256": sha256_file(parent_path),
            "amendment_content_hash": parent["amendment_content_hash"],
        },
        "original_freeze_manifest_sha256": sha256_file(
            "manifests/intermediate_freeze_manifest.json"
        ),
        "user_decision": "Continue from the previous attack interruption.",
        "test_sealed": True,
        "calibration_sealed": True,
        "trigger": {
            "source_scope_path": str(source_scope_path).replace("\\", "/"),
            "source_scope_sha256": sha256_file(source_scope_path),
            "source_approval_sha256": sha256_file(
                "status/approvals/i04_full_attack_generation.json"
            ),
            "partial_path": str(PARTIAL_PATH).replace("\\", "/"),
            "partial_sha256": partial_sha256,
            "rows": len(rows),
            "generated_rows": sum(row["status"] == "generated" for row in rows),
            "failed_rows": len(failed_ids),
            "failed_recipe_ids": failed_ids,
            "interrupted_recipe_id": "attack-base-clean-77152e4e6f27494f",
            "interrupted_attack_id": "translation_roundtrip",
            "error": "Attack produced empty text",
        },
        "recovery_policy": {
            "resume_completed_rows": 727,
            "recompute_completed_rows": False,
            "allow_explicit_failed_rows_only_when_scope_enabled": True,
            "empty_attack_output_policy": "record_explicit_failure_and_continue",
            "maximum_failure_rate": 0.01,
            "same_model": True,
            "same_revision": True,
            "same_attack_parameters": True,
            "test_unsealed": False,
        },
        "code_changes": [
            {
                "path": str(RUNTIME_PATH).replace("\\", "/"),
                "previous_sha256": "48f96a2881733b670bc740bff771d7aebbd46e32434afa3210d063d7c7c90d16",
                "current_sha256": sha256_file(RUNTIME_PATH),
                "bytes": RUNTIME_PATH.stat().st_size,
                "purpose": "scope-gated failed-row resume and explicit empty-attack failure recording",
            },
            {
                "path": "scripts/continue_i04_after_retry.py",
                "sha256": sha256_file("scripts/continue_i04_after_retry.py"),
                "bytes": Path("scripts/continue_i04_after_retry.py").stat().st_size,
                "purpose": "continue with the amendment-04 mixed and assembly freeze",
            },
        ],
        "tests": {
            "path": "tests/test_generation_resume_scope.py",
            "sha256": sha256_file("tests/test_generation_resume_scope.py"),
            "passed": 68,
            "failed": 0,
            "cuda_visible_devices": "",
        },
        "deviation_audit": {
            "status": "approved_minimal_recovery",
            "objective_changed": False,
            "sample_or_recipe_set_changed": False,
            "completed_attack_outputs_changed": False,
            "model_asset_changed": False,
            "attack_parameters_changed": False,
            "failure_threshold_changed": False,
            "calibration_or_test_unsealed": False,
        },
    }
    amendment["amendment_content_hash"] = content_hash(amendment)
    _write_new(AMENDMENT_PATH, amendment)
    amendment_sha256 = sha256_file(AMENDMENT_PATH)

    attack_scope = source_scope
    attack_scope.update(
        {
            "task_id": "I-04-full-attack-resume-1",
            "freeze_manifest": str(AMENDMENT_PATH).replace("\\", "/"),
            "freeze_manifest_sha256": amendment_sha256,
            "expected_resumed_documents": 727,
            "resume_partial_sha256": partial_sha256,
            "resume_explicit_failures": True,
            "resume_source_scope_sha256": sha256_file(source_scope_path),
            "resume_policy": "hash-bound-explicit-failures-v1",
            "interrupted_recipe_id": "attack-base-clean-77152e4e6f27494f",
        }
    )
    _replace_runtime_hash(attack_scope)
    _write_new(ATTACK_SCOPE_PATH, attack_scope)

    mixed_source = json.loads(
        Path("docs/i04_full_mixed_generation_scope_v2.json").read_text(
            encoding="utf-8"
        )
    )
    mixed_source.update(
        {
            "task_id": "I-04-full-mixed-v3",
            "freeze_manifest": str(AMENDMENT_PATH).replace("\\", "/"),
            "freeze_manifest_sha256": amendment_sha256,
        }
    )
    _replace_runtime_hash(mixed_source)
    _write_new(MIXED_SCOPE_PATH, mixed_source)
    print(
        json.dumps(
            {
                "amendment_sha256": amendment_sha256,
                "attack_scope_sha256": sha256_file(ATTACK_SCOPE_PATH),
                "mixed_scope_sha256": sha256_file(MIXED_SCOPE_PATH),
                "resume_rows": len(rows),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
