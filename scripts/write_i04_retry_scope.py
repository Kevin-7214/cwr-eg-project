from __future__ import annotations

import json
from pathlib import Path

from cwr_eg.hashing import sha256_file


def main() -> int:
    output = Path("docs/i04_base_deterministic_retry_scope.json")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite retry scope: {output}")
    scope = json.loads(
        Path("docs/i04_full_base_generation_scope.json").read_text(encoding="utf-8")
    )
    scope.update(
        {
            "task_id": "I-04-base-deterministic-retry-1",
            "freeze_manifest": "manifests/intermediate_freeze_amendment_03.json",
            "freeze_manifest_sha256": sha256_file(
                "manifests/intermediate_freeze_amendment_03.json"
            ),
            "runner_sha256": sha256_file("src/cwr_eg/runtime.py"),
            "recipe_ids": [
                "base-clean-399f2e1a9c4cf811",
                "base-clean-52812039e5522400",
                "base-unbiased-52812039e5522400",
            ],
            "expected_recipe_count": 3,
            "limit": 3,
            "generation_retry_index": 1,
            "maximum_failure_rate": 1.0,
            "residual_failure_policy": "retain_explicit_failure_and_continue",
            "recursive_retry_allowed": False,
            "original_output_path": "artifacts/i04_full/base_generated.jsonl",
            "original_output_sha256": sha256_file(
                "artifacts/i04_full/base_generated.jsonl"
            ),
            "output_path": "artifacts/i04_full/base_retry_1.jsonl",
            "result_path": "artifacts/i04_full/base_retry_1_result.json",
        }
    )
    for field in (
        "resume_partial_sha256",
        "expected_resumed_documents",
        "reuse_source_task",
        "reuse_policy",
    ):
        scope.pop(field, None)
    code_files = [
        entry for entry in scope["code_files"] if entry["path"] != "src/cwr_eg/runtime.py"
    ]
    code_files.append(
        {"path": "src/cwr_eg/runtime.py", "sha256": sha256_file("src/cwr_eg/runtime.py")}
    )
    scope["code_files"] = code_files
    output.write_text(json.dumps(scope, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"path": str(output), "sha256": sha256_file(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
