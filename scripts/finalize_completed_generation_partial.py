from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from cwr_eg.hashing import sha256_file, sha256_text
from cwr_eg.manifest import read_jsonl


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", type=Path, required=True)
    parser.add_argument("--partial", type=Path, required=True)
    parser.add_argument("--expected-partial-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists() or args.result.exists():
        raise FileExistsError("Refusing to overwrite finalized retry artifacts")
    if sha256_file(args.partial) != args.expected_partial_sha256:
        raise RuntimeError("Retry partial SHA-256 mismatch")

    scope = json.loads(args.scope.read_text(encoding="utf-8"))
    rows = read_jsonl(args.partial)
    expected_ids = [str(value) for value in scope["recipe_ids"]]
    actual_ids = [str(row.get("recipe_id")) for row in rows]
    if len(rows) != int(scope["expected_recipe_count"]):
        raise RuntimeError("Retry partial count does not match the frozen scope")
    if len(actual_ids) != len(set(actual_ids)) or set(actual_ids) != set(expected_ids):
        raise RuntimeError("Retry partial IDs do not exactly match the frozen scope")

    retry_index = int(scope["generation_retry_index"])
    generated = 0
    failed = 0
    for row in rows:
        if int(row.get("generation_retry_index", -1)) != retry_index:
            raise RuntimeError("Retry index drift detected")
        status = row.get("status")
        if status == "generated":
            text = str(row.get("text", ""))
            if not text or sha256_text(text) != str(row.get("text_sha256")):
                raise RuntimeError("Generated retry row has invalid text provenance")
            generated += 1
        elif status == "failed":
            if not row.get("failure_type") or not row.get("failure_message"):
                raise RuntimeError("Failed retry row lacks explicit failure provenance")
            failed += 1
        else:
            raise RuntimeError(f"Unsupported retry status: {status!r}")

    partial_sha256 = sha256_file(args.partial)
    os.replace(args.partial, args.output)
    result = {
        "task_id": scope["task_id"],
        "status": "completed_by_validated_partial_recovery",
        "generated": generated,
        "failed": failed,
        "generation_retry_index": retry_index,
        "expected_recipe_count": len(expected_ids),
        "output_path": str(args.output),
        "output_sha256": sha256_file(args.output),
        "recovered_partial_sha256": partial_sha256,
        "recursive_retry_performed": False,
    }
    temporary = args.result.with_suffix(args.result.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, args.result)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
