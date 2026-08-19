from __future__ import annotations

import argparse
import json
from pathlib import Path

from cwr_eg.generated_data import _validate_generated_row
from cwr_eg.hashing import sha256_file
from cwr_eg.manifest import read_jsonl, write_jsonl


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--retry", type=Path, required=True)
    parser.add_argument("--recipe-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.manifest.exists():
        raise FileExistsError("Refusing to overwrite reconciled generation artifacts")
    recipes = [
        row
        for row in read_jsonl(args.recipe_manifest)
        if row["kind"] == "base_generation"
    ]
    recipe_by_id = {str(row["recipe_id"]): row for row in recipes}
    original = read_jsonl(args.original)
    retry = read_jsonl(args.retry)
    if len(original) != len(recipes) or len({row["recipe_id"] for row in original}) != len(original):
        raise RuntimeError("Original base output is incomplete or contains duplicate ids")
    failed_by_id = {
        str(row["recipe_id"]): row for row in original if row.get("status") == "failed"
    }
    retry_by_id = {str(row["recipe_id"]): row for row in retry}
    if set(retry_by_id) != set(failed_by_id) or len(retry_by_id) != len(retry):
        raise RuntimeError("Retry output does not match the exact original failure set")
    reconciled = []
    recovered = 0
    for row in original:
        recipe_id = str(row["recipe_id"])
        if recipe_id not in retry_by_id:
            reconciled.append(row)
            continue
        retry_row = retry_by_id[recipe_id]
        if retry_row.get("status") == "generated":
            replacement = {
                **retry_row,
                "initial_failure": {
                    "failure_type": row["failure_type"],
                    "failure_message": row["failure_message"],
                },
            }
            recovered += 1
        else:
            replacement = {
                **row,
                "generation_retry_index": retry_row.get("generation_retry_index", 1),
                "retry_failure": {
                    "failure_type": retry_row["failure_type"],
                    "failure_message": retry_row["failure_message"],
                },
            }
        reconciled.append(replacement)
    for row in reconciled:
        recipe_id = str(row["recipe_id"])
        if recipe_id not in recipe_by_id:
            raise RuntimeError("Reconciled output contains an unknown recipe id")
        _validate_generated_row(row, recipe_by_id[recipe_id])
    write_jsonl(args.output, reconciled)
    unresolved = len(failed_by_id) - recovered
    payload = {
        "original_path": str(args.original),
        "original_sha256": sha256_file(args.original),
        "retry_path": str(args.retry),
        "retry_sha256": sha256_file(args.retry),
        "output_path": str(args.output),
        "output_sha256": sha256_file(args.output),
        "rows": len(reconciled),
        "initial_failures": len(failed_by_id),
        "recovered": recovered,
        "unresolved_failures": unresolved,
        "unresolved_recipe_ids": [
            row["recipe_id"] for row in reconciled if row.get("status") == "failed"
        ],
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
