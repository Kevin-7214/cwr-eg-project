from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from cwr_eg.hashing import sha256_file
from cwr_eg.manifest import read_jsonl, write_jsonl
from cwr_eg.runtime import _load_approved_generation_partial


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--recipe-manifest", type=Path, required=True)
    parser.add_argument("--kind", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.manifest.exists():
        raise FileExistsError("Refusing to overwrite a partial seed or its manifest")
    recipes = [
        row
        for row in read_jsonl(args.recipe_manifest)
        if str(row["kind"]) == args.kind
    ]
    recipe_ids = {str(row["recipe_id"]) for row in recipes}
    source_rows = read_jsonl(args.source)
    selected = [row for row in source_rows if str(row["recipe_id"]) in recipe_ids]
    if len(selected) != len(source_rows):
        raise RuntimeError("The canary source contains rows outside the full frozen recipe set")
    write_jsonl(args.output, selected)
    output_sha256 = sha256_file(args.output)
    _load_approved_generation_partial(
        args.output,
        expected_sha256=output_sha256,
        expected_count=len(selected),
        recipes=recipes,
    )
    payload = {
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "kind": args.kind,
        "source_path": str(args.source),
        "source_sha256": sha256_file(args.source),
        "recipe_manifest": str(args.recipe_manifest),
        "recipe_manifest_sha256": sha256_file(args.recipe_manifest),
        "output_path": str(args.output),
        "output_sha256": output_sha256,
        "resumed_documents": len(selected),
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
