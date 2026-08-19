from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess

from cwr_eg.hashing import content_hash, sha256_file
from cwr_eg.manifest import read_jsonl


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _record(path: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(PROJECT_ROOT).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _git_head(path: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    fixed_files = [
        PROJECT_ROOT / "configs" / name
        for name in (
            "intermediate.yaml",
            "intermediate_baselines.yaml",
            "intermediate_training_matrix.yaml",
            "registered_registry.yaml",
        )
    ] + [
        PROJECT_ROOT / "manifests" / name
        for name in (
            "intermediate_parents.jsonl",
            "intermediate_recipes.jsonl",
            "intermediate_data_manifest.json",
            "intermediate_canary_parents.jsonl",
            "intermediate_canary_recipes.jsonl",
            "intermediate_canary_manifest.json",
            "pilot_parents.jsonl",
            "corpus_registry.json",
            "model_registry.json",
            "repository_registry.json",
        )
    ] + [
        PROJECT_ROOT / "protocol.md",
        PROJECT_ROOT / "environment.md",
        PROJECT_ROOT / "pyproject.toml",
    ]
    code_files = sorted((PROJECT_ROOT / "src" / "cwr_eg").glob("*.py"))
    test_files = sorted((PROJECT_ROOT / "tests").glob("test_*.py"))
    if any(not path.is_file() for path in fixed_files):
        raise FileNotFoundError("An intermediate freeze input is missing")

    parents = read_jsonl(PROJECT_ROOT / "manifests" / "intermediate_parents.jsonl")
    recipes = read_jsonl(PROJECT_ROOT / "manifests" / "intermediate_recipes.jsonl")
    parent_split_hashes = {
        split: content_hash(
            sorted(
                (str(row["parent_id"]), str(row["text_sha256"]))
                for row in parents
                if row["split"] == split
            )
        )
        for split in ("train", "dev", "calibration", "test")
    }
    recipe_split_hashes = {
        split: content_hash(
            sorted(str(row["recipe_id"]) for row in recipes if row["split"] == split)
        )
        for split in ("train", "dev", "calibration", "test")
    }
    markllm_path = PROJECT_ROOT / "external" / "MarkLLM"
    payload: dict[str, object] = {
        "manifest_version": "intermediate-freeze-v1",
        "profile": "rtx5060-24h-intermediate",
        "seed": 20260815,
        "experiment_executed": False,
        "test_sealed": True,
        "parent_counts": dict(Counter(str(row["split"]) for row in parents)),
        "recipe_counts": dict(Counter(str(row["kind"]) for row in recipes)),
        "parent_split_content_hashes": parent_split_hashes,
        "recipe_split_content_hashes": recipe_split_hashes,
        "files": [_record(path) for path in fixed_files],
        "code_files": [_record(path) for path in code_files],
        "test_files": [_record(path) for path in test_files],
        "repository_state": {
            "project_base_head": _git_head(PROJECT_ROOT),
            "markllm_head": _git_head(markllm_path),
        },
        "expected_scoring_objects": 8250,
        "expected_full_tensor_batches": {"train": 150, "dev": 50},
        "expected_full_tensor_shards": {"train": 10, "dev": 4},
    }
    payload["freeze_content_hash"] = content_hash(payload)
    target = PROJECT_ROOT / "manifests" / "intermediate_freeze_manifest.json"
    if args.verify_only:
        if not target.is_file():
            raise FileNotFoundError("The intermediate freeze manifest is missing")
        current = json.loads(target.read_text(encoding="utf-8"))
        if current != payload:
            raise RuntimeError("Intermediate freeze manifest drift detected")
        print(
            json.dumps(
                {"path": str(target), "sha256": sha256_file(target), "verified": True},
                indent=2,
            )
        )
        return 0
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)
    print(json.dumps({"path": str(target), "sha256": sha256_file(target)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
