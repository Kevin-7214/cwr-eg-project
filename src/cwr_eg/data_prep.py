from __future__ import annotations

from collections import defaultdict
import heapq
import json
from pathlib import Path
from typing import Any

from cwr_eg.hashing import sha256_text
from cwr_eg.manifest import audit_parent_splits, file_record, write_jsonl


PILOT_SOURCES = {
    "c4_en": {"language": "en", "license": "odc-by"},
    "wikipedia_en": {
        "language": "en",
        "license": ["cc-by-sa-3.0", "gfdl"],
    },
    "thucnews_zh": {"language": "zh", "license": "apache-2.0"},
    "wikipedia_zh": {
        "language": "zh",
        "license": ["cc-by-sa-3.0", "gfdl"],
    },
}

SPLIT_SLOTS = ("train", "train", "train", "train", "dev", "dev", "calibration", "test")
WATERMARK_FAMILIES = ("kgw", "unigram", "unbiased", "synthid")
ATTACKS = ("paraphrase", "translation_roundtrip", "copy_edit", "truncation")


def _stable_score(seed: int, parent_id: str) -> int:
    return int(sha256_text(f"{seed}:{parent_id}"), 16)


def _select_source_rows(
    path: Path,
    *,
    count: int,
    minimum_characters: int,
    maximum_characters: int,
    seed: int,
) -> list[dict[str, Any]]:
    selected: list[tuple[int, int, dict[str, Any]]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            text = str(row.get("text", ""))
            parent_id = str(row.get("parent_id", ""))
            if len(text) < minimum_characters or not parent_id:
                continue
            score = _stable_score(seed, parent_id)
            candidate = (
                -score,
                line_number,
                {
                    "parent_id": parent_id,
                    "text": text[:maximum_characters],
                    "source_text_sha256": sha256_text(text),
                    "source_line": line_number,
                    "genre": row.get("genre"),
                },
            )
            if len(selected) < count:
                heapq.heappush(selected, candidate)
            elif candidate > selected[0]:
                heapq.heapreplace(selected, candidate)
    if len(selected) != count:
        raise ValueError(f"Only found {len(selected)} eligible rows in {path}")
    return [item[2] for item in sorted(selected, key=lambda item: -item[0])]


def _recipe_rows(parents: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    recipes: list[dict[str, Any]] = []
    base_recipes: list[dict[str, Any]] = []
    for parent in parents:
        parent_id = str(parent["parent_id"])
        clean = {
            "recipe_id": "base-clean-" + sha256_text(parent_id)[:16],
            "split": parent["split"],
            "parent_ids": [parent_id],
            "kind": "base_generation",
            "watermark_family": None,
            "key_id": None,
            "seed": seed,
            "status": "planned_not_generated",
        }
        recipes.append(clean)
        base_recipes.append(clean)
        parity = _stable_score(seed, parent_id) % 2
        for family in WATERMARK_FAMILIES:
            key_id = f"{family}_key_{'a' if parity == 0 else 'b'}"
            item = {
                "recipe_id": f"base-{family}-" + sha256_text(parent_id)[:16],
                "split": parent["split"],
                "parent_ids": [parent_id],
                "kind": "base_generation",
                "watermark_family": family,
                "key_id": key_id,
                "seed": seed,
                "status": "planned_not_generated",
            }
            recipes.append(item)
            base_recipes.append(item)

    for index, base in enumerate(base_recipes):
        recipes.append(
            {
                "recipe_id": "attack-" + base["recipe_id"],
                "split": base["split"],
                "parent_ids": base["parent_ids"],
                "kind": "matched_attack",
                "base_recipe_id": base["recipe_id"],
                "attack_id": ATTACKS[index % len(ATTACKS)],
                "boundary_quality": "weak"
                if ATTACKS[index % len(ATTACKS)] in {"paraphrase", "translation_roundtrip"}
                else "exact",
                "seed": seed,
                "status": "planned_not_generated",
            }
        )

    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for parent in parents:
        by_split[str(parent["split"])].append(parent)
    mixed_index = 0
    for split in ("train", "dev", "calibration", "test"):
        items = sorted(by_split[split], key=lambda row: _stable_score(seed + 1, row["parent_id"]))
        for left, right in zip(items[::2], items[1::2], strict=True):
            first_family = WATERMARK_FAMILIES[mixed_index % len(WATERMARK_FAMILIES)]
            second_family = WATERMARK_FAMILIES[(mixed_index + 1) % len(WATERMARK_FAMILIES)]
            recipes.append(
                {
                    "recipe_id": f"mixed-{mixed_index:03d}",
                    "split": split,
                    "parent_ids": [left["parent_id"], right["parent_id"]],
                    "kind": "mixed_document",
                    "components": [
                        {"watermark_family": first_family, "key_slot": "a"},
                        {"watermark_family": second_family, "key_slot": "b"},
                    ],
                    "overlap_mode": "adjacent",
                    "seed": seed,
                    "status": "planned_not_generated",
                }
            )
            mixed_index += 1
    return recipes


def prepare_pilot_data(
    *,
    legacy_corpus_dir: str | Path,
    output_dir: str | Path,
    seed: int = 20260813,
    minimum_characters: int = 2000,
    maximum_characters: int = 6000,
) -> dict[str, Any]:
    corpus_dir = Path(legacy_corpus_dir)
    target = Path(output_dir)
    parents: list[dict[str, Any]] = []
    source_files: list[dict[str, Any]] = []
    for source_name, metadata in PILOT_SOURCES.items():
        source_path = corpus_dir / f"{source_name}.jsonl"
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        rows = _select_source_rows(
            source_path,
            count=8,
            minimum_characters=minimum_characters,
            maximum_characters=maximum_characters,
            seed=seed,
        )
        for split, row in zip(SPLIT_SLOTS, rows, strict=True):
            text = row.pop("text")
            parents.append(
                {
                    **row,
                    "source": source_name,
                    "language": metadata["language"],
                    "license": metadata["license"],
                    "split": split,
                    "text": text,
                    "text_sha256": sha256_text(text),
                    "selection_seed": seed,
                    "selection_policy": "lowest_sha256_seed_parent_id",
                }
            )
        source_files.append(file_record(source_path))

    parents.sort(key=lambda row: (row["split"], row["source"], row["parent_id"]))
    audit = audit_parent_splits(
        parents,
        expected_counts={"calibration": 4, "dev": 8, "test": 4, "train": 16},
    )
    recipes = _recipe_rows(parents, seed)
    parent_path = target / "pilot_parents.jsonl"
    recipe_path = target / "pilot_recipes.jsonl"
    write_jsonl(parent_path, parents)
    write_jsonl(recipe_path, recipes)
    manifest = {
        "manifest_version": "0.1.0-pre-experiment",
        "seed": seed,
        "minimum_characters": minimum_characters,
        "maximum_characters": maximum_characters,
        "parent_audit": audit,
        "recipe_counts": {
            "base_generation": sum(row["kind"] == "base_generation" for row in recipes),
            "matched_attack": sum(row["kind"] == "matched_attack" for row in recipes),
            "mixed_document": sum(row["kind"] == "mixed_document" for row in recipes),
        },
        "source_files": source_files,
        "outputs": [file_record(parent_path), file_record(recipe_path)],
        "experiment_executed": False,
    }
    manifest_path = target / "pilot_data_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest
