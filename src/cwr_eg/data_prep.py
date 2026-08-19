from __future__ import annotations

from collections import Counter, defaultdict
import heapq
import json
from pathlib import Path
from typing import Any, Mapping

from cwr_eg.hashing import sha256_text
from cwr_eg.manifest import audit_parent_splits, file_record, write_jsonl


SOURCE_METADATA = {
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

PILOT_SOURCES = SOURCE_METADATA

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
    excluded_parent_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    excluded = excluded_parent_ids or set()
    selected: list[tuple[int, int, dict[str, Any]]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            text = str(row.get("text", ""))
            parent_id = str(row.get("parent_id", ""))
            if len(text) < minimum_characters or not parent_id or parent_id in excluded:
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


def _split_source_rows(
    rows: list[dict[str, Any]], split_counts: Mapping[str, int]
) -> list[tuple[str, dict[str, Any]]]:
    ordered_splits = ("train", "dev", "calibration", "test")
    if set(split_counts) != set(ordered_splits):
        raise ValueError("split_counts must define Train, Dev, Calibration, and Test")
    if any(int(split_counts[name]) < 1 for name in ordered_splits):
        raise ValueError("Every split requires at least one parent per source")
    slots = [
        split
        for split in ordered_splits
        for _ in range(int(split_counts[split]))
    ]
    if len(rows) != len(slots):
        raise ValueError("Selected row count does not match split_counts")
    return list(zip(slots, rows, strict=True))


def _key_slot_by_parent(
    parents: list[dict[str, Any]], seed: int
) -> dict[str, str]:
    slots: dict[str, str] = {}
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for parent in parents:
        grouped[(str(parent["source"]), str(parent["split"]))].append(parent)
    for group, rows in sorted(grouped.items()):
        ordered = sorted(
            rows,
            key=lambda row: (
                _stable_score(seed + 11, f"{group[0]}:{group[1]}:{row['parent_id']}"),
                str(row["parent_id"]),
            ),
        )
        for index, row in enumerate(ordered):
            slots[str(row["parent_id"])] = "a" if index % 2 == 0 else "b"
    return slots


def _intermediate_recipe_rows(
    parents: list[dict[str, Any]], seed: int
) -> list[dict[str, Any]]:
    key_slots = _key_slot_by_parent(parents, seed)
    base_recipes: list[dict[str, Any]] = []
    grouped_parents: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for parent in parents:
        grouped_parents[(str(parent["source"]), str(parent["split"]))].append(parent)

    for (source, split), rows in sorted(grouped_parents.items()):
        ordered = sorted(
            rows,
            key=lambda row: (
                _stable_score(seed + 17, str(row["parent_id"])),
                str(row["parent_id"]),
            ),
        )
        for parent in ordered:
            parent_id = str(parent["parent_id"])
            common = {
                "split": split,
                "source": source,
                "language": parent["language"],
                "parent_ids": [parent_id],
                "kind": "base_generation",
                "seed": seed,
                "status": "planned_not_generated",
            }
            base_recipes.append(
                {
                    **common,
                    "recipe_id": "base-clean-" + sha256_text(parent_id)[:16],
                    "base_variant": "clean",
                    "watermark_family": None,
                    "key_id": None,
                }
            )
            for family in WATERMARK_FAMILIES:
                base_recipes.append(
                    {
                        **common,
                        "recipe_id": f"base-{family}-" + sha256_text(parent_id)[:16],
                        "base_variant": family,
                        "watermark_family": family,
                        "key_id": f"{family}_key_{key_slots[parent_id]}",
                    }
                )

    attacks: list[dict[str, Any]] = []
    base_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for base in base_recipes:
        base_groups[
            (str(base["source"]), str(base["split"]), str(base["base_variant"]))
        ].append(base)
    for group, rows in sorted(base_groups.items()):
        ordered = sorted(
            rows,
            key=lambda row: (
                _stable_score(seed + 23, str(row["recipe_id"])),
                str(row["recipe_id"]),
            ),
        )
        rotation = _stable_score(seed + 29, ":".join(group)) % len(ATTACKS)
        for index, base in enumerate(ordered):
            attack_id = ATTACKS[(rotation + index) % len(ATTACKS)]
            attacks.append(
                {
                    "recipe_id": "attack-" + str(base["recipe_id"]),
                    "split": base["split"],
                    "source": base["source"],
                    "language": base["language"],
                    "parent_ids": base["parent_ids"],
                    "kind": "matched_attack",
                    "base_recipe_id": base["recipe_id"],
                    "base_variant": base["base_variant"],
                    "watermark_family": base["watermark_family"],
                    "key_id": base["key_id"],
                    "attack_id": attack_id,
                    "boundary_quality": "weak"
                    if attack_id in {"paraphrase", "translation_roundtrip"}
                    else "exact",
                    "seed": seed,
                    "status": "planned_not_generated",
                }
            )

    mixed: list[dict[str, Any]] = []
    mixed_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for parent in parents:
        mixed_groups[(str(parent["language"]), str(parent["split"]))].append(parent)
    mixed_index = 0
    for (language, split), rows in sorted(mixed_groups.items()):
        ordered = sorted(
            rows,
            key=lambda row: (
                _stable_score(seed + 31, str(row["parent_id"])),
                str(row["parent_id"]),
            ),
        )
        if len(ordered) % 2:
            raise ValueError("Mixed pairing requires even language-by-split parent counts")
        for pair_index in range(0, len(ordered), 2):
            left, right = ordered[pair_index : pair_index + 2]
            first_family = WATERMARK_FAMILIES[mixed_index % len(WATERMARK_FAMILIES)]
            second_family = WATERMARK_FAMILIES[(mixed_index + 1) % len(WATERMARK_FAMILIES)]
            mixed.append(
                {
                    "recipe_id": f"mixed-{mixed_index:04d}",
                    "split": split,
                    "source": "mixed",
                    "language": language,
                    "parent_ids": [left["parent_id"], right["parent_id"]],
                    "kind": "mixed_document",
                    "components": [
                        {
                            "watermark_family": first_family,
                            "key_slot": key_slots[str(left["parent_id"])],
                        },
                        {
                            "watermark_family": second_family,
                            "key_slot": key_slots[str(right["parent_id"])],
                        },
                    ],
                    "overlap_mode": "adjacent",
                    "seed": seed,
                    "status": "planned_not_generated",
                }
            )
            mixed_index += 1
    return sorted(base_recipes + attacks + mixed, key=lambda row: str(row["recipe_id"]))


def _load_excluded_parent_ids(path: str | Path | None) -> set[str]:
    if path is None:
        return set()
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    excluded: set[str] = set()
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            parent_id = str(row.get("parent_id", ""))
            if not parent_id:
                raise ValueError(f"Missing parent_id in exclusion row {line_number}")
            excluded.add(parent_id)
    return excluded


def prepare_intermediate_data(
    *,
    legacy_corpus_dir: str | Path,
    output_dir: str | Path,
    excluded_parent_manifest: str | Path,
    seed: int = 20260815,
    source_count: int = 200,
    split_counts_per_source: Mapping[str, int] | None = None,
    minimum_characters: int = 2000,
    maximum_characters: int = 6000,
) -> dict[str, Any]:
    split_counts = dict(
        split_counts_per_source
        or {"train": 75, "dev": 25, "calibration": 50, "test": 50}
    )
    if sum(int(value) for value in split_counts.values()) != source_count:
        raise ValueError("Per-source split counts must sum to source_count")
    corpus_dir = Path(legacy_corpus_dir)
    target = Path(output_dir)
    excluded = _load_excluded_parent_ids(excluded_parent_manifest)
    parents: list[dict[str, Any]] = []
    source_files: list[dict[str, Any]] = []
    for source_name, metadata in SOURCE_METADATA.items():
        source_path = corpus_dir / f"{source_name}.jsonl"
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        rows = _select_source_rows(
            source_path,
            count=source_count,
            minimum_characters=minimum_characters,
            maximum_characters=maximum_characters,
            seed=seed,
            excluded_parent_ids=excluded,
        )
        for split, row in _split_source_rows(rows, split_counts):
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
                    "selection_policy": "lowest_sha256_seed_parent_id_excluding_pilot",
                }
            )
        source_files.append(file_record(source_path))

    parents.sort(key=lambda row: (row["split"], row["source"], row["parent_id"]))
    expected_splits = {
        split: int(count) * len(SOURCE_METADATA) for split, count in split_counts.items()
    }
    audit = audit_parent_splits(parents, expected_counts=expected_splits)
    if excluded.intersection(str(row["parent_id"]) for row in parents):
        raise RuntimeError("Intermediate parent manifest overlaps the exclusion manifest")
    recipes = _intermediate_recipe_rows(parents, seed)
    counts = Counter(str(row["kind"]) for row in recipes)
    expected_recipe_counts = {
        "base_generation": len(parents) * 5,
        "matched_attack": len(parents) * 5,
        "mixed_document": len(parents) // 2,
    }
    if dict(counts) != expected_recipe_counts:
        raise RuntimeError(f"Recipe counts {dict(counts)} != {expected_recipe_counts}")

    parent_path = target / "intermediate_parents.jsonl"
    recipe_path = target / "intermediate_recipes.jsonl"
    write_jsonl(parent_path, parents)
    write_jsonl(recipe_path, recipes)
    manifest = {
        "manifest_version": "intermediate-v1",
        "profile": "rtx5060-24h-intermediate",
        "seed": seed,
        "minimum_characters": minimum_characters,
        "maximum_characters": maximum_characters,
        "source_count": source_count,
        "split_counts_per_source": split_counts,
        "parent_audit": audit,
        "excluded_parent_manifest": file_record(excluded_parent_manifest),
        "excluded_parent_count": len(excluded),
        "overlap_with_excluded": 0,
        "recipe_counts": expected_recipe_counts,
        "source_files": source_files,
        "outputs": [file_record(parent_path), file_record(recipe_path)],
        "experiment_executed": False,
    }
    manifest_path = target / "intermediate_data_manifest.json"
    temporary = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(manifest_path)
    return manifest


def prepare_intermediate_canary(
    *,
    parent_manifest: str | Path,
    recipe_manifest: str | Path,
    output_dir: str | Path,
    seed: int = 20260815,
    train_parents_per_source: int = 15,
    dev_parents_per_source: int = 5,
) -> dict[str, Any]:
    from cwr_eg.manifest import read_jsonl

    parents = read_jsonl(parent_manifest)
    recipes = read_jsonl(recipe_manifest)
    parent_by_id = {str(row["parent_id"]): row for row in parents}
    if len(parent_by_id) != len(parents):
        raise ValueError("Intermediate parent ids must be unique")
    mixed_by_group: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for recipe in recipes:
        if recipe["kind"] == "mixed_document" and recipe["split"] in {"train", "dev"}:
            mixed_by_group[(str(recipe["language"]), str(recipe["split"]))].append(recipe)

    sources_by_language: dict[str, list[str]] = defaultdict(list)
    for source, metadata in SOURCE_METADATA.items():
        sources_by_language[str(metadata["language"])].append(source)
    selected_mixed: list[dict[str, Any]] = []
    selected_parent_ids: set[str] = set()
    targets = {"train": train_parents_per_source, "dev": dev_parents_per_source}
    for language in ("en", "zh"):
        first_source, second_source = sorted(sources_by_language[language])
        for split in ("train", "dev"):
            target_per_source = targets[split]
            categories: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
            for recipe in mixed_by_group[(language, split)]:
                source_pair = tuple(
                    sorted(
                        str(parent_by_id[str(parent_id)]["source"])
                        for parent_id in recipe["parent_ids"]
                    )
                )
                categories[source_pair].append(recipe)
            for rows in categories.values():
                rows.sort(
                    key=lambda row: (
                        _stable_score(seed + 41, str(row["recipe_id"])),
                        str(row["recipe_id"]),
                    )
                )
            same_first = (first_source, first_source)
            cross = (first_source, second_source)
            same_second = (second_source, second_source)
            chosen_counts: tuple[int, int] | None = None
            for same_count in range(target_per_source // 2 + 1):
                cross_count = target_per_source - 2 * same_count
                if (
                    len(categories[same_first]) >= same_count
                    and len(categories[same_second]) >= same_count
                    and len(categories[cross]) >= cross_count
                ):
                    chosen_counts = (same_count, cross_count)
                    break
            if chosen_counts is None:
                raise RuntimeError(
                    f"Cannot construct a balanced reusable canary for {language}:{split}"
                )
            same_count, cross_count = chosen_counts
            chosen = (
                categories[same_first][:same_count]
                + categories[cross][:cross_count]
                + categories[same_second][:same_count]
            )
            selected_mixed.extend(chosen)
            for recipe in chosen:
                selected_parent_ids.update(str(item) for item in recipe["parent_ids"])

    selected_parents = [
        row for row in parents if str(row["parent_id"]) in selected_parent_ids
    ]
    expected_parent_count = len(SOURCE_METADATA) * (
        train_parents_per_source + dev_parents_per_source
    )
    if len(selected_parents) != expected_parent_count:
        raise RuntimeError("Canary parent selection did not produce the frozen size")
    source_split_counts = Counter(
        (str(row["source"]), str(row["split"])) for row in selected_parents
    )
    for source in SOURCE_METADATA:
        if source_split_counts[(source, "train")] != train_parents_per_source:
            raise RuntimeError("Canary Train source balance failed")
        if source_split_counts[(source, "dev")] != dev_parents_per_source:
            raise RuntimeError("Canary Dev source balance failed")
    selected_mixed_ids = {str(row["recipe_id"]) for row in selected_mixed}
    selected_recipes = [
        row
        for row in recipes
        if (
            row["kind"] in {"base_generation", "matched_attack"}
            and str(row["parent_ids"][0]) in selected_parent_ids
        )
        or str(row["recipe_id"]) in selected_mixed_ids
    ]
    expected_counts = {
        "base_generation": expected_parent_count * 5,
        "matched_attack": expected_parent_count * 5,
        "mixed_document": expected_parent_count // 2,
    }
    if Counter(str(row["kind"]) for row in selected_recipes) != expected_counts:
        raise RuntimeError("Canary recipe counts do not match the frozen 10% scope")
    if any(str(row["split"]) not in {"train", "dev"} for row in selected_recipes):
        raise RuntimeError("Canary must not unseal Calibration or Test")

    target = Path(output_dir)
    parent_path = target / "intermediate_canary_parents.jsonl"
    recipe_path = target / "intermediate_canary_recipes.jsonl"
    write_jsonl(parent_path, selected_parents)
    write_jsonl(recipe_path, selected_recipes)
    manifest = {
        "manifest_version": "intermediate-canary-v1",
        "seed": seed,
        "selection_unit": "reusable_full_manifest_mixed_pair",
        "allowed_splits": ["train", "dev"],
        "calibration_and_test_sealed": True,
        "parent_count": expected_parent_count,
        "recipe_counts": expected_counts,
        "source_split_counts": {
            f"{source}:{split}": source_split_counts[(source, split)]
            for source in sorted(SOURCE_METADATA)
            for split in ("train", "dev")
        },
        "full_inputs": [file_record(parent_manifest), file_record(recipe_manifest)],
        "outputs": [file_record(parent_path), file_record(recipe_path)],
        "experiment_executed": False,
    }
    manifest_path = target / "intermediate_canary_manifest.json"
    temporary = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(manifest_path)
    return manifest


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
