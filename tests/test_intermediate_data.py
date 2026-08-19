from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

from cwr_eg.config import load_yaml, validate_experiment_config
from cwr_eg.data_prep import ATTACKS, SOURCE_METADATA, WATERMARK_FAMILIES, _intermediate_recipe_rows
from cwr_eg.hashing import sha256_file
from cwr_eg.manifest import read_jsonl


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _parents() -> list[dict[str, object]]:
    counts = {"train": 75, "dev": 25, "calibration": 50, "test": 50}
    rows: list[dict[str, object]] = []
    for source, metadata in SOURCE_METADATA.items():
        index = 0
        for split, count in counts.items():
            for _ in range(count):
                rows.append(
                    {
                        "parent_id": f"{source}-{index:03d}",
                        "source": source,
                        "language": metadata["language"],
                        "split": split,
                    }
                )
                index += 1
    return rows


def test_intermediate_config_is_frozen() -> None:
    validate_experiment_config(load_yaml(PROJECT_ROOT / "configs" / "intermediate.yaml"))


def test_intermediate_recipe_balance_and_pairing() -> None:
    parents = _parents()
    recipes = _intermediate_recipe_rows(parents, 20260815)
    assert Counter(row["kind"] for row in recipes) == {
        "base_generation": 4000,
        "matched_attack": 4000,
        "mixed_document": 400,
    }
    assert len({row["recipe_id"] for row in recipes}) == 8400

    key_groups: dict[tuple[str, str, str], Counter[str]] = defaultdict(Counter)
    for row in recipes:
        if row["kind"] == "base_generation" and row["watermark_family"] is not None:
            key_groups[(row["source"], row["split"], row["watermark_family"])][
                row["key_id"]
            ] += 1
    assert all(max(counts.values()) - min(counts.values()) <= 1 for counts in key_groups.values())
    assert len(key_groups) == len(SOURCE_METADATA) * 4 * len(WATERMARK_FAMILIES)

    attack_groups: dict[tuple[str, str, str], Counter[str]] = defaultdict(Counter)
    for row in recipes:
        if row["kind"] == "matched_attack":
            attack_groups[(row["source"], row["split"], row["base_variant"])][
                row["attack_id"]
            ] += 1
    assert all(set(counts) == set(ATTACKS) for counts in attack_groups.values())
    assert all(max(counts.values()) - min(counts.values()) <= 1 for counts in attack_groups.values())

    parent_lookup = {row["parent_id"]: row for row in parents}
    mixed_parent_ids: list[str] = []
    for row in recipes:
        if row["kind"] != "mixed_document":
            continue
        components = [parent_lookup[parent_id] for parent_id in row["parent_ids"]]
        assert len({item["language"] for item in components}) == 1
        assert len({item["split"] for item in components}) == 1
        mixed_parent_ids.extend(row["parent_ids"])
    assert Counter(mixed_parent_ids) == Counter({parent_id: 1 for parent_id in parent_lookup})


def test_frozen_intermediate_and_canary_manifests_are_complete() -> None:
    parents = read_jsonl(PROJECT_ROOT / "manifests" / "intermediate_parents.jsonl")
    recipes = read_jsonl(PROJECT_ROOT / "manifests" / "intermediate_recipes.jsonl")
    pilot_ids = {
        row["parent_id"]
        for row in read_jsonl(PROJECT_ROOT / "manifests" / "pilot_parents.jsonl")
    }
    assert len(parents) == len({row["parent_id"] for row in parents}) == 800
    assert not pilot_ids.intersection(row["parent_id"] for row in parents)
    assert Counter(row["split"] for row in parents) == {
        "train": 300,
        "dev": 100,
        "calibration": 200,
        "test": 200,
    }
    assert Counter(row["kind"] for row in recipes) == {
        "base_generation": 4000,
        "matched_attack": 4000,
        "mixed_document": 400,
    }
    assert len({row["recipe_id"] for row in recipes}) == 8400
    assert 8400 - sum(
        row["kind"] == "mixed_document" and row["split"] == "train"
        for row in recipes
    ) == 8250

    canary_parents = read_jsonl(
        PROJECT_ROOT / "manifests" / "intermediate_canary_parents.jsonl"
    )
    canary_recipes = read_jsonl(
        PROJECT_ROOT / "manifests" / "intermediate_canary_recipes.jsonl"
    )
    assert len(canary_parents) == 80
    assert {row["split"] for row in canary_parents} == {"train", "dev"}
    assert Counter(row["kind"] for row in canary_recipes) == {
        "base_generation": 400,
        "matched_attack": 400,
        "mixed_document": 40,
    }
    manifest = load_yaml(PROJECT_ROOT / "configs" / "intermediate.yaml")
    assert sha256_file(PROJECT_ROOT / manifest["data"]["parent_manifest"]) == (
        "02fd389173f76e2113e40601e1be8324d9a3359fb3b58a11fc48eeb0aa87276d"
    )
