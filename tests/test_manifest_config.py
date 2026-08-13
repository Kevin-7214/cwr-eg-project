from __future__ import annotations

from collections import Counter
from pathlib import Path

from cwr_eg.assets import audit_registry_files
from cwr_eg.config import load_yaml, validate_pilot_config
from cwr_eg.manifest import audit_parent_splits, read_jsonl


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_frozen_config_and_asset_registries() -> None:
    validate_pilot_config(load_yaml(PROJECT_ROOT / "configs" / "pilot.yaml"))
    assert audit_registry_files(PROJECT_ROOT) == {"ok": True, "errors": []}


def test_generated_pilot_manifests_are_isolated_and_complete() -> None:
    parents = read_jsonl(PROJECT_ROOT / "manifests" / "pilot_parents.jsonl")
    recipes = read_jsonl(PROJECT_ROOT / "manifests" / "pilot_recipes.jsonl")
    audit = audit_parent_splits(
        parents, {"calibration": 4, "dev": 8, "test": 4, "train": 16}
    )
    assert audit["unique_parent_ids"] == 32
    assert Counter(row["source"] for row in parents) == {
        "c4_en": 8,
        "wikipedia_en": 8,
        "thucnews_zh": 8,
        "wikipedia_zh": 8,
    }
    assert Counter(row["kind"] for row in recipes) == {
        "base_generation": 160,
        "matched_attack": 160,
        "mixed_document": 16,
    }
    parent_split = {row["parent_id"]: row["split"] for row in parents}
    assert all(
        parent_split[parent_id] == recipe["split"]
        for recipe in recipes
        for parent_id in recipe["parent_ids"]
    )
