from __future__ import annotations

from cwr_eg.generated_data import assemble_generated_documents
from cwr_eg.hashing import sha256_text
from cwr_eg.manifest import read_jsonl, write_jsonl


def _generated(recipe: dict, text: str, intervals: list[list[int]]) -> dict:
    return {
        **recipe,
        "status": "generated",
        "text": text,
        "text_sha256": sha256_text(text),
        "watermark_intervals": intervals,
    }


def test_assembly_forbids_silent_loss_and_excludes_train_mixed_features(tmp_path) -> None:
    recipes = [
        {
            "recipe_id": "base",
            "kind": "base_generation",
            "parent_ids": ["p1"],
            "split": "train",
        },
        {
            "recipe_id": "attack",
            "kind": "matched_attack",
            "parent_ids": ["p1"],
            "split": "train",
        },
        {
            "recipe_id": "mixed-train",
            "kind": "mixed_document",
            "parent_ids": ["p1", "p2"],
            "split": "train",
        },
        {
            "recipe_id": "mixed-dev",
            "kind": "mixed_document",
            "parent_ids": ["p3", "p4"],
            "split": "dev",
        },
    ]
    recipe_path = tmp_path / "recipes.jsonl"
    generated_path = tmp_path / "generated.jsonl"
    write_jsonl(recipe_path, recipes)
    write_jsonl(
        generated_path,
        [
            _generated(recipes[0], "abcd", []),
            _generated(recipes[1], "abcd", []),
            _generated(recipes[2], "abcd", [[0, 2], [2, 4]]),
            _generated(recipes[3], "abcd", [[0, 2], [2, 4]]),
        ],
    )
    result = assemble_generated_documents(
        recipe_manifest=recipe_path,
        input_paths=[generated_path],
        output_path=tmp_path / "all.jsonl",
        feature_documents_path=tmp_path / "features.jsonl",
    )
    assert result["recipes"] == 4
    assert result["feature_documents"] == 3
    assert result["train_mixed_excluded"] == 1
    assert {row["recipe_id"] for row in read_jsonl(tmp_path / "features.jsonl")} == {
        "base",
        "attack",
        "mixed-dev",
    }
