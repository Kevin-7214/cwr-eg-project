from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from cwr_eg.hashing import sha256_file, sha256_text
from cwr_eg.manifest import read_jsonl, write_jsonl


def _validate_generated_row(row: dict[str, Any], recipe: dict[str, Any]) -> None:
    if row.get("status") == "failed":
        if not row.get("failure_type") or not row.get("failure_message"):
            raise ValueError("Failed recipes require an explicit type and message")
        return
    if row.get("status") != "generated":
        raise ValueError("Every generated-data row must be generated or failed")
    text = str(row.get("text", ""))
    if not text or sha256_text(text) != str(row.get("text_sha256")):
        raise ValueError("Generated text is empty or has a mismatched hash")
    if str(row.get("kind")) != str(recipe["kind"]):
        raise ValueError("Generated row kind changed from its frozen recipe")
    if list(row.get("parent_ids", ())) != list(recipe["parent_ids"]):
        raise ValueError("Generated row parent ids changed from its frozen recipe")
    intervals = row.get("watermark_intervals", ())
    previous_end = 0
    for interval in intervals:
        start, end = (int(interval[0]), int(interval[1]))
        if start < previous_end or not 0 <= start < end <= len(text):
            raise ValueError("Generated watermark intervals are invalid")
        previous_end = end
    if recipe["kind"] == "mixed_document" and len(intervals) != 2:
        raise ValueError("Mixed documents require two exact component intervals")


def assemble_generated_documents(
    *,
    recipe_manifest: str | Path,
    input_paths: Iterable[str | Path],
    output_path: str | Path,
    feature_documents_path: str | Path,
) -> dict[str, Any]:
    recipes = read_jsonl(recipe_manifest)
    recipe_by_id = {str(row["recipe_id"]): row for row in recipes}
    if len(recipe_by_id) != len(recipes):
        raise ValueError("Frozen recipe ids must be unique")
    generated_by_id: dict[str, dict[str, Any]] = {}
    input_records = []
    for input_path in input_paths:
        path = Path(input_path)
        input_records.append({"path": str(path), "sha256": sha256_file(path)})
        for row in read_jsonl(path):
            recipe_id = str(row["recipe_id"])
            if recipe_id not in recipe_by_id:
                raise ValueError("Generated output contains a recipe outside the frozen manifest")
            if recipe_id in generated_by_id:
                raise ValueError("Generated output contains a duplicate recipe id")
            _validate_generated_row(row, recipe_by_id[recipe_id])
            generated_by_id[recipe_id] = row
    missing = sorted(set(recipe_by_id) - set(generated_by_id))
    if missing:
        raise RuntimeError(f"Generated output silently misses recipes: {missing[:5]}")
    ordered = [generated_by_id[str(recipe["recipe_id"])] for recipe in recipes]
    target = Path(output_path)
    feature_target = Path(feature_documents_path)
    if target.exists() or feature_target.exists():
        raise FileExistsError("Refusing to overwrite assembled generated data")
    write_jsonl(target, ordered)
    feature_rows = [
        row
        for row in ordered
        if row.get("status") == "generated"
        and not (row["kind"] == "mixed_document" and row["split"] == "train")
    ]
    write_jsonl(feature_target, feature_rows)
    statuses = Counter(str(row["status"]) for row in ordered)
    kinds = Counter(str(row["kind"]) for row in ordered)
    return {
        "recipes": len(recipes),
        "status_counts": dict(statuses),
        "kind_counts": dict(kinds),
        "feature_documents": len(feature_rows),
        "train_mixed_excluded": sum(
            row["kind"] == "mixed_document" and row["split"] == "train"
            for row in ordered
        ),
        "inputs": input_records,
        "output_path": str(target),
        "output_sha256": sha256_file(target),
        "feature_documents_path": str(feature_target),
        "feature_documents_sha256": sha256_file(feature_target),
    }
