from __future__ import annotations

import pytest

from cwr_eg.hashing import sha256_file, sha256_text
from cwr_eg.manifest import write_jsonl
from cwr_eg.runtime import _load_approved_generation_partial


def _recipe() -> dict:
    return {
        "recipe_id": "base-1",
        "kind": "base_generation",
        "parent_ids": ["parent-1"],
        "split": "train",
        "source": "source-1",
        "language": "en",
        "base_variant": "clean",
        "watermark_family": None,
        "key_id": None,
    }


def _row(recipe: dict) -> dict:
    text = "completed text"
    return {
        **recipe,
        "status": "generated",
        "text": text,
        "text_sha256": sha256_text(text),
    }


def test_generation_partial_requires_explicit_hash_approval(tmp_path) -> None:
    partial = tmp_path / "generated.jsonl.partial"
    write_jsonl(partial, [_row(_recipe())])
    with pytest.raises(RuntimeError, match="not hash-approved"):
        _load_approved_generation_partial(
            partial, expected_sha256=None, expected_count=None, recipes=[_recipe()]
        )


def test_generation_partial_hash_count_and_recipe_are_bound(tmp_path) -> None:
    recipe = _recipe()
    partial = tmp_path / "generated.jsonl.partial"
    write_jsonl(partial, [_row(recipe)])
    rows = _load_approved_generation_partial(
        partial,
        expected_sha256=sha256_file(partial),
        expected_count=1,
        recipes=[recipe],
    )
    assert [row["recipe_id"] for row in rows] == ["base-1"]
    with pytest.raises(RuntimeError, match="resumed-document count"):
        _load_approved_generation_partial(
            partial,
            expected_sha256=sha256_file(partial),
            expected_count=2,
            recipes=[recipe],
        )


def test_generation_partial_rejects_hash_or_recipe_drift(tmp_path) -> None:
    recipe = _recipe()
    partial = tmp_path / "generated.jsonl.partial"
    write_jsonl(partial, [_row(recipe)])
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        _load_approved_generation_partial(
            partial, expected_sha256="0" * 64, expected_count=1, recipes=[recipe]
        )
    changed = {**recipe, "split": "dev"}
    with pytest.raises(RuntimeError, match="frozen recipe field"):
        _load_approved_generation_partial(
            partial,
            expected_sha256=sha256_file(partial),
            expected_count=1,
            recipes=[changed],
        )


def test_generation_partial_rejects_duplicate_or_failed_rows(tmp_path) -> None:
    recipe = _recipe()
    partial = tmp_path / "generated.jsonl.partial"
    row = _row(recipe)
    write_jsonl(partial, [row, row])
    with pytest.raises(RuntimeError, match="duplicate recipe id"):
        _load_approved_generation_partial(
            partial,
            expected_sha256=sha256_file(partial),
            expected_count=2,
            recipes=[recipe],
        )
    write_jsonl(partial, [{**row, "status": "failed"}])
    with pytest.raises(RuntimeError, match="Only approved completed rows"):
        _load_approved_generation_partial(
            partial,
            expected_sha256=sha256_file(partial),
            expected_count=1,
            recipes=[recipe],
        )


def test_generation_partial_can_resume_explicit_failures_when_scoped(tmp_path) -> None:
    recipe = _recipe()
    partial = tmp_path / "generated.jsonl.partial"
    failed = {
        **recipe,
        "status": "failed",
        "failure_type": "RuntimeError",
        "failure_message": "deterministic empty output",
    }
    write_jsonl(partial, [failed])
    rows = _load_approved_generation_partial(
        partial,
        expected_sha256=sha256_file(partial),
        expected_count=1,
        recipes=[recipe],
        allow_failed_rows=True,
    )
    assert rows == [failed]

    write_jsonl(partial, [{**failed, "failure_message": ""}])
    with pytest.raises(RuntimeError, match="invalid failure provenance"):
        _load_approved_generation_partial(
            partial,
            expected_sha256=sha256_file(partial),
            expected_count=1,
            recipes=[recipe],
            allow_failed_rows=True,
        )
