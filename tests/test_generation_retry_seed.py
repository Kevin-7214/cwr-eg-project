from __future__ import annotations

import pytest

from cwr_eg.runtime import _base_generation_seed


def test_generation_retry_seed_is_deterministic_and_versioned() -> None:
    recipe = {"seed": 20260815, "recipe_id": "base-clean-example"}
    original = _base_generation_seed(recipe)
    retry = _base_generation_seed(recipe, 1)
    assert original == _base_generation_seed(recipe, 0)
    assert retry == _base_generation_seed(recipe, 1)
    assert retry != original


def test_generation_retry_seed_rejects_negative_index() -> None:
    with pytest.raises(ValueError, match="cannot be negative"):
        _base_generation_seed({"seed": 1, "recipe_id": "r"}, -1)
