from __future__ import annotations

import pytest

from cwr_eg.runtime import _inference_character_scores


def test_inference_accepts_checkpoint_character_logits() -> None:
    assert _inference_character_scores({"character_logits": [1.0, 2.0]}) == [1.0, 2.0]


def test_inference_prefers_legacy_character_scores() -> None:
    row = {"character_scores": [3.0], "character_logits": [4.0]}
    assert _inference_character_scores(row) == [3.0]


def test_inference_rejects_missing_character_scores() -> None:
    with pytest.raises(ValueError, match="no character score vector"):
        _inference_character_scores({})
