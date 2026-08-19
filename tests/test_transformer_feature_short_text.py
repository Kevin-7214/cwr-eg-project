from __future__ import annotations

import torch

from cwr_eg.transformer_features import _prepend_eos_for_single_token_rows


def test_single_token_rows_receive_eos_context_without_changing_offsets() -> None:
    encoded = {
        "input_ids": torch.tensor([[42], [17]]),
        "attention_mask": torch.tensor([[1], [1]]),
    }
    offsets = [[[0, 1]], [[0, 2]]]
    changed = _prepend_eos_for_single_token_rows(
        encoded, offsets, eos_token_id=99
    )
    assert changed == {0, 1}
    assert encoded["input_ids"].tolist() == [[99, 42], [99, 17]]
    assert encoded["attention_mask"].tolist() == [[1, 1], [1, 1]]
    assert offsets == [[[0, 0], [0, 1]], [[0, 0], [0, 2]]]


def test_regular_rows_are_unchanged() -> None:
    encoded = {
        "input_ids": torch.tensor([[42, 43]]),
        "attention_mask": torch.tensor([[1, 1]]),
    }
    offsets = [[[0, 1], [1, 2]]]
    changed = _prepend_eos_for_single_token_rows(
        encoded, offsets, eos_token_id=99
    )
    assert changed == set()
    assert encoded["input_ids"].tolist() == [[42, 43]]
    assert offsets == [[[0, 1], [1, 2]]]
