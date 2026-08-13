from __future__ import annotations

import numpy as np

from cwr_eg.counterfactual import build_pairing_masks


def test_pairing_masks_never_use_same_descendant_or_parent_as_negative() -> None:
    result = build_pairing_masks(
        ["a", "a", "b", "b"],
        ["clean", "kgw", "clean", "unigram"],
        [1.0, 0.8, 1.0, 0.6],
    )
    assert not np.any(np.diag(result.positive_mask))
    assert not np.any(np.diag(result.negative_mask))
    assert result.positive_mask[0, 1]
    assert not result.negative_mask[0, 1]
    assert result.negative_mask[0, 2]
    assert result.diagnostics["positive_pairs"] == 4
