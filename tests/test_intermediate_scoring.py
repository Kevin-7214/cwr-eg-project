from __future__ import annotations

import numpy as np
import pytest
import torch

from cwr_eg.modeling import CwrEgModel, CwrEgModelConfig
from cwr_eg.scoring import (
    FeatureDocument,
    map_token_logits_to_characters,
    nearest_valid_fill,
    score_feature_document,
)


def test_checkpoint_logits_map_to_unicode_codepoints_before_nearest_fill() -> None:
    logits = np.asarray([1.0, 3.0])
    offsets = np.asarray([[0, 2], [3, 4]])
    values, coverage, covered = map_token_logits_to_characters(
        logits, offsets, np.asarray([True, True]), raw_length=5
    )
    assert coverage == pytest.approx(3 / 5)
    assert covered.tolist() == [True, True, False, True, False]
    assert values.tolist() == [1.0, 1.0, 1.0, 3.0, 3.0]


def test_nearest_fill_rejects_zero_mapping_coverage() -> None:
    with pytest.raises(ValueError, match="At least one"):
        nearest_valid_fill(np.zeros(3), np.zeros(3, dtype=bool))


def test_generic_residual_uses_serialized_train_null_prototype() -> None:
    config = CwrEgModelConfig(
        view_dims={"proxy": 4, "representation": 3, "perturbation": 4, "validity": 4},
        hidden_dim=2,
        invariant_dim=2,
        private_dim=2,
        dropout=0.0,
    )
    model = CwrEgModel(config)
    for parameter in model.parameters():
        parameter.data.zero_()
    feature = FeatureDocument(
        views={
            name: np.zeros((2, dimension), dtype=np.float32)
            for name, dimension in config.view_dims.items()
        },
        valid_mask=np.ones(2, dtype=bool),
        offsets=np.asarray([[0, 1], [1, 2]]),
    )
    result = score_feature_document(
        [model],
        feature,
        raw_length=2,
        positions=2,
        device=torch.device("cpu"),
        null_prototypes=[torch.ones(2)],
    )
    assert result["generic_residual_score"] == pytest.approx(2**0.5)
    assert result["generic_residual_source"] == "train-null-prototype-euclidean-v1"
