from __future__ import annotations

import numpy as np

from cwr_eg.baselines import (
    MahalanobisBaseline,
    direct_statistical_features,
    energy_ood,
    linear_evidence_fusion,
    maximum_softmax_ood,
    prototype_distance,
)


def test_reference_baseline_scores_are_finite_and_oriented() -> None:
    logits = np.asarray([[5.0, 0.0], [0.0, 0.0]])
    msp = maximum_softmax_ood(logits)
    assert msp[0] < msp[1]
    assert np.all(np.isfinite(energy_ood(logits)))
    distances = prototype_distance(np.asarray([[0.0, 0.0]]), np.asarray([[0.0, 0.0], [2.0, 2.0]]))
    assert distances.tolist() == [0.0]


def test_direct_statistics_and_unstructured_fusion() -> None:
    features = direct_statistical_features([1, 2], [0.5, 0.7], [2, 3], [2, 4])
    assert features.tolist() == [1.5, 0.6, 2.5, 1.5]
    fused = linear_evidence_fusion(np.asarray([1.0]), np.asarray([3.0]), 0.25)
    assert fused.tolist() == [2.5]


def test_mahalanobis_baseline_scores_near_known_centers_lower() -> None:
    values = np.asarray([[0.0, 0.0], [0.1, 0.0], [2.0, 2.0], [2.1, 2.0]])
    model = MahalanobisBaseline.fit(values, [0, 0, 1, 1])
    scores = model.score(np.asarray([[0.05, 0.0], [10.0, 10.0]]))
    assert scores[0] < scores[1]
