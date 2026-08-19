from __future__ import annotations

import numpy as np

from scripts.analyze_i05_dev import _binary_auc, _classification_metrics


def test_binary_auc_treats_four_watermark_families_as_positive() -> None:
    labels = np.asarray([0, 1, 2, 3, 4], dtype=np.int64)
    scores = np.asarray([0.0, 0.8, 0.7, 0.9, 1.0], dtype=np.float64)
    assert _binary_auc(labels, scores) == 1.0


def test_classification_metrics_keeps_all_five_classes() -> None:
    labels = np.asarray([0, 1, 2, 3, 4], dtype=np.int64)
    result = _classification_metrics(labels, labels.copy())
    assert result["macro_f1"] == 1.0
    assert set(result["per_class"]) == {"clean", "kgw", "unigram", "unbiased", "synthid"}
