from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class PairingResult:
    positive_mask: np.ndarray
    negative_mask: np.ndarray
    pair_weights: np.ndarray
    diagnostics: dict[str, Any]


def build_pairing_masks(
    parent_ids: Sequence[str],
    intervention_ids: Sequence[str],
    quality_weights: Sequence[float] | None = None,
) -> PairingResult:
    if len(parent_ids) != len(intervention_ids) or not parent_ids:
        raise ValueError("parent_ids and intervention_ids must be non-empty and aligned")
    count = len(parent_ids)
    weights = np.ones(count, dtype=np.float32)
    if quality_weights is not None:
        weights = np.asarray(quality_weights, dtype=np.float32)
        if weights.shape != (count,) or np.any((weights < 0.0) | (weights > 1.0)):
            raise ValueError("quality weights must lie in [0, 1]")
    parent = np.asarray(parent_ids, dtype=object)
    intervention = np.asarray(intervention_ids, dtype=object)
    same_parent = parent[:, None] == parent[None, :]
    different_intervention = intervention[:, None] != intervention[None, :]
    identity = np.eye(count, dtype=bool)
    positive = same_parent & different_intervention & ~identity
    negative = ~same_parent & ~identity
    pair_weights = np.sqrt(weights[:, None] * weights[None, :])
    groups = defaultdict(int)
    for parent_id in parent_ids:
        groups[parent_id] += 1
    return PairingResult(
        positive_mask=positive,
        negative_mask=negative,
        pair_weights=pair_weights,
        diagnostics={
            "examples": count,
            "parent_groups": len(groups),
            "positive_pairs": int(positive.sum()),
            "negative_pairs": int(negative.sum()),
            "singleton_groups": sum(size == 1 for size in groups.values()),
        },
    )
