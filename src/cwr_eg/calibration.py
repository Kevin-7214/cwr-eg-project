from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.stats import beta

from cwr_eg.hashing import content_hash


def empirical_upper_p(value: float, null_values: Iterable[float]) -> float:
    null = np.asarray(tuple(null_values), dtype=np.float64)
    if null.ndim != 1 or not len(null) or not np.all(np.isfinite(null)):
        raise ValueError("A finite, non-empty one-dimensional null is required")
    return float((1 + np.count_nonzero(null >= value)) / (len(null) + 1))


def empirical_lower_p(value: float, null_values: Iterable[float]) -> float:
    null = np.asarray(tuple(null_values), dtype=np.float64)
    if null.ndim != 1 or not len(null) or not np.all(np.isfinite(null)):
        raise ValueError("A finite, non-empty one-dimensional null is required")
    return float((1 + np.count_nonzero(null <= value)) / (len(null) + 1))


def search_aware_p(observed_maximum: float, null_maxima: Iterable[float]) -> float:
    return empirical_upper_p(observed_maximum, null_maxima)


def bonferroni(p_value: float, tests: int) -> float:
    if not 0.0 <= p_value <= 1.0 or tests < 1:
        raise ValueError("Invalid p-value or number of tests")
    return min(1.0, p_value * tests)


def clopper_pearson_upper(
    errors: int, trials: int, confidence_level: float = 0.95
) -> float:
    if trials < 1 or errors < 0 or errors > trials:
        raise ValueError("Require 0 <= errors <= trials and trials >= 1")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie in (0, 1)")
    if errors == trials:
        return 1.0
    alpha = 1.0 - confidence_level
    return float(beta.ppf(1.0 - alpha, errors + 1, trials - errors))


@dataclass(frozen=True, slots=True)
class CalibrationBundleHeader:
    calibration_id: str
    protocol_version: str
    data_manifest_hash: str
    search_config_hash: str
    registered_registry_hash: str
    normalization_version: str
    model_version: str
    languages: tuple[str, ...]
    strata: tuple[str, ...]

    def compatibility_hash(self) -> str:
        return content_hash(
            {
                "protocol_version": self.protocol_version,
                "search_config_hash": self.search_config_hash,
                "registered_registry_hash": self.registered_registry_hash,
                "normalization_version": self.normalization_version,
                "model_version": self.model_version,
                "languages": self.languages,
                "strata": self.strata,
            }
        )

    def require_compatible(self, expected: "CalibrationBundleHeader") -> None:
        if self.compatibility_hash() != expected.compatibility_hash():
            raise ValueError("calibration_bundle_mismatch")
