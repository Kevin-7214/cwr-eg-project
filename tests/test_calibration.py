from __future__ import annotations

import numpy as np
import pytest

from cwr_eg.calibration import (
    CalibrationBundleHeader,
    bonferroni,
    clopper_pearson_upper,
    empirical_upper_p,
)


def _header(search_hash: str = "search") -> CalibrationBundleHeader:
    return CalibrationBundleHeader(
        calibration_id="cal-v1",
        protocol_version="0.1",
        data_manifest_hash="data",
        search_config_hash=search_hash,
        registered_registry_hash="registered",
        normalization_version="norm-v1",
        model_version="model-v1",
        languages=("en", "zh"),
        strata=("language", "length"),
    )


def test_empirical_p_has_plus_one_smoothing_and_is_monotone() -> None:
    null = np.asarray([0.0, 1.0, 2.0, 3.0])
    assert empirical_upper_p(4.0, null) == pytest.approx(0.2)
    assert empirical_upper_p(2.0, null) >= empirical_upper_p(3.0, null)
    assert bonferroni(0.2, 10) == 1.0


def test_exact_fwer_upper_matches_predeclared_zero_error_requirement() -> None:
    assert clopper_pearson_upper(0, 198) > 0.015
    assert clopper_pearson_upper(0, 199) <= 0.015
    assert clopper_pearson_upper(0, 200) == pytest.approx(0.014867, abs=1e-6)


def test_bundle_compatibility_changes_with_search_space() -> None:
    _header().require_compatible(_header())
    with pytest.raises(ValueError, match="calibration_bundle_mismatch"):
        _header().require_compatible(_header("changed-search"))
