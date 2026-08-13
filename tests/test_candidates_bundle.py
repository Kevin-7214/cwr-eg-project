from __future__ import annotations

import numpy as np
import pytest

from cwr_eg.bundle import CalibrationBundle
from cwr_eg.calibration import CalibrationBundleHeader
from cwr_eg.candidates import generate_candidates


def header() -> CalibrationBundleHeader:
    return CalibrationBundleHeader(
        calibration_id="cal",
        protocol_version="0.1",
        data_manifest_hash="data",
        search_config_hash="search",
        registered_registry_hash="registered",
        normalization_version="norm",
        model_version="model",
        languages=("en", "zh"),
        strata=("language",),
    )


def test_candidate_search_uses_raw_character_coordinates() -> None:
    scores = np.zeros(100)
    scores[40:60] = 10.0
    candidates = generate_candidates(scores, [20], stride_fraction=0.5, candidate_quantile=0.9)
    assert candidates
    assert any(item.interval.char_start <= 40 and item.interval.char_end >= 60 for item in candidates)


def test_calibration_bundle_round_trip_and_checksum(tmp_path) -> None:
    bundle = CalibrationBundle(
        header=header(),
        generic_null_by_stratum={"en:all": np.asarray([1.0, 2.0, 3.0])},
        registered_null_by_stratum={"en:all": np.asarray([0.0, 1.0, 2.0])},
        gap_null_by_stratum={"en:all": np.asarray([-1.0, 0.0, 1.0])},
        validity_rules={"minimum_effective_length": 64},
    )
    path = bundle.save(tmp_path / "bundle")
    loaded = CalibrationBundle.load(path, expected_header=header())
    assert loaded.p_value("generic", "en:all", 4.0) == pytest.approx(0.25)
    with pytest.raises(KeyError):
        loaded.p_value("generic", "zh:all", 1.0)
