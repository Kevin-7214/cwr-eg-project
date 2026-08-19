from __future__ import annotations

import json

import pytest

from cwr_eg.bundle import CalibrationBundle, fit_parent_calibration_bundle_from_records
from cwr_eg.calibration import CalibrationBundleHeader


def _header() -> CalibrationBundleHeader:
    return CalibrationBundleHeader(
        calibration_id="intermediate-cal-v1",
        protocol_version="0.2",
        data_manifest_hash="data",
        search_config_hash="search",
        registered_registry_hash="registered",
        normalization_version="raw-unicode-codepoint-v1",
        model_version="ensemble-v1",
        languages=("en", "zh"),
        strata=("language",),
    )


def test_parent_calibration_merges_clean_descendants_and_reaches_alpha(tmp_path) -> None:
    records = tmp_path / "records.jsonl"
    with records.open("w", encoding="utf-8") as handle:
        for language in ("en", "zh"):
            for parent_index in range(100):
                parent_id = f"{language}-{parent_index:03d}"
                for route_index, route in enumerate(("generic", "registered", "gap")):
                    for descendant_offset in (0.0, 0.5):
                        handle.write(
                            json.dumps(
                                {
                                    "split": "calibration",
                                    "parent_id": parent_id,
                                    "route": route,
                                    "stratum": f"{language}:all",
                                    "maximum_statistic": parent_index + route_index + descendant_offset,
                                    "is_null_descendant": True,
                                    "full_search_executed": True,
                                }
                            )
                            + "\n"
                        )
    output = fit_parent_calibration_bundle_from_records(
        records_path=records,
        output_dir=tmp_path / "bundle",
        header=_header(),
        validity_rules={"minimum_mapping_coverage": 0.98},
        minimum_parents_per_stratum=100,
    )
    bundle = CalibrationBundle.load(output)
    assert len(bundle.generic_null_by_stratum["en:all"]) == 100
    assert bundle.generic_null_by_stratum["en:all"][0] == pytest.approx(0.5)
    assert bundle.metadata["minimum_empirical_p_by_stratum"]["en:all"] == pytest.approx(
        1 / 101
    )
