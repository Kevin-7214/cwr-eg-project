from __future__ import annotations

import numpy as np

from cwr_eg.contracts import CharacterInterval
from cwr_eg.features import ExtractedViews, FeatureView
from cwr_eg.runtime import _atomic_write_feature_npz, _recover_feature_entry


def test_atomic_feature_file_can_rebuild_missing_manifest_entry(tmp_path) -> None:
    intervals = (CharacterInterval(0, 1), CharacterInterval(1, 2))
    mask = np.ones(2, dtype=bool)
    views = {
        name: FeatureView(
            name=name,
            values=np.ones((2, dimension), dtype=np.float32),
            raw_intervals=intervals,
            valid_mask=mask,
            metadata={},
        )
        for name, dimension in {
            "proxy": 2,
            "representation": 3,
            "perturbation": 2,
            "validity": 2,
        }.items()
    }
    extracted = ExtractedViews(
        document_id="recipe-1",
        views=views,
        normalization_version="norm-v1",
        extractor_version="extractor-v1",
    )
    path = tmp_path / "recipe-1.npz"
    dimensions = _atomic_write_feature_npz(path, extracted=extracted, numpy_module=np)
    assert path.is_file()
    assert not path.with_suffix(".npz.tmp").exists()
    row = {
        "recipe_id": "recipe-1",
        "parent_ids": ["parent-1"],
        "split": "train",
        "source": "c4_en",
        "language": "en",
    }
    entry, recovered_dimensions = _recover_feature_entry(row, path, np)
    assert entry["extractor_version"] == "extractor-v1"
    assert entry["normalization_version"] == "norm-v1"
    assert recovered_dimensions == dimensions
