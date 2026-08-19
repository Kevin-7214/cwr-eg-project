from __future__ import annotations

import pytest

from cwr_eg.hashing import sha256_file
from cwr_eg.runtime import _verify_freeze_manifest


def test_intermediate_actions_verify_the_frozen_manifest_hash(tmp_path) -> None:
    path = tmp_path / "freeze.json"
    path.write_text("{}\n", encoding="utf-8")
    assert _verify_freeze_manifest(
        {"freeze_manifest": str(path), "freeze_manifest_sha256": sha256_file(path)}
    ) == sha256_file(path)
    with pytest.raises(RuntimeError, match="freeze manifest"):
        _verify_freeze_manifest(
            {"freeze_manifest": str(path), "freeze_manifest_sha256": "0" * 64}
        )
