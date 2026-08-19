from __future__ import annotations

import pytest

from cwr_eg.hashing import sha256_file
from cwr_eg.manifest import write_jsonl
from cwr_eg.runtime import _verify_feature_resume_manifest


def test_feature_resume_manifest_requires_the_approved_hash(tmp_path) -> None:
    manifest = tmp_path / "feature_manifest.jsonl"
    write_jsonl(manifest, [{"recipe_id": "recipe-1"}])
    _verify_feature_resume_manifest(manifest, sha256_file(manifest))
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        _verify_feature_resume_manifest(manifest, "0" * 64)


def test_feature_resume_manifest_must_exist_when_frozen(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="resume manifest is missing"):
        _verify_feature_resume_manifest(
            tmp_path / "feature_manifest.jsonl", "0" * 64
        )
