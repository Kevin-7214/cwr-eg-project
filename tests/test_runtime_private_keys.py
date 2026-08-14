from __future__ import annotations

import os

import pytest

from cwr_eg.hashing import sha256_file
from cwr_eg.enums import TailDirection
from cwr_eg.markllm_adapter import MarkLlmRegisteredAdapter
from cwr_eg.markllm_bridge import _unbiased_key_bytes
from cwr_eg.runtime import (
    _compose_adjacent_components,
    _deterministic_attack,
    _verify_code_files,
    _load_private_keys,
)


def test_unbiased_key_material_is_fixed_width_and_stable() -> None:
    first = _unbiased_key_bytes(123456)
    assert len(first) == 128
    assert first == _unbiased_key_bytes("123456")
    with pytest.raises(ValueError):
        _unbiased_key_bytes(0)


def test_private_key_file_requires_exact_hash_and_ids(tmp_path, monkeypatch) -> None:
    path = tmp_path / "keys.env"
    path.write_text("CWR_EG_KEY_KGW_KEY_A=123\n", encoding="utf-8")
    scope = {
        "key_file": str(path),
        "key_file_sha256": sha256_file(path),
        "required_key_ids": ["kgw_key_a"],
    }
    monkeypatch.delenv("CWR_EG_KEY_KGW_KEY_A", raising=False)
    assert _load_private_keys(scope) == sha256_file(path)
    assert os.environ["CWR_EG_KEY_KGW_KEY_A"] == "123"
    scope["key_file_sha256"] = "wrong"
    with pytest.raises(RuntimeError, match="SHA-256"):
        _load_private_keys(scope)


def test_deterministic_attacks_preserve_declared_boundaries() -> None:
    assert _deterministic_attack("a  b\nc", "copy_edit", 0.75) == "a b c"
    assert _deterministic_attack("abcdefgh", "truncation", 0.75) == "abcdef"
    with pytest.raises(ValueError, match="truncation_fraction"):
        _deterministic_attack("abc", "truncation", 1.0)


def test_adjacent_components_receive_exact_non_overlapping_intervals() -> None:
    text, intervals = _compose_adjacent_components(["abc", "de"], "\n\n")
    assert text == "abc\n\nde"
    assert intervals == [[0, 3], [5, 7]]
    assert [text[start:end] for start, end in intervals] == ["abc", "de"]
    with pytest.raises(ValueError, match="At least two"):
        _compose_adjacent_components(["abc"], "\n\n")


def test_unbiased_adapter_declares_lower_tail_without_loading_model() -> None:
    adapter = MarkLlmRegisteredAdapter(
        bridge=object(),
        family="unbiased",
        tokenizer_id="local-qwen",
        tokenizer_revision="fixed",
    )
    assert adapter.declaration.tail_direction is TailDirection.LOWER


def test_code_file_manifest_rejects_hash_drift_and_path_escape(
    tmp_path, monkeypatch
) -> None:
    source = tmp_path / "module.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    scope = {"code_files": [{"path": "module.py", "sha256": sha256_file(source)}]}
    assert _verify_code_files(scope) == {"module.py": sha256_file(source)}
    scope["code_files"][0]["sha256"] = "wrong"
    with pytest.raises(RuntimeError, match="Code SHA-256 mismatch"):
        _verify_code_files(scope)
    with pytest.raises(ValueError, match="project root"):
        _verify_code_files(
            {"code_files": [{"path": "../outside.py", "sha256": "unused"}]}
        )
