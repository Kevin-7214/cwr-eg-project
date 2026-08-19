from __future__ import annotations

from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import pytest

from cwr_eg.markllm_bridge import MarkLlmBridge, oriented_evidence_strength


def test_unbiased_statistic_preserves_lower_tail_and_adds_upward_strength() -> None:
    transformed = oriented_evidence_strength("unbiased", 7.25)
    assert transformed == {
        "raw_tail_direction": "lower",
        "evidence_strength": -7.25,
        "evidence_transform_version": "negate-lower-tail-statistic-v1",
    }
    assert oriented_evidence_strength("kgw", 2.5)["evidence_strength"] == 2.5


def test_markllm_family_key_instance_is_cached(monkeypatch, tmp_path) -> None:
    calls = 0
    watermark = object()

    class FakeAutoWatermark:
        @staticmethod
        def load(*args, **kwargs):
            nonlocal calls
            calls += 1
            return watermark

    package = ModuleType("watermark")
    module = ModuleType("watermark.auto_watermark")
    module.AutoWatermark = FakeAutoWatermark
    monkeypatch.setitem(sys.modules, "watermark", package)
    monkeypatch.setitem(sys.modules, "watermark.auto_watermark", module)
    monkeypatch.setenv("CWR_EG_KEY_KGW_KEY_A", "15485863")
    bridge = object.__new__(MarkLlmBridge)
    bridge.settings = SimpleNamespace(repository=Path(tmp_path))
    bridge.transformers_config = object()
    bridge._watermark_cache = {}
    assert bridge.load_watermark("kgw", "kgw_key_a") is watermark
    assert bridge.load_watermark("kgw", "kgw_key_a") is watermark
    assert calls == 1
