from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

import cwr_eg.runtime as runtime_module
from cwr_eg.hashing import sha256_file
from cwr_eg.runtime import _model_smoke, _run_model_smoke_operation


class _Tensor:
    def __init__(self, tokens: int) -> None:
        self.shape = (1, tokens)

    def __getitem__(self, index: int) -> "_Tensor":
        return self


class _Model:
    def __init__(self) -> None:
        self.forward_calls = 0
        self.generate_calls = 0

    def __call__(self, **encoded: object) -> SimpleNamespace:
        self.forward_calls += 1
        return SimpleNamespace(logits=object())

    def generate(self, **kwargs: object) -> _Tensor:
        self.generate_calls += 1
        return _Tensor(tokens=7)


class _Finite:
    def all(self) -> "_Finite":
        return self

    def item(self) -> bool:
        return True


class _Torch:
    @staticmethod
    def inference_mode() -> nullcontext[None]:
        return nullcontext()

    @staticmethod
    def isfinite(value: object) -> _Finite:
        return _Finite()


class _Tokenizer:
    eos_token_id = 151645

    @staticmethod
    def decode(value: object, *, skip_special_tokens: bool) -> str:
        assert skip_special_tokens is True
        return "decoded"


def test_forward_only_never_generates() -> None:
    model = _Model()
    result = _run_model_smoke_operation(
        "forward_only",
        model=model,
        encoded={"input_ids": _Tensor(tokens=4)},
        tokenizer=_Tokenizer(),
        torch_module=_Torch(),
        max_new_tokens=None,
    )
    assert result == {"logits_finite": True}
    assert model.forward_calls == 1
    assert model.generate_calls == 0


def test_generate_only_never_calls_standalone_forward() -> None:
    model = _Model()
    result = _run_model_smoke_operation(
        "generate_only",
        model=model,
        encoded={"input_ids": _Tensor(tokens=4)},
        tokenizer=_Tokenizer(),
        torch_module=_Torch(),
        max_new_tokens=3,
    )
    assert result == {
        "output_tokens": 7,
        "generated_tokens": 3,
        "generated_text": "decoded",
    }
    assert model.forward_calls == 0
    assert model.generate_calls == 1


def test_model_smoke_fails_before_any_model_import_without_exact_operation() -> None:
    with pytest.raises(ValueError, match="operation"):
        _model_smoke(Path("configs/pilot.yaml"), {})


def test_model_smoke_rejects_asset_paths_outside_model_directory(tmp_path) -> None:
    scope = {
        "operation": "forward_only",
        "local_files_only": True,
        "trust_remote_code": False,
        "do_sample": False,
        "runner_sha256": sha256_file(Path(runtime_module.__file__)),
        "model_path": str(tmp_path),
        "model_files": [
            {"path": "../outside", "bytes": 0, "sha256": "not-reached"}
        ],
    }
    with pytest.raises(ValueError, match="inside model_path"):
        _model_smoke(Path("configs/pilot.yaml"), scope)
