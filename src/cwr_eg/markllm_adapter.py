from __future__ import annotations

from typing import Any, Mapping

from cwr_eg.contracts import CharacterInterval, RegisteredEvidence
from cwr_eg.enums import Applicability, KeyStatus, TailDirection
from cwr_eg.markllm_bridge import MarkLlmBridge
from cwr_eg.registered import (
    AuthorizedKey,
    DetectorDeclaration,
    RegisteredDetectorAdapter,
)


class MarkLlmRegisteredAdapter(RegisteredDetectorAdapter):
    def __init__(
        self,
        *,
        bridge: MarkLlmBridge,
        family: str,
        tokenizer_id: str,
        tokenizer_revision: str,
        minimum_effective_length: int = 64,
        supported_languages: tuple[str, ...] = ("en", "zh"),
    ) -> None:
        self.bridge = bridge
        self.family = family
        tail_direction = (
            TailDirection.LOWER if family == "unbiased" else TailDirection.UPPER
        )
        self.declaration = DetectorDeclaration(
            detector_id=f"markllm-{family}-c45ddc4",
            scheme_id=family,
            scheme_family=family,
            tokenizer_id=tokenizer_id,
            tokenizer_revision=tokenizer_revision,
            requires_key=True,
            supports_scheme_only=False,
            minimum_effective_length=minimum_effective_length,
            supported_languages=supported_languages,
            tail_direction=tail_direction,
            emits_local_evidence=False,
            license="apache-2.0",
            source_url="https://github.com/THU-BPM/MarkLLM",
            source_revision="c45ddc40f7b761beabe55a1b8dc4690e531d1c6d",
        )

    def score(
        self,
        raw_text: str,
        interval: CharacterInterval,
        language: str,
        authorized_key: AuthorizedKey | None,
        context: Mapping[str, Any],
    ) -> RegisteredEvidence:
        interval.validate_text(raw_text)
        effective_length = int(context.get("effective_length", 0))
        self.validate_request(language, authorized_key, effective_length)
        assert authorized_key is not None
        if not authorized_key.key_id.startswith(self.family + "_key_"):
            raise PermissionError("authorized_key_family_mismatch")
        result = self.bridge.detect(
            raw_text[interval.char_start : interval.char_end],
            self.family,
            authorized_key.key_id,
            authorized_key.secret,
        )
        return RegisteredEvidence(
            detector_id=self.declaration.detector_id,
            scheme_id=self.declaration.scheme_id,
            scheme_family=self.declaration.scheme_family,
            key_id_hash=authorized_key.key_id_hash,
            key_status=KeyStatus.REGISTERED,
            interval=interval,
            raw_statistic=float(result["score"]),
            tail_direction=self.declaration.tail_direction,
            single_test_p=None,
            adjusted_p=None,
            applicability=Applicability.VALID,
            reason_codes=(),
        )
