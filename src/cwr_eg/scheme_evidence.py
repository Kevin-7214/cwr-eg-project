from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from cwr_eg.contracts import CharacterInterval, RegisteredEvidence
from cwr_eg.enums import Applicability, KeyStatus, TailDirection
from cwr_eg.registered import DetectorDeclaration, RegisteredDetectorAdapter


SchemeScoreFunction = Callable[[str, CharacterInterval, str, Mapping[str, Any]], float]


class CalibratedSchemeOnlyAdapter(RegisteredDetectorAdapter):
    """Expose only a predeclared key-independent statistic, never a private-head label."""

    def __init__(
        self,
        *,
        scheme_family: str,
        score_function: SchemeScoreFunction,
        detector_id: str,
        model_version: str,
        calibration_id: str,
        minimum_effective_length: int = 64,
        supported_languages: tuple[str, ...] = ("en", "zh"),
    ) -> None:
        if not model_version or not calibration_id:
            raise ValueError("Scheme-only evidence requires frozen model and calibration IDs")
        self.score_function = score_function
        self.model_version = model_version
        self.calibration_id = calibration_id
        self.declaration = DetectorDeclaration(
            detector_id=detector_id,
            scheme_id=scheme_family,
            scheme_family=scheme_family,
            tokenizer_id="cwr-eg-character-evidence",
            tokenizer_revision=model_version,
            requires_key=False,
            supports_scheme_only=True,
            minimum_effective_length=minimum_effective_length,
            supported_languages=supported_languages,
            tail_direction=TailDirection.UPPER,
            emits_local_evidence=True,
            license="project-owned",
            source_url="local:cwr_eg/scheme_evidence.py",
            source_revision=model_version,
        )

    def score(
        self,
        raw_text: str,
        interval: CharacterInterval,
        language: str,
        authorized_key: Any,
        context: Mapping[str, Any],
    ) -> RegisteredEvidence:
        if authorized_key is not None:
            raise PermissionError("Scheme-only detector must not receive a key")
        interval.validate_text(raw_text)
        self.validate_request(language, None, int(context.get("effective_length", 0)))
        if context.get("calibration_id") != self.calibration_id:
            raise ValueError("calibration_bundle_mismatch")
        statistic = float(self.score_function(raw_text, interval, language, context))
        return RegisteredEvidence(
            detector_id=self.declaration.detector_id,
            scheme_id=self.declaration.scheme_id,
            scheme_family=self.declaration.scheme_family,
            key_id_hash=None,
            key_status=KeyStatus.SCHEME_ONLY,
            interval=interval,
            raw_statistic=statistic,
            tail_direction=TailDirection.UPPER,
            single_test_p=None,
            adjusted_p=None,
            applicability=Applicability.VALID,
            reason_codes=(),
        )
