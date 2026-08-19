from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import math
from typing import Any, Mapping, Sequence

from cwr_eg.enums import (
    Applicability,
    DecisionLabel,
    KeyStatus,
    TailDirection,
    ValidityStatus,
)


def _probability(value: float | None, name: str) -> None:
    if value is not None and not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1]")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


@dataclass(frozen=True, slots=True, order=True)
class CharacterInterval:
    char_start: int
    char_end: int

    def __post_init__(self) -> None:
        if self.char_start < 0 or self.char_end <= self.char_start:
            raise ValueError(
                f"Invalid half-open interval [{self.char_start}, {self.char_end})"
            )

    @property
    def length(self) -> int:
        return self.char_end - self.char_start

    def validate_text(self, text: str) -> None:
        if self.char_end > len(text):
            raise ValueError("Interval exceeds raw Unicode code-point length")


@dataclass(frozen=True, slots=True)
class GenericResidualEvidence:
    evidence_id: str
    document_id: str
    interval: CharacterInterval
    scale_id: str
    raw_score: float
    generic_p: float | None
    effective_length: int
    model_version: str
    calibration_id: str
    character_scores_ref: str | None = None
    representation_diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _probability(self.generic_p, "generic_p")
        if self.effective_length < 0:
            raise ValueError("effective_length cannot be negative")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update(payload.pop("interval"))
        return _jsonable(payload)


@dataclass(frozen=True, slots=True)
class RegisteredEvidence:
    detector_id: str
    scheme_id: str
    scheme_family: str
    key_id_hash: str | None
    key_status: KeyStatus
    interval: CharacterInterval
    raw_statistic: float
    tail_direction: TailDirection
    single_test_p: float | None
    adjusted_p: float | None
    applicability: Applicability
    reason_codes: tuple[str, ...] = ()
    evidence_strength: float | None = None
    evidence_transform_version: str | None = None

    def __post_init__(self) -> None:
        _probability(self.single_test_p, "single_test_p")
        _probability(self.adjusted_p, "adjusted_p")
        if self.key_status is KeyStatus.REGISTERED and not self.key_id_hash:
            raise ValueError("Registered-key evidence requires key_id_hash")
        if self.key_status is not KeyStatus.REGISTERED and self.key_id_hash:
            raise ValueError("Only registered-key evidence may expose key_id_hash")
        if self.evidence_strength is not None and not math.isfinite(
            self.evidence_strength
        ):
            raise ValueError("evidence_strength must be finite")
        if (self.evidence_strength is None) != (
            self.evidence_transform_version is None
        ):
            raise ValueError("Evidence strength and transform version must be paired")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update(payload.pop("interval"))
        return _jsonable(payload)


@dataclass(frozen=True, slots=True)
class ValidityDiagnostics:
    status: ValidityStatus
    effective_length: int
    mapping_coverage: float
    repetition_ratio: float
    code_or_template_ratio: float
    language_supported: bool
    token_entropy: float | None = None
    domain_shift_score: float | None = None
    reason_codes: tuple[str, ...] = ()
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.effective_length < 0:
            raise ValueError("effective_length cannot be negative")
        for name in ("mapping_coverage", "repetition_ratio", "code_or_template_ratio"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")
        if self.status is ValidityStatus.PASS and self.reason_codes:
            raise ValueError("Passing diagnostics cannot carry reason codes")

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass(frozen=True, slots=True)
class SegmentDecision:
    segment_id: str
    interval: CharacterInterval
    label: DecisionLabel
    generic_evidence: GenericResidualEvidence
    registered_candidates: tuple[RegisteredEvidence, ...]
    validity: ValidityDiagnostics
    calibration_id: str
    model_version: str
    normalization_version: str
    gap_score: float | None = None
    gap_p: float | None = None
    best_registered_adjusted_p: float | None = None
    abstain_reasons: tuple[str, ...] = ()
    overlap_flags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _probability(self.gap_p, "gap_p")
        _probability(self.best_registered_adjusted_p, "best_registered_adjusted_p")
        if self.interval != self.generic_evidence.interval:
            raise ValueError("Segment and generic-evidence intervals must match")
        if self.calibration_id != self.generic_evidence.calibration_id:
            raise ValueError("Segment and generic evidence calibration IDs must match")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update(payload.pop("interval"))
        return _jsonable(payload)


@dataclass(frozen=True, slots=True)
class DocumentDecision:
    document_id: str
    document_label: DecisionLabel
    document_label_set: tuple[DecisionLabel, ...]
    segments: tuple[SegmentDecision, ...]
    document_generic_p: float | None
    document_registered_p: float | None
    document_gap_p: float | None
    any_accept: bool
    document_validity: ValidityDiagnostics
    search_space_summary: Mapping[str, Any]
    versions: Mapping[str, str]
    runtime: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _probability(self.document_generic_p, "document_generic_p")
        _probability(self.document_registered_p, "document_registered_p")
        _probability(self.document_gap_p, "document_gap_p")
        actual = tuple(dict.fromkeys(segment.label for segment in self.segments))
        if set(actual) != set(self.document_label_set):
            raise ValueError("document_label_set must match the segment decisions")

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass(frozen=True, slots=True)
class ParentSample:
    parent_id: str
    source: str
    language: str
    split: str
    text: str
    text_sha256: str
    source_line: int
    license: str | Sequence[str]

    def __post_init__(self) -> None:
        if self.split not in {"train", "dev", "calibration", "test"}:
            raise ValueError(f"Unsupported split: {self.split}")
        if not self.parent_id or not self.source or not self.language:
            raise ValueError("Parent samples require identifiers and language")
        if not self.text:
            raise ValueError("Parent sample text cannot be empty")

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))
