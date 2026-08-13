from __future__ import annotations

from collections.abc import Sequence

from cwr_eg.contracts import (
    CharacterInterval,
    DocumentDecision,
    GenericResidualEvidence,
    RegisteredEvidence,
    SegmentDecision,
    ValidityDiagnostics,
)
from cwr_eg.enums import Applicability, DecisionLabel, KeyStatus, ValidityStatus


ACCEPT_LABELS = {
    DecisionLabel.KNOWN_SCHEME_KNOWN_KEY,
    DecisionLabel.KNOWN_SCHEME_UNKNOWN_KEY,
    DecisionLabel.SUSPECTED_UNKNOWN_SCHEME,
}


def decide_segment(
    *,
    segment_id: str,
    interval: CharacterInterval,
    generic: GenericResidualEvidence,
    registered: Sequence[RegisteredEvidence],
    validity: ValidityDiagnostics,
    gap_score: float | None,
    gap_p: float | None,
    alpha: float = 0.01,
    normalization_version: str,
) -> SegmentDecision:
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    if interval != generic.interval:
        raise ValueError("Decision interval must match generic evidence")

    valid_registered = tuple(
        item
        for item in registered
        if item.applicability is not Applicability.INVALID and item.adjusted_p is not None
    )
    significant = tuple(item for item in valid_registered if item.adjusted_p <= alpha)
    key_hits = tuple(item for item in significant if item.key_status is KeyStatus.REGISTERED)
    scheme_hits = tuple(item for item in significant if item.key_status is KeyStatus.SCHEME_ONLY)
    significant_families = {item.scheme_family for item in significant}
    generic_hit = generic.generic_p is not None and generic.generic_p <= alpha
    gap_hit = gap_p is not None and gap_p <= alpha
    reasons: list[str] = []

    if validity.status is ValidityStatus.FAIL:
        label = DecisionLabel.UNCERTAIN
        reasons.extend(validity.reason_codes or ("invalid_evidence",))
    elif len(significant_families) > 1:
        label = DecisionLabel.UNCERTAIN
        reasons.append("registered_evidence_conflict")
    elif key_hits:
        label = DecisionLabel.KNOWN_SCHEME_KNOWN_KEY
    elif scheme_hits and generic_hit and gap_hit:
        label = DecisionLabel.KNOWN_SCHEME_UNKNOWN_KEY
    elif generic_hit and not significant:
        label = DecisionLabel.SUSPECTED_UNKNOWN_SCHEME
    elif generic_hit or significant:
        label = DecisionLabel.UNCERTAIN
        if generic_hit and not gap_hit:
            reasons.append("generic_significant_gap_not_significant")
        elif gap_hit and not generic_hit:
            reasons.append("gap_significant_generic_not_significant")
        elif scheme_hits:
            reasons.append("no_key_independent_scheme_test")
        else:
            reasons.append("borderline_threshold")
    else:
        label = DecisionLabel.NONE

    best_p = min(
        (item.adjusted_p for item in valid_registered if item.adjusted_p is not None),
        default=None,
    )
    return SegmentDecision(
        segment_id=segment_id,
        interval=interval,
        label=label,
        generic_evidence=generic,
        registered_candidates=tuple(
            sorted(
                registered,
                key=lambda item: 1.0 if item.adjusted_p is None else item.adjusted_p,
            )
        ),
        validity=validity,
        calibration_id=generic.calibration_id,
        model_version=generic.model_version,
        normalization_version=normalization_version,
        gap_score=gap_score,
        gap_p=gap_p,
        best_registered_adjusted_p=best_p,
        abstain_reasons=tuple(dict.fromkeys(reasons)),
    )


def aggregate_document(
    *,
    document_id: str,
    segments: Sequence[SegmentDecision],
    document_validity: ValidityDiagnostics,
    document_generic_p: float | None,
    document_registered_p: float | None,
    document_gap_p: float | None,
    search_space_summary: dict[str, object],
    versions: dict[str, str],
    runtime: dict[str, object] | None = None,
) -> DocumentDecision:
    labels = tuple(dict.fromkeys(segment.label for segment in segments))
    if document_validity.status is ValidityStatus.FAIL or DecisionLabel.UNCERTAIN in labels:
        document_label = DecisionLabel.UNCERTAIN
    else:
        priority = (
            DecisionLabel.KNOWN_SCHEME_KNOWN_KEY,
            DecisionLabel.KNOWN_SCHEME_UNKNOWN_KEY,
            DecisionLabel.SUSPECTED_UNKNOWN_SCHEME,
            DecisionLabel.NONE,
        )
        document_label = next(
            (candidate for candidate in priority if candidate in labels),
            DecisionLabel.NONE,
        )
    return DocumentDecision(
        document_id=document_id,
        document_label=document_label,
        document_label_set=labels,
        segments=tuple(segments),
        document_generic_p=document_generic_p,
        document_registered_p=document_registered_p,
        document_gap_p=document_gap_p,
        any_accept=document_label in ACCEPT_LABELS,
        document_validity=document_validity,
        search_space_summary=search_space_summary,
        versions=versions,
        runtime=runtime or {},
    )
