from __future__ import annotations

import pytest

from cwr_eg.contracts import (
    CharacterInterval,
    GenericResidualEvidence,
    RegisteredEvidence,
    ValidityDiagnostics,
)
from cwr_eg.decision import decide_segment
from cwr_eg.enums import (
    Applicability,
    DecisionLabel,
    KeyStatus,
    TailDirection,
    ValidityStatus,
)


INTERVAL = CharacterInterval(0, 100)


def generic(p_value: float | None) -> GenericResidualEvidence:
    return GenericResidualEvidence(
        evidence_id="g",
        document_id="d",
        interval=INTERVAL,
        scale_id="char-100",
        raw_score=3.0,
        generic_p=p_value,
        effective_length=100,
        model_version="m",
        calibration_id="c",
    )


def validity(status: ValidityStatus = ValidityStatus.PASS) -> ValidityDiagnostics:
    return ValidityDiagnostics(
        status=status,
        effective_length=100,
        mapping_coverage=1.0,
        repetition_ratio=0.0,
        code_or_template_ratio=0.0,
        language_supported=True,
        reason_codes=() if status is ValidityStatus.PASS else ("mapping_coverage_low",),
    )


def registered(
    family: str,
    key_status: KeyStatus,
    p_value: float,
) -> RegisteredEvidence:
    return RegisteredEvidence(
        detector_id="detector-" + family,
        scheme_id=family,
        scheme_family=family,
        key_id_hash="sha256:key" if key_status is KeyStatus.REGISTERED else None,
        key_status=key_status,
        interval=INTERVAL,
        raw_statistic=4.0,
        tail_direction=TailDirection.UPPER,
        single_test_p=p_value,
        adjusted_p=p_value,
        applicability=Applicability.VALID,
    )


@pytest.mark.parametrize(
    ("generic_p", "registered_items", "gap_p", "state", "expected"),
    [
        (0.5, (), 0.5, ValidityStatus.FAIL, DecisionLabel.UNCERTAIN),
        (
            0.5,
            (registered("kgw", KeyStatus.REGISTERED, 0.001),),
            0.5,
            ValidityStatus.PASS,
            DecisionLabel.KNOWN_SCHEME_KNOWN_KEY,
        ),
        (
            0.001,
            (registered("kgw", KeyStatus.SCHEME_ONLY, 0.001),),
            0.001,
            ValidityStatus.PASS,
            DecisionLabel.KNOWN_SCHEME_UNKNOWN_KEY,
        ),
        (0.001, (), 0.001, ValidityStatus.PASS, DecisionLabel.SUSPECTED_UNKNOWN_SCHEME),
        (0.5, (), 0.5, ValidityStatus.PASS, DecisionLabel.NONE),
        (
            0.001,
            (registered("kgw", KeyStatus.SCHEME_ONLY, 0.001),),
            0.5,
            ValidityStatus.PASS,
            DecisionLabel.UNCERTAIN,
        ),
    ],
)
def test_five_class_truth_table(
    generic_p: float,
    registered_items: tuple[RegisteredEvidence, ...],
    gap_p: float,
    state: ValidityStatus,
    expected: DecisionLabel,
) -> None:
    decision = decide_segment(
        segment_id="s",
        interval=INTERVAL,
        generic=generic(generic_p),
        registered=registered_items,
        validity=validity(state),
        gap_score=2.0,
        gap_p=gap_p,
        alpha=0.01,
        normalization_version="n",
    )
    assert decision.label is expected


def test_conflicting_registered_families_abstain() -> None:
    decision = decide_segment(
        segment_id="s",
        interval=INTERVAL,
        generic=generic(0.001),
        registered=(
            registered("kgw", KeyStatus.REGISTERED, 0.001),
            registered("unigram", KeyStatus.REGISTERED, 0.001),
        ),
        validity=validity(),
        gap_score=0.0,
        gap_p=0.5,
        normalization_version="n",
    )
    assert decision.label is DecisionLabel.UNCERTAIN
    assert "registered_evidence_conflict" in decision.abstain_reasons
