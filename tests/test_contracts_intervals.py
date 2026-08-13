from __future__ import annotations

import pytest

from cwr_eg.contracts import CharacterInterval, ValidityDiagnostics
from cwr_eg.enums import ValidityStatus
from cwr_eg.intervals import character_iou, intersection, merge_intervals, subtract_interval


def test_half_open_intervals_and_operations() -> None:
    left = CharacterInterval(0, 5)
    right = CharacterInterval(3, 8)
    assert intersection(left, right) == CharacterInterval(3, 5)
    assert character_iou(left, right) == pytest.approx(2 / 8)
    assert merge_intervals((left, right)) == (CharacterInterval(0, 8),)
    assert subtract_interval(CharacterInterval(0, 10), CharacterInterval(3, 7)) == (
        CharacterInterval(0, 3),
        CharacterInterval(7, 10),
    )


def test_touching_intervals_merge_but_do_not_intersect() -> None:
    left = CharacterInterval(0, 2)
    right = CharacterInterval(2, 4)
    assert intersection(left, right) is None
    assert merge_intervals((left, right)) == (CharacterInterval(0, 4),)


def test_contract_validation_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        CharacterInterval(1, 1)
    with pytest.raises(ValueError):
        ValidityDiagnostics(
            status=ValidityStatus.PASS,
            effective_length=10,
            mapping_coverage=1.1,
            repetition_ratio=0.0,
            code_or_template_ratio=0.0,
            language_supported=True,
        )
