from __future__ import annotations

import math

from cwr_eg.contracts import CharacterInterval
from cwr_eg.enums import DecisionLabel
from cwr_eg.metrics import EvaluationRecord, evaluate_records


def test_extended_metrics_include_parent_fwer_oscr_localization_and_strata() -> None:
    records = [
        EvaluationRecord(
            parent_id="null-a",
            true_label=DecisionLabel.NONE,
            predicted_label=DecisionLabel.NONE,
            score=0.1,
            language="en",
        ),
        EvaluationRecord(
            parent_id="null-a",
            true_label=DecisionLabel.NONE,
            predicted_label=DecisionLabel.KNOWN_SCHEME_KNOWN_KEY,
            score=0.8,
            language="en",
            attack_id="paraphrase",
        ),
        EvaluationRecord(
            parent_id="null-b",
            true_label=DecisionLabel.NONE,
            predicted_label=DecisionLabel.NONE,
            score=0.2,
            language="zh",
        ),
        EvaluationRecord(
            parent_id="known",
            true_label=DecisionLabel.KNOWN_SCHEME_KNOWN_KEY,
            predicted_label=DecisionLabel.KNOWN_SCHEME_KNOWN_KEY,
            score=0.9,
            knownness_score=0.9,
            true_intervals=(CharacterInterval(0, 10),),
            predicted_intervals=(CharacterInterval(1, 11),),
            language="en",
            watermark_family="kgw",
        ),
        EvaluationRecord(
            parent_id="unknown",
            true_label=DecisionLabel.SUSPECTED_UNKNOWN_SCHEME,
            predicted_label=DecisionLabel.SUSPECTED_UNKNOWN_SCHEME,
            score=0.5,
            knownness_score=0.1,
            true_intervals=(CharacterInterval(2, 6),),
            predicted_intervals=(CharacterInterval(2, 6),),
            language="zh",
        ),
    ]
    result = evaluate_records(records, stratify_by=("language", "attack_id"))
    assert result["parent_fwer"]["trials"] == 2
    assert result["parent_fwer"]["errors"] == 1
    assert result["parent_fwer"]["rate"] == 0.5
    assert not math.isnan(result["oscr"])
    assert 0.0 <= result["mean_character_iou"] <= 1.0
    assert result["boundary_error"]["matched_events"] == 2
    assert set(result["stratified"]["language"]) == {"en", "zh"}
