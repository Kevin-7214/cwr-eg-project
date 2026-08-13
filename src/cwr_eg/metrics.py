from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from cwr_eg.calibration import clopper_pearson_upper
from cwr_eg.contracts import CharacterInterval
from cwr_eg.enums import DecisionLabel
from cwr_eg.intervals import character_iou


@dataclass(frozen=True, slots=True)
class EvaluationRecord:
    parent_id: str
    true_label: DecisionLabel
    predicted_label: DecisionLabel
    score: float
    true_intervals: tuple[CharacterInterval, ...] = ()
    predicted_intervals: tuple[CharacterInterval, ...] = ()


def confusion_matrix(records: Sequence[EvaluationRecord]) -> dict[str, dict[str, int]]:
    labels = tuple(DecisionLabel)
    counts = Counter((record.true_label, record.predicted_label) for record in records)
    return {
        truth.value: {
            prediction.value: counts[(truth, prediction)] for prediction in labels
        }
        for truth in labels
    }


def macro_f1(records: Sequence[EvaluationRecord]) -> float:
    scores: list[float] = []
    for label in DecisionLabel:
        true_positive = sum(
            record.true_label is label and record.predicted_label is label for record in records
        )
        false_positive = sum(
            record.true_label is not label and record.predicted_label is label for record in records
        )
        false_negative = sum(
            record.true_label is label and record.predicted_label is not label for record in records
        )
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
    return float(np.mean(scores))


def event_f1(
    truth: Sequence[CharacterInterval],
    prediction: Sequence[CharacterInterval],
    iou_threshold: float = 0.5,
) -> float:
    pairs = sorted(
        (
            (character_iou(true_interval, predicted_interval), true_index, predicted_index)
            for true_index, true_interval in enumerate(truth)
            for predicted_index, predicted_interval in enumerate(prediction)
        ),
        reverse=True,
    )
    matched_truth: set[int] = set()
    matched_prediction: set[int] = set()
    for score, true_index, predicted_index in pairs:
        if score < iou_threshold:
            break
        if true_index not in matched_truth and predicted_index not in matched_prediction:
            matched_truth.add(true_index)
            matched_prediction.add(predicted_index)
    true_positive = len(matched_truth)
    denominator = 2 * true_positive + len(prediction) - true_positive + len(truth) - true_positive
    return 1.0 if denominator == 0 else 2 * true_positive / denominator


def document_fwer(
    records: Sequence[EvaluationRecord], confidence_level: float = 0.95
) -> dict[str, float | int]:
    null = [record for record in records if record.true_label is DecisionLabel.NONE]
    errors = sum(record.predicted_label is not DecisionLabel.NONE for record in null)
    rate = math.nan if not null else errors / len(null)
    upper = math.nan if not null else clopper_pearson_upper(errors, len(null), confidence_level)
    return {"errors": errors, "trials": len(null), "rate": rate, "upper": upper}


def cluster_bootstrap_macro_f1(
    records: Sequence[EvaluationRecord],
    *,
    replicates: int = 2000,
    seed: int = 20260813,
) -> dict[str, Any]:
    grouped: dict[str, list[EvaluationRecord]] = {}
    for record in records:
        grouped.setdefault(record.parent_id, []).append(record)
    parent_ids = sorted(grouped)
    if len(parent_ids) < 2 or replicates < 1:
        raise ValueError("Cluster bootstrap needs at least two parents and one replicate")
    generator = np.random.default_rng(seed)
    estimates = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        sampled = generator.choice(parent_ids, size=len(parent_ids), replace=True)
        sample_records = [record for parent_id in sampled for record in grouped[parent_id]]
        estimates[index] = macro_f1(sample_records)
    return {
        "estimate": macro_f1(records),
        "lower": float(np.quantile(estimates, 0.025)),
        "upper": float(np.quantile(estimates, 0.975)),
        "replicates": replicates,
        "cluster_unit": "parent_id",
    }


def evaluate_records(records: Sequence[EvaluationRecord]) -> dict[str, Any]:
    return {
        "records": len(records),
        "macro_f1": macro_f1(records),
        "confusion_matrix": confusion_matrix(records),
        "fwer": document_fwer(records),
        "mean_event_f1": float(
            np.mean(
                [event_f1(record.true_intervals, record.predicted_intervals) for record in records]
            )
        )
        if records
        else math.nan,
    }
