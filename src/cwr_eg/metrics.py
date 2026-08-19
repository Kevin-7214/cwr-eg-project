from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from cwr_eg.calibration import clopper_pearson_upper
from cwr_eg.contracts import CharacterInterval
from cwr_eg.enums import DecisionLabel
from cwr_eg.intervals import character_iou, intersection, merge_intervals


@dataclass(frozen=True, slots=True)
class EvaluationRecord:
    parent_id: str
    true_label: DecisionLabel
    predicted_label: DecisionLabel
    score: float
    true_intervals: tuple[CharacterInterval, ...] = ()
    predicted_intervals: tuple[CharacterInterval, ...] = ()
    knownness_score: float | None = None
    source: str | None = None
    language: str | None = None
    watermark_family: str | None = None
    key_id: str | None = None
    attack_id: str | None = None


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


def per_class_metrics(records: Sequence[EvaluationRecord]) -> dict[str, dict[str, float | int]]:
    result: dict[str, dict[str, float | int]] = {}
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
        precision_denominator = true_positive + false_positive
        recall_denominator = true_positive + false_negative
        precision = (
            0.0 if precision_denominator == 0 else true_positive / precision_denominator
        )
        recall = 0.0 if recall_denominator == 0 else true_positive / recall_denominator
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        result[label.value] = {
            "support": recall_denominator,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    return result


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


def parent_fwer(
    records: Sequence[EvaluationRecord], confidence_level: float = 0.95
) -> dict[str, float | int]:
    null_by_parent: dict[str, list[EvaluationRecord]] = defaultdict(list)
    for record in records:
        if record.true_label is DecisionLabel.NONE:
            null_by_parent[record.parent_id].append(record)
    errors = sum(
        any(item.predicted_label is not DecisionLabel.NONE for item in descendants)
        for descendants in null_by_parent.values()
    )
    trials = len(null_by_parent)
    return {
        "errors": errors,
        "trials": trials,
        "rate": math.nan if trials == 0 else errors / trials,
        "upper": math.nan
        if trials == 0
        else clopper_pearson_upper(errors, trials, confidence_level),
        "cluster_unit": "parent_id",
        "descendant_rule": "any_false_positive",
    }


def oscr(records: Sequence[EvaluationRecord]) -> float:
    known_labels = {
        DecisionLabel.KNOWN_SCHEME_KNOWN_KEY,
        DecisionLabel.KNOWN_SCHEME_UNKNOWN_KEY,
    }
    known = [record for record in records if record.true_label in known_labels]
    unknown = [
        record
        for record in records
        if record.true_label is DecisionLabel.SUSPECTED_UNKNOWN_SCHEME
    ]
    if not known or not unknown or any(
        record.knownness_score is None for record in known + unknown
    ):
        return math.nan
    thresholds = sorted(
        {float(record.knownness_score) for record in known + unknown}, reverse=True
    )
    points: list[tuple[float, float]] = [(0.0, 0.0)]
    for threshold in thresholds:
        correct_known = sum(
            record.predicted_label is record.true_label
            and float(record.knownness_score) >= threshold
            for record in known
        )
        false_known = sum(
            float(record.knownness_score) >= threshold for record in unknown
        )
        points.append((false_known / len(unknown), correct_known / len(known)))
    points.append(
        (
            1.0,
            sum(record.predicted_label is record.true_label for record in known)
            / len(known),
        )
    )
    deduplicated: dict[float, float] = {}
    for false_positive_rate, correct_classification_rate in points:
        deduplicated[false_positive_rate] = max(
            deduplicated.get(false_positive_rate, 0.0), correct_classification_rate
        )
    ordered = sorted(deduplicated.items())
    x_values = np.asarray([item[0] for item in ordered], dtype=np.float64)
    y_values = np.asarray([item[1] for item in ordered], dtype=np.float64)
    return float(np.sum(np.diff(x_values) * (y_values[:-1] + y_values[1:]) / 2.0))


def character_set_iou(
    truth: Sequence[CharacterInterval], prediction: Sequence[CharacterInterval]
) -> float:
    merged_truth = merge_intervals(truth)
    merged_prediction = merge_intervals(prediction)
    truth_length = sum(item.length for item in merged_truth)
    prediction_length = sum(item.length for item in merged_prediction)
    overlap = sum(
        common.length
        for true_interval in merged_truth
        for predicted_interval in merged_prediction
        if (common := intersection(true_interval, predicted_interval)) is not None
    )
    union = truth_length + prediction_length - overlap
    return 1.0 if union == 0 else overlap / union


def boundary_error(
    truth: Sequence[CharacterInterval], prediction: Sequence[CharacterInterval]
) -> dict[str, float | int]:
    candidates = sorted(
        (
            (character_iou(true_interval, predicted_interval), true_index, predicted_index)
            for true_index, true_interval in enumerate(truth)
            for predicted_index, predicted_interval in enumerate(prediction)
        ),
        reverse=True,
    )
    matched_truth: set[int] = set()
    matched_prediction: set[int] = set()
    errors: list[float] = []
    for overlap, true_index, predicted_index in candidates:
        if overlap <= 0.0:
            continue
        if true_index in matched_truth or predicted_index in matched_prediction:
            continue
        matched_truth.add(true_index)
        matched_prediction.add(predicted_index)
        true_interval = truth[true_index]
        predicted_interval = prediction[predicted_index]
        errors.append(
            (
                abs(true_interval.char_start - predicted_interval.char_start)
                + abs(true_interval.char_end - predicted_interval.char_end)
            )
            / 2.0
        )
    return {
        "mean_absolute_codepoints": float(np.mean(errors)) if errors else math.nan,
        "matched_events": len(errors),
        "unmatched_true_events": len(truth) - len(matched_truth),
        "unmatched_predicted_events": len(prediction) - len(matched_prediction),
    }


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


def _evaluate_core(records: Sequence[EvaluationRecord]) -> dict[str, Any]:
    boundary_rows = [
        boundary_error(record.true_intervals, record.predicted_intervals)
        for record in records
    ]
    matched_boundary = [
        float(row["mean_absolute_codepoints"])
        for row in boundary_rows
        if math.isfinite(float(row["mean_absolute_codepoints"]))
    ]
    return {
        "records": len(records),
        "macro_f1": macro_f1(records),
        "per_class": per_class_metrics(records),
        "confusion_matrix": confusion_matrix(records),
        "oscr": oscr(records),
        "document_fwer": document_fwer(records),
        "parent_fwer": parent_fwer(records),
        "mean_character_iou": float(
            np.mean(
                [
                    character_set_iou(record.true_intervals, record.predicted_intervals)
                    for record in records
                ]
            )
        )
        if records
        else math.nan,
        "boundary_error": {
            "mean_absolute_codepoints": float(np.mean(matched_boundary))
            if matched_boundary
            else math.nan,
            "matched_events": sum(int(row["matched_events"]) for row in boundary_rows),
            "unmatched_true_events": sum(
                int(row["unmatched_true_events"]) for row in boundary_rows
            ),
            "unmatched_predicted_events": sum(
                int(row["unmatched_predicted_events"]) for row in boundary_rows
            ),
        },
        "mean_event_f1": float(
            np.mean(
                [event_f1(record.true_intervals, record.predicted_intervals) for record in records]
            )
        )
        if records
        else math.nan,
    }


def evaluate_records(
    records: Sequence[EvaluationRecord],
    *,
    stratify_by: Sequence[str] = (),
) -> dict[str, Any]:
    result = _evaluate_core(records)
    stratified: dict[str, dict[str, Any]] = {}
    supported = {"source", "language", "watermark_family", "key_id", "attack_id"}
    for field_name in stratify_by:
        if field_name not in supported:
            raise ValueError(f"Unsupported evaluation stratum: {field_name}")
        groups: dict[str, list[EvaluationRecord]] = defaultdict(list)
        for record in records:
            value = getattr(record, field_name)
            groups["null" if value is None else str(value)].append(record)
        stratified[field_name] = {
            value: _evaluate_core(group) for value, group in sorted(groups.items())
        }
    result["stratified"] = stratified
    return result
