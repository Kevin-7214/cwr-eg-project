from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
import math

from cwr_eg.contracts import ValidityDiagnostics
from cwr_eg.enums import ValidityStatus


def normalized_entropy(items: Sequence[int | str]) -> float | None:
    if len(items) < 2:
        return None
    counts = Counter(items)
    if len(counts) < 2:
        return 0.0
    total = len(items)
    entropy = -sum((count / total) * math.log(count / total) for count in counts.values())
    return entropy / math.log(len(counts))


def repetition_ratio(text: str, ngram: int = 4) -> float:
    if ngram < 1 or len(text) < ngram:
        return 0.0
    grams = [text[index : index + ngram] for index in range(len(text) - ngram + 1)]
    return 1.0 - len(set(grams)) / len(grams)


def evaluate_validity(
    *,
    text: str,
    language: str,
    effective_length: int,
    mapping_coverage_value: float,
    token_ids: Sequence[int] = (),
    code_or_template_ratio: float = 0.0,
    domain_shift_score: float | None = None,
    minimum_effective_length: int = 64,
    minimum_mapping_coverage: float = 0.98,
    maximum_repetition_ratio: float = 0.35,
    supported_languages: Sequence[str] = ("en", "zh"),
    warning_forces_uncertain: bool = False,
) -> ValidityDiagnostics:
    reasons: list[str] = []
    repeated = repetition_ratio(text)
    if effective_length < minimum_effective_length:
        reasons.append("insufficient_effective_length")
    if mapping_coverage_value < minimum_mapping_coverage:
        reasons.append("mapping_coverage_low")
    if language not in supported_languages:
        reasons.append("unsupported_language")
    if repeated > maximum_repetition_ratio:
        reasons.append("excessive_repetition")
    status = ValidityStatus.FAIL if reasons else ValidityStatus.PASS
    if warning_forces_uncertain and domain_shift_score is not None and domain_shift_score > 1.0:
        status = ValidityStatus.FAIL
        reasons.append("calibration_domain_shift")
    return ValidityDiagnostics(
        status=status,
        effective_length=effective_length,
        token_entropy=normalized_entropy(token_ids),
        mapping_coverage=mapping_coverage_value,
        repetition_ratio=repeated,
        code_or_template_ratio=code_or_template_ratio,
        domain_shift_score=domain_shift_score,
        language_supported=language in supported_languages,
        reason_codes=tuple(reasons),
        details={},
    )
