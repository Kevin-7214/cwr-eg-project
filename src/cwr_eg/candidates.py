from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from cwr_eg.contracts import CharacterInterval
from cwr_eg.intervals import merge_intervals


@dataclass(frozen=True, slots=True)
class Candidate:
    candidate_id: str
    interval: CharacterInterval
    scale_id: str
    raw_score: float
    source: str = "generic"


def generate_candidates(
    character_scores: Sequence[float],
    window_lengths: Sequence[int],
    stride_fraction: float = 0.25,
    candidate_quantile: float = 0.95,
    merge_gap_chars: int = 0,
) -> tuple[Candidate, ...]:
    scores = np.asarray(character_scores, dtype=np.float64)
    if scores.ndim != 1 or not len(scores) or not np.all(np.isfinite(scores)):
        raise ValueError("character_scores must be finite and one-dimensional")
    if not 0.0 < stride_fraction <= 1.0:
        raise ValueError("stride_fraction must lie in (0, 1]")
    if not 0.0 <= candidate_quantile <= 1.0:
        raise ValueError("candidate_quantile must lie in [0, 1]")

    raw: list[Candidate] = []
    for length in sorted(set(int(value) for value in window_lengths)):
        if length < 1 or length > len(scores):
            continue
        stride = max(1, int(round(length * stride_fraction)))
        starts = list(range(0, len(scores) - length + 1, stride))
        final_start = len(scores) - length
        if not starts or starts[-1] != final_start:
            starts.append(final_start)
        means = np.asarray([scores[start : start + length].mean() for start in starts])
        threshold = float(np.quantile(means, candidate_quantile, method="higher"))
        for rank, (start, value) in enumerate(zip(starts, means, strict=True)):
            if value >= threshold:
                raw.append(
                    Candidate(
                        candidate_id=f"generic-L{length}-{rank}",
                        interval=CharacterInterval(start, start + length),
                        scale_id=f"char-{length}",
                        raw_score=float(value),
                    )
                )

    merged_intervals = merge_intervals((item.interval for item in raw), merge_gap_chars)
    result: list[Candidate] = []
    for index, interval in enumerate(merged_intervals):
        contributors = [item for item in raw if _overlaps(item.interval, interval)]
        best = max(contributors, key=lambda item: item.raw_score)
        result.append(
            Candidate(
                candidate_id=f"generic-merged-{index}",
                interval=interval,
                scale_id=best.scale_id,
                raw_score=best.raw_score,
            )
        )
    return tuple(result)


def refine_candidate(
    candidate: Candidate, character_scores: Sequence[float], quantile: float = 0.5
) -> Candidate:
    scores = np.asarray(character_scores, dtype=np.float64)
    start, end = candidate.interval.char_start, candidate.interval.char_end
    local = scores[start:end]
    if len(local) < 2:
        return candidate
    threshold = float(np.quantile(local, quantile))
    selected = np.flatnonzero(local >= threshold)
    if not len(selected):
        return candidate
    interval = CharacterInterval(start + int(selected[0]), start + int(selected[-1]) + 1)
    return Candidate(
        candidate_id=candidate.candidate_id + "-refined",
        interval=interval,
        scale_id=candidate.scale_id,
        raw_score=float(local[selected].mean()),
        source=candidate.source,
    )


def _overlaps(left: CharacterInterval, right: CharacterInterval) -> bool:
    return left.char_start < right.char_end and right.char_start < left.char_end
