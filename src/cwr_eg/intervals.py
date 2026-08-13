from __future__ import annotations

from collections.abc import Iterable

from cwr_eg.contracts import CharacterInterval


def intersection(
    left: CharacterInterval, right: CharacterInterval
) -> CharacterInterval | None:
    start = max(left.char_start, right.char_start)
    end = min(left.char_end, right.char_end)
    return CharacterInterval(start, end) if start < end else None


def character_iou(left: CharacterInterval, right: CharacterInterval) -> float:
    overlap = intersection(left, right)
    if overlap is None:
        return 0.0
    union = left.length + right.length - overlap.length
    return overlap.length / union


def merge_intervals(
    intervals: Iterable[CharacterInterval], max_gap: int = 0
) -> tuple[CharacterInterval, ...]:
    if max_gap < 0:
        raise ValueError("max_gap cannot be negative")
    ordered = sorted(intervals)
    if not ordered:
        return ()
    merged: list[CharacterInterval] = [ordered[0]]
    for current in ordered[1:]:
        previous = merged[-1]
        if current.char_start <= previous.char_end + max_gap:
            merged[-1] = CharacterInterval(
                previous.char_start, max(previous.char_end, current.char_end)
            )
        else:
            merged.append(current)
    return tuple(merged)


def subtract_interval(
    source: CharacterInterval, cut: CharacterInterval
) -> tuple[CharacterInterval, ...]:
    overlap = intersection(source, cut)
    if overlap is None:
        return (source,)
    result: list[CharacterInterval] = []
    if source.char_start < overlap.char_start:
        result.append(CharacterInterval(source.char_start, overlap.char_start))
    if overlap.char_end < source.char_end:
        result.append(CharacterInterval(overlap.char_end, source.char_end))
    return tuple(result)
