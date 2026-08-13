from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
import unicodedata

from cwr_eg.contracts import CharacterInterval


@dataclass(frozen=True, slots=True)
class NormalizedText:
    raw_text: str
    normalized_text: str
    raw_to_normalized: tuple[int, ...]
    normalization_form: str

    def __post_init__(self) -> None:
        if len(self.raw_to_normalized) != len(self.raw_text) + 1:
            raise ValueError("A mapping entry is required for every raw boundary")
        if self.raw_to_normalized[0] != 0:
            raise ValueError("Boundary mapping must start at zero")
        if self.raw_to_normalized[-1] != len(self.normalized_text):
            raise ValueError("Boundary mapping must end at normalized length")
        if any(
            left > right
            for left, right in zip(
                self.raw_to_normalized, self.raw_to_normalized[1:], strict=False
            )
        ):
            raise ValueError("Boundary mapping must be monotone")

    def normalized_span_to_raw(
        self, start: int, end: int
    ) -> CharacterInterval:
        if start < 0 or end <= start or end > len(self.normalized_text):
            raise ValueError("Invalid normalized half-open interval")
        raw_start = max(0, bisect_right(self.raw_to_normalized, start) - 1)
        raw_end = min(
            len(self.raw_text), bisect_right(self.raw_to_normalized, end) - 1
        )
        if raw_end <= raw_start:
            raw_end = min(len(self.raw_text), raw_start + 1)
        return CharacterInterval(raw_start, raw_end)

    def map_token_offsets(
        self, offsets: list[tuple[int, int]]
    ) -> tuple[CharacterInterval | None, ...]:
        mapped: list[CharacterInterval | None] = []
        for start, end in offsets:
            if start == end:
                mapped.append(None)
            else:
                mapped.append(self.normalized_span_to_raw(start, end))
        return tuple(mapped)


class TextNormalizer:
    def __init__(self, form: str = "NFKC") -> None:
        if form not in {"NFC", "NFKC"}:
            raise ValueError("Only NFC and NFKC are supported")
        self.form = form

    def normalize(self, raw_text: str) -> NormalizedText:
        normalized = unicodedata.normalize(self.form, raw_text)
        boundaries = tuple(
            len(unicodedata.normalize(self.form, raw_text[:index]))
            for index in range(len(raw_text) + 1)
        )
        return NormalizedText(raw_text, normalized, boundaries, self.form)


def mapping_coverage(
    offsets: list[tuple[int, int]], mapped: tuple[CharacterInterval | None, ...]
) -> float:
    expected = sum(start != end for start, end in offsets)
    if expected == 0:
        return 0.0
    return sum(item is not None for item in mapped) / expected
