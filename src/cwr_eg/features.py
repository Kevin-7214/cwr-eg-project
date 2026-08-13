from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from cwr_eg.contracts import CharacterInterval


@dataclass(frozen=True, slots=True)
class FeatureView:
    name: str
    values: np.ndarray
    raw_intervals: tuple[CharacterInterval | None, ...]
    valid_mask: np.ndarray
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        values = np.asarray(self.values)
        mask = np.asarray(self.valid_mask)
        if values.ndim != 2:
            raise ValueError("Feature values must have shape (positions, dimensions)")
        if mask.shape != (values.shape[0],):
            raise ValueError("Feature mask must match the position axis")
        if len(self.raw_intervals) != values.shape[0]:
            raise ValueError("Each feature position requires a raw interval mapping")


@dataclass(frozen=True, slots=True)
class ExtractedViews:
    document_id: str
    views: Mapping[str, FeatureView]
    normalization_version: str
    extractor_version: str


class GenericFeatureExtractor(ABC):
    """Single-text inference interface; paired texts are intentionally absent."""

    @abstractmethod
    def extract(self, document_id: str, raw_text: str, language: str) -> ExtractedViews:
        raise NotImplementedError


def project_token_values_to_characters(
    token_values: np.ndarray,
    token_intervals: tuple[CharacterInterval | None, ...],
    raw_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(token_values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or values.shape[0] != len(token_intervals):
        raise ValueError("Token values and intervals are not aligned")
    sums = np.zeros((raw_length, values.shape[1]), dtype=np.float64)
    counts = np.zeros(raw_length, dtype=np.int64)
    for value, interval in zip(values, token_intervals, strict=True):
        if interval is None:
            continue
        if interval.char_end > raw_length:
            raise ValueError("Token interval exceeds raw text")
        sums[interval.char_start : interval.char_end] += value
        counts[interval.char_start : interval.char_end] += 1
    valid = counts > 0
    sums[valid] /= counts[valid, None]
    return sums, valid


def stack_character_views(
    views: Mapping[str, tuple[np.ndarray, np.ndarray]], raw_length: int
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    joint_mask = np.ones(raw_length, dtype=bool)
    for name, (values, mask) in views.items():
        array = np.asarray(values, dtype=np.float32)
        valid = np.asarray(mask, dtype=bool)
        if array.shape[0] != raw_length or valid.shape != (raw_length,):
            raise ValueError(f"Character view {name} has incompatible shape")
        arrays[name] = array
        joint_mask &= valid
    return arrays, joint_mask
