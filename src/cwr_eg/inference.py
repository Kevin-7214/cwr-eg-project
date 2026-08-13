from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from cwr_eg.bundle import CalibrationBundle
from cwr_eg.candidates import Candidate, generate_candidates, refine_candidate
from cwr_eg.contracts import (
    DocumentDecision,
    GenericResidualEvidence,
    RegisteredEvidence,
)
from cwr_eg.decision import aggregate_document, decide_segment
from cwr_eg.hashing import sha256_text
from cwr_eg.validity import evaluate_validity


@dataclass(frozen=True, slots=True)
class InferenceVersions:
    model_version: str
    calibration_id: str
    normalization_version: str
    manifest_version: str
    code_revision: str


RegisteredScorer = Callable[[str, Candidate], Sequence[RegisteredEvidence]]


class InferencePipeline:
    def __init__(
        self,
        *,
        calibration: CalibrationBundle,
        registered_scorer: RegisteredScorer,
        versions: InferenceVersions,
        alpha: float = 0.01,
        window_lengths: Sequence[int] = (256, 512, 1024),
        stride_fraction: float = 0.25,
        candidate_quantile: float = 0.95,
        merge_gap_chars: int = 32,
    ) -> None:
        self.calibration = calibration
        self.registered_scorer = registered_scorer
        self.versions = versions
        self.alpha = alpha
        self.window_lengths = tuple(window_lengths)
        self.stride_fraction = stride_fraction
        self.candidate_quantile = candidate_quantile
        self.merge_gap_chars = merge_gap_chars

    def infer_from_character_scores(
        self,
        *,
        document_id: str,
        raw_text: str,
        language: str,
        character_scores: Sequence[float],
        effective_length: int,
        mapping_coverage: float,
        runtime: Mapping[str, Any] | None = None,
    ) -> DocumentDecision:
        scores = np.asarray(character_scores, dtype=np.float64)
        if scores.shape != (len(raw_text),):
            raise ValueError("Character scores must align with raw Unicode code points")
        candidates = tuple(
            refine_candidate(candidate, scores)
            for candidate in generate_candidates(
                scores,
                self.window_lengths,
                self.stride_fraction,
                self.candidate_quantile,
                self.merge_gap_chars,
            )
        )
        stratum = f"{language}:all"
        document_maximum = max((candidate.raw_score for candidate in candidates), default=0.0)
        document_generic_p = self.calibration.p_value("generic", stratum, document_maximum)
        segments = []
        all_registered_p: list[float] = []
        all_gap_p: list[float] = []
        for candidate in candidates:
            registered = tuple(self.registered_scorer(raw_text, candidate))
            best_registered_statistic = max(
                (item.raw_statistic for item in registered), default=0.0
            )
            gap_score = candidate.raw_score - best_registered_statistic
            gap_p = self.calibration.p_value("gap", stratum, gap_score)
            all_gap_p.append(gap_p)
            registered_document_p = self.calibration.p_value(
                "registered", stratum, best_registered_statistic
            )
            calibrated_registered = tuple(
                RegisteredEvidence(
                    detector_id=item.detector_id,
                    scheme_id=item.scheme_id,
                    scheme_family=item.scheme_family,
                    key_id_hash=item.key_id_hash,
                    key_status=item.key_status,
                    interval=item.interval,
                    raw_statistic=item.raw_statistic,
                    tail_direction=item.tail_direction,
                    single_test_p=item.single_test_p,
                    adjusted_p=max(item.adjusted_p or 0.0, registered_document_p),
                    applicability=item.applicability,
                    reason_codes=item.reason_codes,
                )
                for item in registered
            )
            all_registered_p.append(registered_document_p)
            generic = GenericResidualEvidence(
                evidence_id="generic-" + sha256_text(f"{document_id}:{candidate.candidate_id}")[:20],
                document_id=document_id,
                interval=candidate.interval,
                scale_id=candidate.scale_id,
                raw_score=candidate.raw_score,
                generic_p=document_generic_p,
                effective_length=effective_length,
                model_version=self.versions.model_version,
                calibration_id=self.versions.calibration_id,
            )
            validity = evaluate_validity(
                text=raw_text[candidate.interval.char_start : candidate.interval.char_end],
                language=language,
                effective_length=effective_length,
                mapping_coverage_value=mapping_coverage,
                **self.calibration.validity_rules,
            )
            segments.append(
                decide_segment(
                    segment_id="segment-" + candidate.candidate_id,
                    interval=candidate.interval,
                    generic=generic,
                    registered=calibrated_registered,
                    validity=validity,
                    gap_score=gap_score,
                    gap_p=gap_p,
                    alpha=self.alpha,
                    normalization_version=self.versions.normalization_version,
                )
            )
        document_validity = evaluate_validity(
            text=raw_text,
            language=language,
            effective_length=effective_length,
            mapping_coverage_value=mapping_coverage,
            **self.calibration.validity_rules,
        )
        return aggregate_document(
            document_id=document_id,
            segments=segments,
            document_validity=document_validity,
            document_generic_p=document_generic_p,
            document_registered_p=min(all_registered_p, default=None),
            document_gap_p=min(all_gap_p, default=None),
            search_space_summary={
                "window_scales": list(self.window_lengths),
                "candidate_count": len(candidates),
                "registered_test_count": sum(
                    len(segment.registered_candidates) for segment in segments
                ),
            },
            versions={
                "model_version": self.versions.model_version,
                "calibration_id": self.versions.calibration_id,
                "normalization_version": self.versions.normalization_version,
                "manifest_version": self.versions.manifest_version,
                "code_revision": self.versions.code_revision,
            },
            runtime=dict(runtime or {}),
        )
