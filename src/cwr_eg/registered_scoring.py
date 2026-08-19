from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable

from cwr_eg.candidates import generate_candidates, refine_candidate
from cwr_eg.contracts import CharacterInterval
from cwr_eg.hashing import sha256_file, sha256_text
from cwr_eg.manifest import read_jsonl, write_jsonl
from cwr_eg.markllm_bridge import MarkLlmBridge


# Minimum number of detector tokens required before a MarkLLM registered detector
# can score a candidate segment. Derived from the pinned MarkLLM checkout
# (commit c45ddc40f7b761beabe55a1b8dc4690e531d1c6d) algorithm configs:
#   KGW      prefix_length=1 -> needs >= prefix_length + 1 scoreable tokens.
#   Unigram  no prefix       -> needs >= 1 token, otherwise the z-score is 0/0.
#   Unbiased prefix_length=5 -> needs >= prefix_length + 1 tokens. A one-token
#                              segment reaches Qwen2 with an empty sequence after
#                              the detector drops the final token and fails during
#                              attention projection reshape.
#   SynthID  ngram_len=5     -> needs >= ngram_len tokens for its n-gram masks.
_FAMILY_MIN_DETECTOR_TOKENS: dict[str, int] = {
    "kgw": 2,
    "unigram": 1,
    "unbiased": 6,
    "synthid": 5,
}


def _minimum_detector_tokens(family: str) -> int:
    try:
        return _FAMILY_MIN_DETECTOR_TOKENS[family]
    except KeyError:
        raise ValueError(f"Unsupported watermark family: {family}")


def _is_inapplicable_detector_length_error(error: ValueError) -> bool:
    """Recognize only the observed MarkLLM short-segment applicability failure."""
    message = str(error)
    return message.startswith(
        "Must have at least 1 token to score after the first min_prefix_len="
    )


def _detector_token_count(bridge: MarkLlmBridge, text: str) -> int:
    """Return detector tokenizer length without adding synthetic special tokens."""
    encoded = bridge.tokenizer(
        text,
        add_special_tokens=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    input_ids = encoded["input_ids"]
    if hasattr(input_ids, "tolist"):
        input_ids = input_ids.tolist()
    if input_ids and isinstance(input_ids[0], list):
        if len(input_ids) != 1:
            raise RuntimeError("Unexpected batched tokenizer output in registered scoring")
        input_ids = input_ids[0]
    return len(input_ids)


def score_registered_evidence(
    *,
    bridge: MarkLlmBridge,
    score_records_path: str | Path,
    output_path: str | Path,
    families: Iterable[str],
    authorized_key_slots: Iterable[str],
    window_lengths: Iterable[int],
    stride_fraction: float,
    candidate_quantile: float,
    merge_gap_chars: int,
    include_scheme_only: bool = False,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    family_names = tuple(str(item) for item in families)
    key_slots = tuple(str(item) for item in authorized_key_slots)
    frozen_families = {"kgw", "unigram", "unbiased", "synthid"}
    if not family_names or not set(family_names).issubset(frozen_families):
        raise ValueError("Registered scoring families must be a non-empty frozen subset")
    if not key_slots or not set(key_slots).issubset({"a", "b"}):
        raise ValueError("Authorized key slots must be a non-empty subset of A/B")
    rows = read_jsonl(score_records_path)
    target = Path(output_path)
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite registered scores: {target}")
    partial = target.with_suffix(target.suffix + ".partial")
    outputs = read_jsonl(partial) if partial.exists() else []
    completed = {str(row["recipe_id"]) for row in outputs}
    selected = {str(row["recipe_id"]) for row in rows}
    if not completed.issubset(selected):
        raise RuntimeError("Partial registered scores contain an unexpected recipe id")
    if progress_callback is not None:
        progress_callback(len(outputs), len(rows))
    for row in rows:
        recipe_id = str(row["recipe_id"])
        if recipe_id in completed:
            continue
        text = str(row["text"])
        character_logits = row["character_logits"]
        mechanism_logits = row.get("mechanism_logits")
        if include_scheme_only and (
            not isinstance(mechanism_logits, dict)
            or not set(family_names).issubset(mechanism_logits)
        ):
            raise ValueError("Scheme-only scoring requires every frozen mechanism logit")
        candidates = [
            refine_candidate(candidate, character_logits)
            for candidate in generate_candidates(
                character_logits,
                tuple(int(value) for value in window_lengths),
                stride_fraction,
                candidate_quantile,
                merge_gap_chars,
            )
        ]
        interval_records: list[tuple[str, CharacterInterval, str]] = [
            (candidate.candidate_id, candidate.interval, "candidate")
            for candidate in candidates
        ]
        full_interval = CharacterInterval(0, len(text))
        if not any(interval == full_interval for _, interval, _ in interval_records):
            interval_records.append(("full-text", full_interval, "full_text"))
        evidence: list[dict[str, Any]] = []
        inapplicable_registered_tests: list[dict[str, Any]] = []
        for candidate_id, interval, interval_role in interval_records:
            segment = text[interval.char_start : interval.char_end]
            detector_token_count = _detector_token_count(bridge, segment)
            for family in family_names:
                for key_slot in key_slots:
                    key_id = f"{family}_key_{key_slot}"
                    if detector_token_count < _minimum_detector_tokens(family):
                        reason_code = (
                            "zero_detector_tokens"
                            if detector_token_count == 0
                            else "insufficient_detector_tokens"
                        )
                        inapplicable_registered_tests.append(
                            {
                                "candidate_id": candidate_id,
                                "interval_role": interval_role,
                                "scheme_family": family,
                                "key_id_hash": "sha256:" + sha256_text(key_id),
                                "char_start": interval.char_start,
                                "char_end": interval.char_end,
                                "detector_token_count": detector_token_count,
                                "applicability": "invalid",
                                "reason_codes": [
                                    interval_role,
                                    f"candidate_id:{candidate_id}",
                                    reason_code,
                                ],
                                "error_type": None,
                                "error_message": None,
                            }
                        )
                        continue
                    try:
                        detected = bridge.detect(segment, family, key_id)
                    except ValueError as error:
                        if not _is_inapplicable_detector_length_error(error):
                            raise
                        inapplicable_registered_tests.append(
                            {
                                "candidate_id": candidate_id,
                                "interval_role": interval_role,
                                "scheme_family": family,
                                "key_id_hash": "sha256:" + sha256_text(key_id),
                                "char_start": interval.char_start,
                                "char_end": interval.char_end,
                                "detector_token_count": detector_token_count,
                                "applicability": "invalid",
                                "reason_codes": [
                                    interval_role,
                                    f"candidate_id:{candidate_id}",
                                    "insufficient_detector_tokens",
                                ],
                                "error_type": type(error).__name__,
                                "error_message": str(error)[:240],
                            }
                        )
                        continue
                    evidence.append(
                        {
                            "candidate_id": candidate_id,
                            "interval_role": interval_role,
                            "detector_id": f"markllm-{family}-c45ddc4",
                            "scheme_id": family,
                            "scheme_family": family,
                            "key_id_hash": "sha256:" + sha256_text(key_id),
                            "key_status": "registered",
                            "char_start": interval.char_start,
                            "char_end": interval.char_end,
                            "raw_statistic": float(detected["raw_statistic"]),
                            "tail_direction": str(detected["raw_tail_direction"]),
                            "evidence_strength": float(detected["evidence_strength"]),
                            "evidence_transform_version": str(
                                detected["evidence_transform_version"]
                            ),
                            "single_test_p": None,
                            "adjusted_p": None,
                            "applicability": "valid",
                            "reason_codes": [interval_role, f"candidate_id:{candidate_id}"],
                        }
                    )
                if include_scheme_only:
                    evidence.append(
                        {
                            "candidate_id": candidate_id,
                            "interval_role": interval_role,
                            "detector_id": f"cwr-eg-private-mechanism-{family}-v1",
                            "scheme_id": family,
                            "scheme_family": family,
                            "key_id_hash": None,
                            "key_status": "scheme_only",
                            "char_start": interval.char_start,
                            "char_end": interval.char_end,
                            "raw_statistic": float(mechanism_logits[family]),
                            "tail_direction": "upper",
                            "evidence_strength": float(mechanism_logits[family]),
                            "evidence_transform_version": "identity-upper-tail-v1",
                            "single_test_p": None,
                            "adjusted_p": None,
                            "applicability": "valid",
                            "reason_codes": [
                                interval_role,
                                f"candidate_id:{candidate_id}",
                                "key_independent_checkpoint_mechanism_probe",
                            ],
                        }
                    )
        outputs.append(
            {
                **row,
                "authorization_scenario": "_and_".join(key_slots),
                "registered_evidence": evidence,
                "registered_search_space": {
                    "candidate_intervals": len(candidates),
                    "full_text_interval": True,
                    "families": list(family_names),
                    "authorized_key_slots": list(key_slots),
                    "scheme_only_probes": bool(include_scheme_only),
                    "tests": len(evidence),
                    "inapplicable_registered_tests": len(inapplicable_registered_tests),
                },
                "registered_inapplicable_tests": inapplicable_registered_tests,
            }
        )
        write_jsonl(partial, outputs)
        if progress_callback is not None:
            progress_callback(len(outputs), len(rows))
    partial.replace(target)
    return {
        "documents": len(outputs),
        "output_path": str(target),
        "output_sha256": sha256_file(target),
        "families": list(family_names),
        "authorized_key_slots": list(key_slots),
        "scheme_only_probes": bool(include_scheme_only),
    }
