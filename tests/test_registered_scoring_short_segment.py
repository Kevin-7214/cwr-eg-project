from __future__ import annotations

from cwr_eg.manifest import read_jsonl, write_jsonl
from cwr_eg.registered_scoring import score_registered_evidence


class _Encoding(dict):
    pass


class _Tokenizer:
    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        return_attention_mask: bool,
        return_token_type_ids: bool,
    ):
        assert add_special_tokens is False
        # Report a comfortably scoreable token count so the family-minimum
        # pre-check does not interfere; this test isolates the ValueError path.
        return _Encoding(input_ids=list(range(100 if text else 0)))


class _ShortSegmentBridge:
    tokenizer = _Tokenizer()

    def detect(self, text: str, family: str, key_id: str):
        if family == "kgw" and len(text) < 10:
            raise ValueError(
                "Must have at least 1 token to score after the first min_prefix_len=1 tokens required by the seeding scheme."
            )
        if family == "unbiased":
            return {
                "raw_statistic": 0.01,
                "raw_tail_direction": "lower",
                "evidence_strength": -0.01,
                "evidence_transform_version": "negate-lower-tail-statistic-v1",
            }
        return {
            "raw_statistic": 1.5,
            "raw_tail_direction": "upper",
            "evidence_strength": 1.5,
            "evidence_transform_version": "identity-upper-tail-v1",
        }


def test_short_kgw_candidate_is_inapplicable_not_fatal(tmp_path) -> None:
    source = tmp_path / "scores.jsonl"
    write_jsonl(
        source,
        [
            {
                "document_id": "doc-short",
                "recipe_id": "doc-short",
                "text": "abcdefghij",
                "character_logits": [float(i) for i in range(10)],
            }
        ],
    )

    result = score_registered_evidence(
        bridge=_ShortSegmentBridge(),
        score_records_path=source,
        output_path=tmp_path / "registered.jsonl",
        families=("kgw", "unigram", "unbiased", "synthid"),
        authorized_key_slots=("a", "b"),
        window_lengths=(4,),
        stride_fraction=0.5,
        candidate_quantile=0.9,
        merge_gap_chars=0,
    )

    assert result["documents"] == 1
    row = read_jsonl(tmp_path / "registered.jsonl")[0]
    skipped = row["registered_inapplicable_tests"]
    assert skipped
    assert {item["scheme_family"] for item in skipped} == {"kgw"}
    assert {
        item["reason_codes"][-1] for item in skipped
    } == {"insufficient_detector_tokens"}

    # Full text remains scoreable for KGW; only too-short candidate tests are omitted.
    kgw_valid = [
        item
        for item in row["registered_evidence"]
        if item["scheme_family"] == "kgw"
    ]
    assert kgw_valid
    assert any(item["interval_role"] == "full_text" for item in kgw_valid)
    assert row["registered_search_space"]["inapplicable_registered_tests"] == len(skipped)
