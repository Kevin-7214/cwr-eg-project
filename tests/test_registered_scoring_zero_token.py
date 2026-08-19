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
        # Whitespace-only candidate is zero tokens; normal text is scoreable.
        ids = [] if not text.strip() else list(range(max(2, len(text.strip()))))
        return _Encoding(input_ids=ids)


class _Bridge:
    tokenizer = _Tokenizer()

    def detect(self, text: str, family: str, key_id: str):
        # A zero-token segment must never reach any detector.
        assert text.strip()
        return {
            "raw_statistic": 1.0,
            "raw_tail_direction": "upper",
            "evidence_strength": 1.0,
            "evidence_transform_version": "identity-upper-tail-v1",
        }


def test_zero_token_candidate_is_explicitly_inapplicable(tmp_path, monkeypatch) -> None:
    source = tmp_path / "scores.jsonl"
    write_jsonl(
        source,
        [
            {
                "document_id": "doc-zero-token",
                "recipe_id": "doc-zero-token",
                "text": "    abcdefghij",
                "character_logits": [10.0, 10.0, 10.0, 10.0] + [0.0] * 10,
            }
        ],
    )

    # Force one whitespace-only candidate plus the normal full-text interval.
    import cwr_eg.registered_scoring as rs
    from cwr_eg.contracts import CharacterInterval

    class _Candidate:
        candidate_id = "candidate-whitespace"
        interval = CharacterInterval(0, 4)

    monkeypatch.setattr(rs, "generate_candidates", lambda *args, **kwargs: [_Candidate()])
    monkeypatch.setattr(rs, "refine_candidate", lambda candidate, logits: candidate)

    result = score_registered_evidence(
        bridge=_Bridge(),
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
    invalid = row["registered_inapplicable_tests"]
    assert len(invalid) == 8
    assert {item["reason_codes"][-1] for item in invalid} == {"zero_detector_tokens"}
    assert {item["detector_token_count"] for item in invalid} == {0}
    # The full-text interval is still evaluated normally.
    assert any(
        item["interval_role"] == "full_text"
        for item in row["registered_evidence"]
    )
