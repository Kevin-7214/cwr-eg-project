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
        return _Encoding(input_ids=list(range(len(text))))


class _Bridge:
    tokenizer = _Tokenizer()

    def detect(self, text: str, family: str, key_id: str):
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


def test_registered_scoring_covers_candidates_full_text_and_key_masks(tmp_path) -> None:
    source = tmp_path / "scores.jsonl"
    write_jsonl(
        source,
        [
            {
                "document_id": "doc-1",
                "recipe_id": "doc-1",
                "text": "abcdefghij",
                "character_logits": [float(index) for index in range(10)],
            }
        ],
    )
    result = score_registered_evidence(
        bridge=_Bridge(),
        score_records_path=source,
        output_path=tmp_path / "registered.jsonl",
        families=("kgw", "unigram", "unbiased", "synthid"),
        authorized_key_slots=("a",),
        window_lengths=(4,),
        stride_fraction=0.5,
        candidate_quantile=0.9,
        merge_gap_chars=0,
    )
    row = read_jsonl(tmp_path / "registered.jsonl")[0]
    assert result["documents"] == 1
    assert row["registered_search_space"]["full_text_interval"] is True
    assert row["registered_search_space"]["authorized_key_slots"] == ["a"]
    assert {item["scheme_family"] for item in row["registered_evidence"]} == {
        "kgw",
        "unigram",
        "unbiased",
        "synthid",
    }
    unbiased = [
        item for item in row["registered_evidence"] if item["scheme_family"] == "unbiased"
    ]
    assert all(item["raw_statistic"] == 0.01 for item in unbiased)
    assert all(item["evidence_strength"] == -0.01 for item in unbiased)


def test_registered_scoring_adds_key_independent_scheme_probes(tmp_path) -> None:
    source = tmp_path / "scores.jsonl"
    write_jsonl(
        source,
        [
            {
                "document_id": "doc-1",
                "recipe_id": "doc-1",
                "text": "abcdefghij",
                "character_logits": [float(index) for index in range(10)],
                "mechanism_logits": {
                    "kgw": 0.1,
                    "unigram": 0.2,
                    "unbiased": 0.3,
                    "synthid": 0.4,
                },
            }
        ],
    )
    score_registered_evidence(
        bridge=_Bridge(),
        score_records_path=source,
        output_path=tmp_path / "registered.jsonl",
        families=("kgw", "unigram", "unbiased", "synthid"),
        authorized_key_slots=("a",),
        window_lengths=(4,),
        stride_fraction=0.5,
        candidate_quantile=0.9,
        merge_gap_chars=0,
        include_scheme_only=True,
    )
    row = read_jsonl(tmp_path / "registered.jsonl")[0]
    scheme_only = [
        item for item in row["registered_evidence"] if item["key_status"] == "scheme_only"
    ]
    assert scheme_only
    assert {item["key_id_hash"] for item in scheme_only} == {None}
    assert row["registered_search_space"]["scheme_only_probes"] is True
