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


class _RecordingBridge:
    tokenizer = _Tokenizer()

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def detect(self, text: str, family: str, key_id: str):
        self.calls.append((text, family))
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


def test_below_family_minimum_is_inapplicable_without_detector_call(tmp_path, monkeypatch) -> None:
    """A refined candidate too short for a family's detector is explicit
    non-applicability and must never reach MarkLLM (Unbiased would crash on a
    one-token segment)."""
    source = tmp_path / "scores.jsonl"
    write_jsonl(
        source,
        [
            {
                "document_id": "doc-minimum",
                "recipe_id": "doc-minimum",
                "text": "abcdefghij",
                "character_logits": [10.0, 10.0, 10.0, 10.0] + [0.0] * 10,
            }
        ],
    )

    import cwr_eg.registered_scoring as rs
    from cwr_eg.contracts import CharacterInterval

    class _Candidate:
        candidate_id = "candidate-one-token"
        interval = CharacterInterval(0, 1)

    monkeypatch.setattr(rs, "generate_candidates", lambda *args, **kwargs: [_Candidate()])
    monkeypatch.setattr(rs, "refine_candidate", lambda candidate, logits: candidate)

    bridge = _RecordingBridge()
    result = score_registered_evidence(
        bridge=bridge,
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

    # One token is below KGW (2), Unbiased (6), and SynthID (5) minimums, but
    # not below Unigram (1). With two key slots, that is 3 families * 2 = 6
    # inapplicable registered tests for the short candidate.
    short = [item for item in invalid if item["interval_role"] == "candidate"]
    assert {item["scheme_family"] for item in short} == {"kgw", "unbiased", "synthid"}
    assert {item["reason_codes"][-1] for item in short} == {"insufficient_detector_tokens"}
    assert {item["detector_token_count"] for item in short} == {1}

    # The one-token segment must never reach any detector except Unigram.
    one_token_detector_calls = [
        (text, family) for text, family in bridge.calls if text == "a"
    ]
    assert {family for _, family in one_token_detector_calls} == {"unigram"}

    # The full-text interval remains scoreable for every family.
    full_text_evidence = [
        item for item in row["registered_evidence"] if item["interval_role"] == "full_text"
    ]
    assert {item["scheme_family"] for item in full_text_evidence} == {
        "kgw",
        "unigram",
        "unbiased",
        "synthid",
    }
