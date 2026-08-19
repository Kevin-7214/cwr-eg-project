from __future__ import annotations

from cwr_eg.evaluation_records import true_document_label


def test_authorization_and_lofo_truth_labels_are_distinct() -> None:
    kgw_b = {"watermark_family": "kgw", "key_id": "kgw_key_b"}
    assert (
        true_document_label(
            kgw_b, authorized_key_slots=("a", "b"), held_out_family=None
        )
        == "known_scheme_known_key"
    )
    assert (
        true_document_label(kgw_b, authorized_key_slots=("a",), held_out_family=None)
        == "known_scheme_unknown_key"
    )
    assert (
        true_document_label(kgw_b, authorized_key_slots=("a",), held_out_family="kgw")
        == "suspected_unknown_scheme"
    )
    assert (
        true_document_label(
            {"watermark_family": None},
            authorized_key_slots=("a", "b"),
            held_out_family=None,
        )
        == "none"
    )
