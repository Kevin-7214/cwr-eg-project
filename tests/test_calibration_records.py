from __future__ import annotations

from cwr_eg.calibration_records import build_parent_calibration_records
from cwr_eg.manifest import read_jsonl, write_jsonl


def test_calibration_records_keep_two_null_descendants_for_parent_maximum(tmp_path) -> None:
    rows = []
    for attacked in (False, True):
        rows.append(
            {
                "document_id": "attacked" if attacked else "clean",
                "parent_ids": ["parent-1"],
                "split": "calibration",
                "language": "en",
                "watermark_family": None,
                "attack_id": "paraphrase" if attacked else None,
                "character_logits": [float(index) / 20 for index in range(20)],
                "registered_evidence": [
                    {
                        "char_start": 0,
                        "char_end": 20,
                        "interval_role": "full_text",
                        "raw_statistic": 0.001,
                        "evidence_strength": 3.0,
                    }
                ],
            }
        )
    source = tmp_path / "scores.jsonl"
    write_jsonl(source, rows)
    result = build_parent_calibration_records(
        scored_documents_path=source,
        output_path=tmp_path / "calibration.jsonl",
        window_lengths=(5, 10),
        stride_fraction=0.5,
        candidate_quantile=0.9,
        merge_gap_chars=0,
    )
    records = read_jsonl(tmp_path / "calibration.jsonl")
    assert result == {
        **result,
        "records": 6,
        "null_descendants": 2,
        "parents": 1,
    }
    assert {row["null_descendant_kind"] for row in records} == {
        "clean",
        "attacked-clean",
    }
    assert all(row["full_search_executed"] for row in records)
