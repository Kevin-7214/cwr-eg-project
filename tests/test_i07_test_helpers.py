from __future__ import annotations

from cwr_eg.hashing import sha256_text
from cwr_eg.manifest import read_jsonl, write_jsonl
from scripts.i07_test_helpers import mask_registered_records


def test_key_mask_keeps_scheme_only_and_only_authorized_slot(tmp_path) -> None:
    source = tmp_path / "source.jsonl"
    write_jsonl(
        source,
        [
            {
                "recipe_id": "doc-1",
                "registered_evidence": [
                    {
                        "scheme_family": "kgw",
                        "key_status": "registered",
                        "key_id_hash": "sha256:" + sha256_text("kgw_key_a"),
                    },
                    {
                        "scheme_family": "kgw",
                        "key_status": "registered",
                        "key_id_hash": "sha256:" + sha256_text("kgw_key_b"),
                    },
                    {
                        "scheme_family": "kgw",
                        "key_status": "scheme_only",
                        "key_id_hash": None,
                    },
                ],
                "registered_search_space": {
                    "authorized_key_slots": ["a", "b"],
                    "families": ["kgw"],
                    "tests": 3,
                },
            }
        ],
    )
    target = tmp_path / "a-only.jsonl"
    mask_registered_records(source, target, authorized_key_slots=("a",))
    evidence = read_jsonl(target)[0]["registered_evidence"]
    assert len(evidence) == 2
    assert {item["key_status"] for item in evidence} == {"registered", "scheme_only"}
