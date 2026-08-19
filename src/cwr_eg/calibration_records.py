from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from cwr_eg.candidates import generate_candidates, refine_candidate
from cwr_eg.manifest import read_jsonl, write_jsonl


def _registered_strength(item: dict[str, Any]) -> float:
    value = item.get("evidence_strength")
    return float(item["raw_statistic"] if value is None else value)


def build_parent_calibration_records(
    *,
    scored_documents_path: str | Path,
    output_path: str | Path,
    window_lengths: Iterable[int],
    stride_fraction: float,
    candidate_quantile: float,
    merge_gap_chars: int,
) -> dict[str, Any]:
    rows = read_jsonl(scored_documents_path)
    output: list[dict[str, Any]] = []
    null_descendants = 0
    parent_ids: set[str] = set()
    for row in rows:
        if str(row.get("split")) != "calibration":
            raise ValueError("Calibration score input must contain only Calibration rows")
        if row.get("watermark_family") is not None:
            continue
        if len(row["parent_ids"]) != 1:
            raise ValueError("Null calibration descendants require one parent")
        parent_id = str(row["parent_ids"][0])
        parent_ids.add(parent_id)
        null_descendants += 1
        candidates = [
            refine_candidate(candidate, row["character_logits"])
            for candidate in generate_candidates(
                row["character_logits"],
                tuple(int(value) for value in window_lengths),
                stride_fraction,
                candidate_quantile,
                merge_gap_chars,
            )
        ]
        evidence = list(row.get("registered_evidence", ()))
        generic_maximum = max((item.raw_score for item in candidates), default=0.0)
        registered_maximum = max(
            (_registered_strength(item) for item in evidence), default=0.0
        )
        gap_values: list[float] = []
        for candidate in candidates:
            applicable = [
                item
                for item in evidence
                if (
                    int(item["char_start"]) == candidate.interval.char_start
                    and int(item["char_end"]) == candidate.interval.char_end
                )
                or str(item.get("interval_role")) == "full_text"
            ]
            best_registered = max(
                (_registered_strength(item) for item in applicable), default=0.0
            )
            gap_values.append(candidate.raw_score - best_registered)
        gap_maximum = max(gap_values, default=-registered_maximum)
        common = {
            "split": "calibration",
            "parent_id": parent_id,
            "document_id": row["document_id"],
            "stratum": f"{row['language']}:all",
            "is_null_descendant": True,
            "null_descendant_kind": "attacked-clean"
            if row.get("attack_id") is not None
            else "clean",
            "full_search_executed": True,
        }
        for route, value in (
            ("generic", generic_maximum),
            ("registered", registered_maximum),
            ("gap", gap_maximum),
        ):
            output.append({**common, "route": route, "maximum_statistic": value})
    if not output:
        raise ValueError("No null Calibration descendants were found")
    target = Path(output_path)
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite calibration records: {target}")
    write_jsonl(target, output)
    from cwr_eg.hashing import sha256_file

    return {
        "records": len(output),
        "null_descendants": null_descendants,
        "parents": len(parent_ids),
        "output_path": str(target),
        "output_sha256": sha256_file(target),
        "routes": ["generic", "registered", "gap"],
        "aggregation_pending": "parent_id_maximum",
    }
