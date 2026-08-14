from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from cwr_eg.candidates import generate_candidates, refine_candidate
from cwr_eg.config import load_yaml
from cwr_eg.manifest import read_jsonl, write_jsonl


def deterministic_character_scores(text: str) -> list[float]:
    """Create non-scientific deterministic scores for pipeline wiring checks."""
    return [
        float(((ord(character) * 1103515245 + index * 12345) & 0xFFFF) / 65535.0)
        for index, character in enumerate(text)
    ]


def _search_maximum(text: str, config: dict[str, Any]) -> float:
    scores = np.asarray(deterministic_character_scores(text), dtype=np.float64)
    candidates = (
        refine_candidate(candidate, scores)
        for candidate in generate_candidates(
            scores,
            config["search"]["window_char_lengths"],
            float(config["search"]["stride_fraction"]),
            float(config["search"]["candidate_quantile"]),
            int(config["search"]["merge_gap_chars"]),
        )
    )
    return max((candidate.raw_score for candidate in candidates), default=0.0)


def build_calibration_records(
    rows: list[dict[str, Any]], config: dict[str, Any]
) -> list[dict[str, Any]]:
    selected = [
        row
        for row in rows
        if row["split"] == "calibration" and row.get("watermark_family") is None
    ]
    records: list[dict[str, Any]] = []
    for row in sorted(selected, key=lambda item: str(item["recipe_id"])):
        generic_maximum = _search_maximum(str(row["text"]), config)
        for route, maximum in (
            ("generic", generic_maximum),
            ("registered", 0.0),
            ("gap", generic_maximum),
        ):
            records.append(
                {
                    "document_id": row["recipe_id"],
                    "parent_id": row["parent_ids"][0],
                    "split": "calibration",
                    "route": route,
                    "stratum": f"{row['language']}:all",
                    "maximum_statistic": maximum,
                    "full_search_executed": True,
                    "score_source": "deterministic_pipeline_smoke_v1",
                }
            )
    if len(selected) != 4 or len(records) != 12:
        raise ValueError("Expected four Calibration clean documents and twelve route records")
    return records


def build_test_record(
    rows: list[dict[str, Any]], recipe_id: str
) -> list[dict[str, Any]]:
    selected = [row for row in rows if row["recipe_id"] == recipe_id]
    if len(selected) != 1:
        raise ValueError("The Test smoke recipe id must select exactly one row")
    row = selected[0]
    if row["split"] != "test" or row.get("watermark_family") is not None:
        raise ValueError("The inference smoke record must be a clean Test row")
    text = str(row["text"])
    return [
        {
            "document_id": row["recipe_id"],
            "parent_id": row["parent_ids"][0],
            "split": "test",
            "text": text,
            "language": row["language"],
            "character_scores": deterministic_character_scores(text),
            "effective_length": sum(not character.isspace() for character in text),
            "mapping_coverage": 1.0,
            "registered_evidence": [],
            "score_source": "deterministic_pipeline_smoke_v1",
        }
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("calibration", "test"))
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--recipe-id")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite smoke records: {args.output}")
    rows = read_jsonl(args.input)
    config = load_yaml(args.config)
    if args.mode == "calibration":
        if args.recipe_id is not None:
            raise ValueError("Calibration mode does not accept --recipe-id")
        records = build_calibration_records(rows, config)
    else:
        if not args.recipe_id:
            raise ValueError("Test mode requires --recipe-id")
        records = build_test_record(rows, args.recipe_id)
    write_jsonl(args.output, records)
    print(f"wrote={len(records)} path={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
