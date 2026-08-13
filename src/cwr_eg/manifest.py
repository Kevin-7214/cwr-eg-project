from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable
import json
from pathlib import Path
from typing import Any

from cwr_eg.hashing import content_hash, sha256_file


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"JSONL row {line_number} is not an object")
            rows.append(payload)
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(target)


def audit_parent_splits(
    rows: Iterable[dict[str, Any]], expected_counts: dict[str, int] | None = None
) -> dict[str, Any]:
    parent_splits: dict[str, set[str]] = defaultdict(set)
    counts: Counter[str] = Counter()
    hashes: set[str] = set()
    duplicates: list[str] = []
    for row in rows:
        parent_id = str(row["parent_id"])
        split = str(row["split"])
        parent_splits[parent_id].add(split)
        counts[split] += 1
        text_hash = str(row["text_sha256"])
        if text_hash in hashes:
            duplicates.append(text_hash)
        hashes.add(text_hash)
    leaked = sorted(parent_id for parent_id, splits in parent_splits.items() if len(splits) > 1)
    if leaked:
        raise ValueError(f"parent_id split leakage: {leaked[:5]}")
    if duplicates:
        raise ValueError(f"Duplicate pilot text hashes: {duplicates[:5]}")
    if expected_counts and dict(counts) != expected_counts:
        raise ValueError(f"Split counts {dict(counts)} != {expected_counts}")
    return {
        "rows": sum(counts.values()),
        "split_counts": dict(sorted(counts.items())),
        "unique_parent_ids": len(parent_splits),
        "manifest_content_hash": content_hash(list(sorted(hashes))),
    }


def file_record(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    return {
        "path": target.as_posix(),
        "bytes": target.stat().st_size,
        "sha256": sha256_file(target),
    }
