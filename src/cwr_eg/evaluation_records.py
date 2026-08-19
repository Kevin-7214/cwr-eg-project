from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from cwr_eg.hashing import sha256_file
from cwr_eg.manifest import read_jsonl, write_jsonl


ACCEPTED_WATERMARK_LABELS = {
    "known_scheme_known_key",
    "known_scheme_unknown_key",
    "suspected_unknown_scheme",
}


def _families_and_keys(row: dict[str, Any]) -> tuple[list[str], list[str]]:
    family = row.get("watermark_family")
    if family is None:
        return [], []
    if family == "mixed":
        return (
            [str(item) for item in row.get("watermark_families", ())],
            [str(item) for item in row.get("key_ids", ())],
        )
    return [str(family)], [str(row.get("key_id", ""))]


def true_document_label(
    row: dict[str, Any],
    *,
    authorized_key_slots: Iterable[str],
    held_out_family: str | None,
) -> str:
    families, key_ids = _families_and_keys(row)
    if not families:
        return "none"
    authorized = set(str(item) for item in authorized_key_slots)
    known_authorized = False
    known_unauthorized = False
    unknown = False
    for family, key_id in zip(families, key_ids, strict=True):
        if family == held_out_family:
            unknown = True
        elif key_id.rsplit("_", 1)[-1] in authorized:
            known_authorized = True
        else:
            known_unauthorized = True
    if known_authorized:
        return "known_scheme_known_key"
    if known_unauthorized:
        return "known_scheme_unknown_key"
    if unknown:
        return "suspected_unknown_scheme"
    raise RuntimeError("Unable to assign a frozen Test truth label")


def build_evaluation_records(
    *,
    decisions_path: str | Path,
    documents_path: str | Path,
    output_path: str | Path,
    authorized_key_slots: Iterable[str],
    held_out_family: str | None = None,
) -> dict[str, Any]:
    decisions = {str(row["document_id"]): row for row in read_jsonl(decisions_path)}
    documents = {
        str(row["recipe_id"]): row
        for row in read_jsonl(documents_path)
        if str(row.get("split")) == "test"
    }
    if set(decisions) != set(documents):
        missing_decisions = sorted(set(documents) - set(decisions))
        extra_decisions = sorted(set(decisions) - set(documents))
        raise RuntimeError(
            f"Test decision/document mismatch: missing={missing_decisions[:3]}, extra={extra_decisions[:3]}"
        )
    records: list[dict[str, Any]] = []
    source_parent_ids: set[str] = set()
    for document_id in sorted(documents):
        document = documents[document_id]
        source_parent_ids.update(str(item) for item in document["parent_ids"])
        decision = decisions[document_id]
        true_intervals = [list(item) for item in document.get("watermark_intervals", ())]
        predicted_intervals = [
            [int(segment["interval"]["char_start"]), int(segment["interval"]["char_end"])]
            if "interval" in segment
            else [int(segment["char_start"]), int(segment["char_end"])]
            for segment in decision.get("segments", ())
            if str(segment["label"]) in ACCEPTED_WATERMARK_LABELS
        ]
        registered_p = decision.get("document_registered_p")
        generic_p = decision.get("document_generic_p")
        records.append(
            {
                "parent_id": "|".join(sorted(str(item) for item in document["parent_ids"])),
                "document_id": document_id,
                "true_label": true_document_label(
                    document,
                    authorized_key_slots=authorized_key_slots,
                    held_out_family=held_out_family,
                ),
                "predicted_label": str(decision["document_label"]),
                "score": 0.0 if generic_p is None else 1.0 - float(generic_p),
                "knownness_score": 0.0
                if registered_p is None
                else 1.0 - float(registered_p),
                "true_intervals": true_intervals,
                "predicted_intervals": predicted_intervals,
                "source": document.get("source"),
                "language": document.get("language"),
                "watermark_family": document.get("watermark_family"),
                "key_id": document.get("key_id"),
                "attack_id": document.get("attack_id"),
                "authorization_scenario": "_and_".join(authorized_key_slots),
                "held_out_family": held_out_family,
            }
        )
    target = Path(output_path)
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation records: {target}")
    write_jsonl(target, records)
    return {
        "records": len(records),
        "parents": len(source_parent_ids),
        "output_path": str(target),
        "output_sha256": sha256_file(target),
        "authorized_key_slots": list(authorized_key_slots),
        "held_out_family": held_out_family,
    }
