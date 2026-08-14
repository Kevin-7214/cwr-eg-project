from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from cwr_eg.hashing import content_hash


EXPERIMENT_ACTIONS = frozenset(
    {
        "cuda-smoke",
        "model-smoke",
        "generate",
        "attack-generate",
        "extract-features",
        "tensorize",
        "train",
        "calibrate",
        "infer",
        "evaluate",
        "benchmark",
    }
)


def approval_fingerprint(
    *, action: str, config_hash: str, resource_class: str, scope: dict[str, Any]
) -> str:
    if action not in EXPERIMENT_ACTIONS:
        raise ValueError(f"Unknown experiment action: {action}")
    return content_hash(
        {
            "action": action,
            "config_hash": config_hash,
            "resource_class": resource_class,
            "scope": scope,
        }
    )


@dataclass(frozen=True, slots=True)
class ApprovalRecord:
    approval_id: str
    approved: bool
    approved_by: str
    issued_at: datetime
    expires_at: datetime
    action: str
    fingerprint: str
    resource_class: str
    scope: dict[str, Any]
    chat_evidence: str

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ApprovalRecord":
        issued = datetime.fromisoformat(str(payload["issued_at"]))
        expires = datetime.fromisoformat(str(payload["expires_at"]))
        if issued.tzinfo is None or expires.tzinfo is None:
            raise ValueError("Approval times must include UTC offsets")
        return cls(
            approval_id=str(payload["approval_id"]),
            approved=bool(payload["approved"]),
            approved_by=str(payload["approved_by"]),
            issued_at=issued,
            expires_at=expires,
            action=str(payload["action"]),
            fingerprint=str(payload["fingerprint"]),
            resource_class=str(payload["resource_class"]),
            scope=dict(payload["scope"]),
            chat_evidence=str(payload["chat_evidence"]),
        )


def require_approval(
    approval_path: str | Path,
    *,
    action: str,
    expected_fingerprint: str,
    now: datetime | None = None,
) -> ApprovalRecord:
    with Path(approval_path).open("r", encoding="utf-8") as handle:
        record = ApprovalRecord.from_dict(json.load(handle))
    current = now or datetime.now(timezone.utc)
    if not record.approved or record.approved_by != "user_chat":
        raise PermissionError("Experiment approval is absent or not user-issued")
    if record.action != action or record.fingerprint != expected_fingerprint:
        raise PermissionError("Approval does not match the requested experiment scope")
    if current < record.issued_at or current > record.expires_at:
        raise PermissionError("Experiment approval is not currently valid")
    return record
