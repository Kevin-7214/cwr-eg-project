from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


VALID_STATUSES = {
    "pending",
    "in_progress",
    "done",
    "blocked",
    "waiting_user_approval",
}


def append_progress(
    path: str | Path,
    *,
    task_id: str,
    status: str,
    evidence: str,
    details: dict[str, Any] | None = None,
) -> None:
    if status not in VALID_STATUSES:
        raise ValueError(f"Invalid status: {status}")
    payload = {
        "time": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "task_id": task_id,
        "status": status,
        "evidence": evidence,
    }
    if details:
        payload["details"] = details
    with Path(path).open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def latest_progress(path: str | Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                item = json.loads(line)
                latest[str(item["task_id"])] = item
    return latest
