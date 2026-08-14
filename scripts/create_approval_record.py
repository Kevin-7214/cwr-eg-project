from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path

from cwr_eg.approval import approval_fingerprint
from cwr_eg.config import config_hash


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--approval-id", required=True)
    parser.add_argument("--action", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resource-class", required=True)
    parser.add_argument("--scope-file", type=Path, required=True)
    parser.add_argument("--issued-at", required=True)
    parser.add_argument("--expires-at", required=True)
    parser.add_argument("--chat-evidence", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    issued_at = datetime.fromisoformat(args.issued_at)
    expires_at = datetime.fromisoformat(args.expires_at)
    if issued_at.tzinfo is None or expires_at.tzinfo is None or expires_at <= issued_at:
        raise ValueError("Approval times must be ordered and timezone-aware")
    scope = json.loads(args.scope_file.read_text(encoding="utf-8"))
    fingerprint = approval_fingerprint(
        action=args.action,
        config_hash=config_hash(args.config),
        resource_class=args.resource_class,
        scope=scope,
    )
    payload = {
        "approval_id": args.approval_id,
        "approved": True,
        "approved_by": "user_chat",
        "issued_at": issued_at.isoformat(),
        "expires_at": expires_at.isoformat(),
        "action": args.action,
        "fingerprint": fingerprint,
        "resource_class": args.resource_class,
        "scope": scope,
        "chat_evidence": args.chat_evidence,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"approval_id": args.approval_id, "fingerprint": fingerprint}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
