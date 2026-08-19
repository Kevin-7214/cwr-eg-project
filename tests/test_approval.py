from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json

import pytest

from cwr_eg.approval import EXPERIMENT_ACTIONS, approval_fingerprint, require_approval


def test_exact_user_approval_scope_is_required(tmp_path) -> None:
    now = datetime.now(timezone.utc)
    scope = {"model": "qwen-0.5b", "prompts": 1}
    fingerprint = approval_fingerprint(
        action="model-smoke",
        config_hash="config",
        resource_class="local-rtx5060",
        scope=scope,
    )
    path = tmp_path / "approval.json"
    path.write_text(
        json.dumps(
            {
                "approval_id": "approval-1",
                "approved": True,
                "approved_by": "user_chat",
                "issued_at": (now - timedelta(minutes=1)).isoformat(),
                "expires_at": (now + timedelta(minutes=5)).isoformat(),
                "action": "model-smoke",
                "fingerprint": fingerprint,
                "resource_class": "local-rtx5060",
                "scope": scope,
                "chat_evidence": "explicit approval",
            }
        ),
        encoding="utf-8",
    )
    assert require_approval(
        path, action="model-smoke", expected_fingerprint=fingerprint, now=now
    ).approval_id == "approval-1"
    with pytest.raises(PermissionError):
        require_approval(path, action="train", expected_fingerprint=fingerprint, now=now)
    with pytest.raises(PermissionError):
        require_approval(path, action="model-smoke", expected_fingerprint="other", now=now)


def test_checkpoint_scoring_is_a_separately_gated_action(tmp_path) -> None:
    assert "score-checkpoint" in EXPERIMENT_ACTIONS
    fingerprint = approval_fingerprint(
        action="score-checkpoint",
        config_hash="config",
        resource_class="local-rtx5060",
        scope={"checkpoint": "frozen", "documents": 80},
    )
    with pytest.raises(FileNotFoundError):
        require_approval(
            tmp_path / "missing.json",
            action="score-checkpoint",
            expected_fingerprint=fingerprint,
        )
