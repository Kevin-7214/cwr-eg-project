from __future__ import annotations

import json

from cwr_eg.cli import _scope_from_args, build_parser


def test_scope_file_avoids_powershell_json_quoting(tmp_path) -> None:
    path = tmp_path / "scope.json"
    path.write_text(json.dumps({"device_index": 0}), encoding="utf-8")
    args = build_parser().parse_args(
        [
            "fingerprint",
            "cuda-smoke",
            "--resource-class",
            "local-rtx5060",
            "--scope-file",
            str(path),
        ]
    )
    assert _scope_from_args(args) == {"device_index": 0}
