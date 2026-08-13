from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


EXCLUDED_PARTS = {
    ".git",
    ".pytest_cache",
    "__pycache__",
    "external",
    "artifacts",
    "approvals",
}
EXCLUDED_NAMES = {
    "pre_experiment_checksums.sha256",
    "preflight_report.json",
    "progress.jsonl",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--new-tests", type=int, required=True)
    parser.add_argument("--legacy-tests", type=int, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and not (set(path.relative_to(root).parts) & EXCLUDED_PARTS)
        and path.name not in EXCLUDED_NAMES
    )
    records = [(sha256_file(path), path.relative_to(root).as_posix()) for path in files]
    checksum_path = root / "manifests" / "pre_experiment_checksums.sha256"
    checksum_path.write_text(
        "".join(f"{digest}  {relative}\n" for digest, relative in records),
        encoding="utf-8",
        newline="\n",
    )
    report = {
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "status": "ready_at_local_experiment_approval_gate",
        "experiment_executed": False,
        "checksummed_files": len(records),
        "checksums_sha256": sha256_file(checksum_path),
        "new_project_tests_passed": args.new_tests,
        "legacy_project_tests_passed": args.legacy_tests,
        "pilot_parent_samples": 32,
        "pilot_base_generation_recipes": 160,
        "pilot_matched_attack_recipes": 160,
        "pilot_mixed_document_recipes": 16,
        "approval_gate": "waiting_user_approval",
        "known_environment_warning": "current_system_torch_2_0_numpy_2_abi_mismatch",
    }
    (root / "manifests" / "preflight_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
