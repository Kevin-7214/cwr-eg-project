from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

from cwr_eg.hashing import sha256_file


COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object in {path}")
    return payload


def audit_registry_files(project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root)
    models = _load_json(root / "manifests" / "model_registry.json")
    repositories = _load_json(root / "manifests" / "repository_registry.json")
    corpora = _load_json(root / "manifests" / "corpus_registry.json")
    errors: list[str] = []
    for model in models.get("models", []):
        if not COMMIT_PATTERN.fullmatch(str(model.get("revision", ""))):
            errors.append(f"invalid_model_revision:{model.get('id')}")
        if not model.get("license"):
            errors.append(f"missing_model_license:{model.get('id')}")
    for repository in repositories.get("repositories", []):
        if not COMMIT_PATTERN.fullmatch(str(repository.get("commit", ""))):
            errors.append(f"invalid_repository_commit:{repository.get('name')}")
        if not repository.get("license") and "disabled" not in str(repository.get("policy")):
            errors.append(f"unlicensed_repository_enabled:{repository.get('name')}")
    for source in corpora.get("sources", []):
        if source.get("selected_for_pilot") and not source.get("license"):
            errors.append(f"unlicensed_corpus_selected:{source.get('name')}")
    return {"ok": not errors, "errors": errors}


def audit_legacy_assets(project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    legacy = root.parents[2] / "project1"
    corpus_manifest = _load_json(legacy / "manifests" / "corpus_manifest.json")
    model_manifest = _load_json(legacy / "manifests" / "model_manifest.json")
    records: list[dict[str, Any]] = []
    for source in corpus_manifest["sources"]:
        path = legacy / "data" / "corpus" / f"{source['name']}.jsonl"
        actual = sha256_file(path)
        records.append(
            {
                "kind": "corpus",
                "id": source["name"],
                "path": str(path),
                "expected_sha256": source["sha256"],
                "actual_sha256": actual,
                "ok": actual == source["sha256"] and path.stat().st_size == source["bytes"],
            }
        )
    for model in model_manifest["models"]:
        path = legacy / model["file"]
        actual = sha256_file(path)
        records.append(
            {
                "kind": "model_file",
                "id": model["id"],
                "path": str(path),
                "expected_sha256": model["sha256"],
                "actual_sha256": actual,
                "ok": actual == model["sha256"] and path.stat().st_size == model["bytes"],
            }
        )
    registry = audit_registry_files(root)
    return {
        "ok": all(record["ok"] for record in records) and registry["ok"],
        "legacy_root": str(legacy),
        "records": records,
        "registry": registry,
        "experiment_executed": False,
    }
