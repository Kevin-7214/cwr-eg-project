from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from cwr_eg.approval import EXPERIMENT_ACTIONS, approval_fingerprint, require_approval
from cwr_eg.assets import audit_legacy_assets
from cwr_eg.config import config_hash, load_yaml, validate_experiment_config
from cwr_eg.data_prep import (
    prepare_intermediate_canary,
    prepare_intermediate_data,
    prepare_pilot_data,
)
from cwr_eg.progress import latest_progress


def _json_object(value: str) -> dict[str, object]:
    payload = json.loads(value)
    if not isinstance(payload, dict):
        raise argparse.ArgumentTypeError("scope must be a JSON object")
    return payload


def _scope_from_args(args: argparse.Namespace) -> dict[str, object]:
    if args.scope is not None:
        return args.scope
    with args.scope_file.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Scope file must contain a JSON object")
    return payload


def _add_scope_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--scope", type=_json_object)
    group.add_argument("--scope-file", type=Path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cwr-eg")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-config")
    validate.add_argument("--config", type=Path, default=Path("configs/pilot.yaml"))

    prepare = subparsers.add_parser("prepare-data")
    prepare.add_argument("--legacy-corpus-dir", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, default=Path("manifests"))
    prepare.add_argument("--seed", type=int, default=20260813)

    intermediate = subparsers.add_parser("prepare-intermediate-data")
    intermediate.add_argument("--legacy-corpus-dir", type=Path, required=True)
    intermediate.add_argument("--output-dir", type=Path, default=Path("manifests"))
    intermediate.add_argument(
        "--excluded-parent-manifest",
        type=Path,
        default=Path("manifests/pilot_parents.jsonl"),
    )
    intermediate.add_argument("--seed", type=int, default=20260815)

    canary = subparsers.add_parser("prepare-intermediate-canary")
    canary.add_argument(
        "--parent-manifest", type=Path, default=Path("manifests/intermediate_parents.jsonl")
    )
    canary.add_argument(
        "--recipe-manifest", type=Path, default=Path("manifests/intermediate_recipes.jsonl")
    )
    canary.add_argument("--output-dir", type=Path, default=Path("manifests"))
    canary.add_argument("--seed", type=int, default=20260815)

    audit = subparsers.add_parser("audit-assets")
    audit.add_argument("--project-root", type=Path, default=Path.cwd())
    audit.add_argument("--output", type=Path)

    status = subparsers.add_parser("status")
    status.add_argument("--progress", type=Path, default=Path("status/progress.jsonl"))

    fingerprint = subparsers.add_parser("fingerprint")
    fingerprint.add_argument("action", choices=sorted(EXPERIMENT_ACTIONS))
    fingerprint.add_argument("--config", type=Path, default=Path("configs/pilot.yaml"))
    fingerprint.add_argument("--resource-class", required=True)
    _add_scope_arguments(fingerprint)

    for action in sorted(EXPERIMENT_ACTIONS):
        action_parser = subparsers.add_parser(action)
        action_parser.add_argument("--config", type=Path, default=Path("configs/pilot.yaml"))
        action_parser.add_argument("--resource-class", required=True)
        _add_scope_arguments(action_parser)
        action_parser.add_argument("--approval", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "validate-config":
        payload = load_yaml(args.config)
        validate_experiment_config(payload)
        print(json.dumps({"ok": True, "config_hash": config_hash(args.config)}))
        return 0
    if args.command == "prepare-data":
        result = prepare_pilot_data(
            legacy_corpus_dir=args.legacy_corpus_dir,
            output_dir=args.output_dir,
            seed=args.seed,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "prepare-intermediate-data":
        result = prepare_intermediate_data(
            legacy_corpus_dir=args.legacy_corpus_dir,
            output_dir=args.output_dir,
            excluded_parent_manifest=args.excluded_parent_manifest,
            seed=args.seed,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "prepare-intermediate-canary":
        result = prepare_intermediate_canary(
            parent_manifest=args.parent_manifest,
            recipe_manifest=args.recipe_manifest,
            output_dir=args.output_dir,
            seed=args.seed,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "audit-assets":
        result = audit_legacy_assets(args.project_root)
        output = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(output, encoding="utf-8")
        print(output, end="")
        return 0 if result["ok"] else 1
    if args.command == "status":
        print(json.dumps(latest_progress(args.progress), ensure_ascii=False, indent=2))
        return 0

    scope = _scope_from_args(args)
    fingerprint = approval_fingerprint(
        action=args.action if args.command == "fingerprint" else args.command,
        config_hash=config_hash(args.config),
        resource_class=args.resource_class,
        scope=scope,
    )
    if args.command == "fingerprint":
        print(json.dumps({"fingerprint": fingerprint}, indent=2))
        return 0
    require_approval(
        args.approval,
        action=args.command,
        expected_fingerprint=fingerprint,
    )
    from cwr_eg.runtime import execute_approved_action

    return execute_approved_action(args.command, args.config, scope)


if __name__ == "__main__":
    raise SystemExit(main())
