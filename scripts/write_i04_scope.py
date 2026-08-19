from __future__ import annotations

import argparse
import json
from pathlib import Path

from cwr_eg.hashing import sha256_file


RUNTIME_PATH = Path("src/cwr_eg/runtime.py")
FREEZE_PATH = Path("manifests/intermediate_freeze_amendment_03.json")
PARENTS_PATH = Path("manifests/intermediate_parents.jsonl")
RECIPES_PATH = Path("manifests/intermediate_recipes.jsonl")


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _replace_code_hash(scope: dict, path: Path) -> None:
    for entry in scope["code_files"]:
        if entry["path"] == str(path).replace("\\", "/"):
            entry["sha256"] = sha256_file(path)
            return


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("base", "mixed", "attack"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-input-path", default="artifacts/i04_full/base_generated.jsonl")
    parser.add_argument("--base-input-sha256")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite scope: {args.output}")
    template_names = {
        "base": "docs/i03_canary_base_generation_scope.json",
        "mixed": "docs/i03_canary_mixed_generation_scope.json",
        "attack": "docs/i03_canary_attack_generation_scope.json",
    }
    scope = _load(Path(template_names[args.kind]))
    scope["task_id"] = f"I-04-full-{args.kind}"
    scope["freeze_manifest"] = str(FREEZE_PATH).replace("\\", "/")
    scope["freeze_manifest_sha256"] = sha256_file(FREEZE_PATH)
    scope["parent_freeze_manifest_sha256"] = sha256_file(
        "manifests/intermediate_freeze_manifest.json"
    )
    scope["runner_sha256"] = sha256_file(RUNTIME_PATH)
    _replace_code_hash(scope, RUNTIME_PATH)
    scope["recipe_manifest"] = str(RECIPES_PATH).replace("\\", "/")
    scope["recipe_manifest_sha256"] = sha256_file(RECIPES_PATH)
    if args.kind in {"base", "mixed"}:
        scope["parent_manifest"] = str(PARENTS_PATH).replace("\\", "/")
        scope["parent_manifest_sha256"] = sha256_file(PARENTS_PATH)
    settings = {
        "base": (4000, 400, "96568c596c078aa9280deb665e578680aa99f281625a36c9e66435bb222f10d3"),
        "mixed": (400, 40, "772c33cdadd7d2ecc4edb528e0e4e10d5813d4f7b407df0784c11078047afd2c"),
        "attack": (4000, 400, "c6f4ad8a06da5c061a3a2601f8074d5077e07971df62f46d5d9c9236f494803a"),
    }
    expected, resumed, partial_sha256 = settings[args.kind]
    scope["expected_recipe_count"] = expected
    scope["limit"] = expected
    scope["expected_resumed_documents"] = resumed
    scope["resume_partial_sha256"] = partial_sha256
    scope["reuse_source_task"] = "I-03"
    scope["reuse_policy"] = "hash-and-frozen-recipe-bound-v1"
    if args.kind == "attack":
        if not args.base_input_sha256:
            raise ValueError("Attack scope requires the completed full-base SHA-256")
        scope["input_path"] = args.base_input_path
        scope["input_sha256"] = args.base_input_sha256
        scope["output_path"] = "artifacts/i04_full/attacked_generated.jsonl"
        scope["result_path"] = "artifacts/i04_full/attack_generation_result.json"
    else:
        scope["output_path"] = f"artifacts/i04_full/{args.kind}_generated.jsonl"
        scope["result_path"] = f"artifacts/i04_full/{args.kind}_generation_result.json"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(scope, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"path": str(args.output), "sha256": sha256_file(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
