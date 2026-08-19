from __future__ import annotations

from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Callable

from cwr_eg.config import load_yaml
from cwr_eg.hashing import content_hash, sha256_file, sha256_text


def _write_result(scope: dict[str, Any], result: dict[str, Any]) -> None:
    output = scope.get("result_path")
    if output:
        target = Path(str(output))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


def _verify_runner(scope: dict[str, Any]) -> str:
    expected = str(scope["runner_sha256"])
    actual = sha256_file(Path(__file__))
    if actual != expected:
        raise RuntimeError("Runtime SHA-256 does not match the approved scope")
    return actual


def _verify_freeze_manifest(scope: dict[str, Any]) -> str | None:
    path = scope.get("freeze_manifest")
    expected = scope.get("freeze_manifest_sha256")
    if path is None and expected is None:
        return None
    if path is None or expected is None:
        raise ValueError("Freeze manifest path and SHA-256 must be supplied together")
    actual = sha256_file(path)
    if actual != str(expected):
        raise RuntimeError("Intermediate freeze manifest SHA-256 mismatch")
    return actual


def _verify_code_files(scope: dict[str, Any]) -> dict[str, str]:
    entries = scope.get("code_files")
    if not isinstance(entries, list) or not entries:
        raise ValueError("A non-empty code_files manifest is required")
    project_root = Path.cwd().resolve()
    verified: dict[str, str] = {}
    for entry in entries:
        relative_path = Path(str(entry["path"]))
        target = (project_root / relative_path).resolve()
        if relative_path.is_absolute() or not target.is_relative_to(project_root):
            raise ValueError("code_files paths must remain inside the project root")
        actual_sha256 = sha256_file(target)
        if actual_sha256 != str(entry["sha256"]):
            raise RuntimeError(f"Code SHA-256 mismatch: {relative_path}")
        verified[relative_path.as_posix()] = actual_sha256
    return verified


def _verify_model_files(
    model_path: Path, approved_files: Any
) -> dict[str, dict[str, Any]]:
    resolved_model_path = model_path.resolve()
    if not resolved_model_path.is_dir():
        raise FileNotFoundError(f"Local model directory does not exist: {resolved_model_path}")
    if not isinstance(approved_files, list) or not approved_files:
        raise ValueError("A non-empty model_files manifest is required")
    verified: dict[str, dict[str, Any]] = {}
    for entry in approved_files:
        if not isinstance(entry, dict):
            raise ValueError("Every model_files entry must be an object")
        relative_path = Path(str(entry["path"]))
        target = (resolved_model_path / relative_path).resolve()
        if relative_path.is_absolute() or not target.is_relative_to(resolved_model_path):
            raise ValueError("model_files paths must remain inside model_path")
        expected_bytes = int(entry["bytes"])
        if target.stat().st_size != expected_bytes:
            raise RuntimeError(f"Byte size mismatch for approved model file: {relative_path}")
        expected_sha256 = str(entry["sha256"])
        actual_sha256 = sha256_file(target)
        if actual_sha256 != expected_sha256:
            raise RuntimeError(f"SHA-256 mismatch for approved model file: {relative_path}")
        verified[relative_path.as_posix()] = {
            "bytes": expected_bytes,
            "sha256": actual_sha256,
        }
    if "model.safetensors" not in verified:
        raise ValueError("model_files must include model.safetensors")
    return verified


def _load_private_keys(scope: dict[str, Any]) -> str:
    path = Path(str(scope["key_file"])).resolve()
    actual_sha256 = sha256_file(path)
    if actual_sha256 != str(scope["key_file_sha256"]):
        raise RuntimeError("Private key file SHA-256 does not match the approved scope")
    loaded: set[str] = set()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        name, separator, value = line.partition("=")
        if separator != "=" or not name.startswith("CWR_EG_KEY_") or not value:
            raise ValueError("Private key file contains an invalid line")
        os.environ[name] = value
        loaded.add(name)
    required = {"CWR_EG_KEY_" + str(item).upper() for item in scope["required_key_ids"]}
    if not required.issubset(loaded):
        raise RuntimeError("Private key file does not contain every approved key id")
    return actual_sha256


def _verify_repository(repository: Path, expected_commit: str) -> None:
    completed = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    if completed.stdout.strip() != expected_commit:
        raise RuntimeError("External repository revision does not match the approved scope")
    status = subprocess.run(
        ["git", "-C", str(repository), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    )
    if status.stdout.strip():
        raise RuntimeError("External repository has unapproved local changes")


def _record_recipe_failure(
    *,
    outputs: list[dict[str, Any]],
    partial_path: Path,
    recipe: dict[str, Any],
    error: Exception,
    total: int,
    maximum_failure_rate: float,
) -> None:
    from cwr_eg.manifest import write_jsonl

    outputs.append(
        {
            **recipe,
            "status": "failed",
            "failure_type": type(error).__name__,
            "failure_message": str(error)[:500],
        }
    )
    write_jsonl(partial_path, outputs)
    failures = sum(row.get("status") == "failed" for row in outputs)
    message = str(error).lower()
    unrecoverable = (
        "outofmemory" in type(error).__name__.lower()
        or "out of memory" in message
        or "cuda" in message
        or "driver" in message
    )
    if unrecoverable or failures / total > maximum_failure_rate:
        raise RuntimeError(
            f"Generation stopped after {failures}/{total} explicit failures"
        ) from error


def execute_approved_action(
    action: str, config_path: str | Path, scope: dict[str, Any]
) -> int:
    handlers: dict[str, Callable[[Path, dict[str, Any]], dict[str, Any]]] = {
        "cuda-smoke": _cuda_smoke,
        "model-smoke": _model_smoke,
        "generate": _generate,
        "attack-generate": _attack_generate,
        "assemble-data": _assemble_data,
        "extract-features": _extract_features,
        "tensorize": _tensorize,
        "train": _train,
        "score-checkpoint": _score_checkpoint,
        "score-registered": _score_registered,
        "prepare-calibration": _prepare_calibration,
        "prepare-evaluation": _prepare_evaluation,
        "calibrate": _calibrate,
        "infer": _infer,
        "evaluate": _evaluate,
        "benchmark": _benchmark,
    }
    freeze_manifest_sha256 = _verify_freeze_manifest(scope)
    result = handlers[action](Path(config_path), scope)
    if freeze_manifest_sha256 is not None:
        result["freeze_manifest_sha256"] = freeze_manifest_sha256
    result.update({"action": action, "approved_execution": True})
    _write_result(scope, result)
    return 0


def _cuda_smoke(config_path: Path, scope: dict[str, Any]) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    device_index = int(scope.get("device_index", 0))
    device = torch.device(f"cuda:{device_index}")
    capability = torch.cuda.get_device_capability(device)
    expected = tuple(scope.get("expected_capability", (12, 0)))
    if capability < expected:
        raise RuntimeError(f"CUDA capability {capability} is below expected {expected}")
    first = torch.arange(16, dtype=torch.float32, device=device).reshape(4, 4)
    result = first @ first.T
    torch.cuda.synchronize(device)
    return {
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(device),
        "capability": list(capability),
        "finite": bool(torch.isfinite(result).all().item()),
    }


def _run_model_smoke_operation(
    operation: str,
    *,
    model: Any,
    encoded: Any,
    tokenizer: Any,
    torch_module: Any,
    max_new_tokens: int | None,
) -> dict[str, Any]:
    with torch_module.inference_mode():
        if operation == "forward_only":
            forward = model(**encoded)
            return {
                "logits_finite": bool(
                    torch_module.isfinite(forward.logits).all().item()
                )
            }
        if operation == "generate_only":
            generated = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        else:
            raise ValueError(f"Unsupported model-smoke operation: {operation}")
    input_tokens = int(encoded["input_ids"].shape[1])
    output_tokens = int(generated.shape[1])
    return {
        "output_tokens": output_tokens,
        "generated_tokens": output_tokens - input_tokens,
        "generated_text": tokenizer.decode(generated[0], skip_special_tokens=True),
    }


def _model_smoke(config_path: Path, scope: dict[str, Any]) -> dict[str, Any]:
    operation = str(scope.get("operation", ""))
    if operation not in {"forward_only", "generate_only"}:
        raise ValueError("model-smoke operation must be forward_only or generate_only")
    if scope.get("local_files_only") is not True:
        raise ValueError("model-smoke requires local_files_only=true")
    if scope.get("trust_remote_code") is not False:
        raise ValueError("model-smoke requires trust_remote_code=false")
    if scope.get("do_sample") is not False:
        raise ValueError("model-smoke requires do_sample=false")
    actual_runner_sha256 = _verify_runner(scope)

    model_path = Path(str(scope["model_path"])).resolve()
    verified_files = _verify_model_files(model_path, scope.get("model_files"))

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    revision = str(scope["revision"])
    device = str(scope["device"])
    dtype_name = str(scope["dtype"])
    dtype_by_name = {"bfloat16": torch.bfloat16, "float32": torch.float32}
    if dtype_name not in dtype_by_name:
        raise ValueError("model-smoke dtype must be bfloat16 or float32")
    prompt = str(scope["prompt"])
    max_new_tokens: int | None = None
    if operation == "forward_only":
        if "max_new_tokens" in scope:
            raise ValueError("forward_only scope must not contain max_new_tokens")
    else:
        max_new_tokens = int(scope["max_new_tokens"])
        if not 1 <= max_new_tokens <= 64:
            raise ValueError("max_new_tokens must be between 1 and 64")

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        revision=revision,
        local_files_only=True,
        trust_remote_code=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        revision=revision,
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
        torch_dtype=dtype_by_name[dtype_name],
    ).to(device)
    model.eval()
    encoded = tokenizer(prompt, return_tensors="pt").to(device)
    operation_result = _run_model_smoke_operation(
        operation,
        model=model,
        encoded=encoded,
        tokenizer=tokenizer,
        torch_module=torch,
        max_new_tokens=max_new_tokens,
    )
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)
    result = {
        "operation": operation,
        "model_path": str(model_path),
        "revision": revision,
        "device": device,
        "dtype": dtype_name,
        "runner_sha256": actual_runner_sha256,
        "model_files_verified": len(verified_files),
        "weight_bytes": verified_files["model.safetensors"]["bytes"],
        "weight_sha256": verified_files["model.safetensors"]["sha256"],
        "input_tokens": int(encoded["input_ids"].shape[1]),
    }
    result.update(operation_result)
    return result


_GENERATION_RECIPE_FIELDS = (
    "kind",
    "parent_ids",
    "split",
    "source",
    "language",
    "base_variant",
    "watermark_family",
    "key_id",
    "attack_id",
    "base_recipe_id",
    "components",
    "overlap_mode",
)


def _base_generation_seed(recipe: dict[str, Any], retry_index: int = 0) -> int:
    if retry_index < 0:
        raise ValueError("Generation retry index cannot be negative")
    payload: dict[str, Any] = {
        "seed": recipe["seed"],
        "recipe_id": recipe["recipe_id"],
    }
    if retry_index:
        payload["retry_index"] = retry_index
    return int(content_hash(payload)[:16], 16) % (2**31)


def _load_approved_generation_partial(
    partial_path: Path,
    *,
    expected_sha256: str | None,
    expected_count: int | None,
    recipes: list[dict[str, Any]],
    allow_failed_rows: bool = False,
) -> list[dict[str, Any]]:
    from cwr_eg.manifest import read_jsonl

    if expected_sha256 is None:
        if partial_path.exists():
            raise RuntimeError("A generation partial exists but is not hash-approved")
        if expected_count is not None:
            raise ValueError("A resumed-document count requires a partial SHA-256")
        return []
    if not partial_path.exists():
        raise FileNotFoundError("The approved generation partial is missing")
    if sha256_file(partial_path) != str(expected_sha256):
        raise RuntimeError("Generation partial SHA-256 mismatch")
    outputs = read_jsonl(partial_path)
    if expected_count is None or len(outputs) != int(expected_count):
        raise RuntimeError("Generation resumed-document count does not match the approved scope")
    recipe_by_id = {str(recipe["recipe_id"]): recipe for recipe in recipes}
    if len(recipe_by_id) != len(recipes):
        raise ValueError("Selected generation recipes contain duplicate ids")
    completed_ids: set[str] = set()
    for row in outputs:
        recipe_id = str(row.get("recipe_id"))
        if recipe_id in completed_ids:
            raise RuntimeError("Generation partial contains a duplicate recipe id")
        if recipe_id not in recipe_by_id:
            raise RuntimeError("Generation partial contains an unapproved recipe id")
        status = row.get("status")
        if status == "generated":
            text = str(row.get("text", ""))
            if not text or sha256_text(text) != str(row.get("text_sha256")):
                raise RuntimeError("Generation partial contains invalid text provenance")
        elif status == "failed" and allow_failed_rows:
            if not row.get("failure_type") or not row.get("failure_message"):
                raise RuntimeError("Generation partial contains invalid failure provenance")
        else:
            raise RuntimeError("Only approved completed rows may seed a new scope")
        recipe = recipe_by_id[recipe_id]
        for field in _GENERATION_RECIPE_FIELDS:
            if field in recipe and row.get(field) != recipe.get(field):
                raise RuntimeError(
                    f"Generation partial changed frozen recipe field: {field}"
                )
        completed_ids.add(recipe_id)
    return outputs


def _generate(config_path: Path, scope: dict[str, Any]) -> dict[str, Any]:
    from cwr_eg.manifest import read_jsonl, write_jsonl
    from cwr_eg.markllm_bridge import MarkLlmBridge, MarkLlmSettings
    from cwr_eg.monitoring import ExperimentMonitor

    runner_sha256 = _verify_runner(scope)
    verified_code = _verify_code_files(scope)
    if scope.get("local_files_only") is not True:
        raise ValueError("generate requires local_files_only=true")
    if scope.get("trust_remote_code") is not False:
        raise ValueError("generate requires trust_remote_code=false")
    if str(scope["dtype"]) != "bfloat16":
        raise ValueError("Approved local generation requires bfloat16")
    bridge_path = Path(__file__).with_name("markllm_bridge.py")
    bridge_sha256 = sha256_file(bridge_path)
    if bridge_sha256 != str(scope["bridge_sha256"]):
        raise RuntimeError("MarkLLM bridge SHA-256 does not match the approved scope")
    model_path = Path(scope["model_path"]).resolve()
    verified_files = _verify_model_files(model_path, scope.get("model_files"))
    repository = Path(scope["markllm_repository"]).resolve()
    _verify_repository(repository, str(scope["markllm_commit"]))
    key_file_sha256 = _load_private_keys(scope)
    if sha256_file(scope["parent_manifest"]) != str(scope["parent_manifest_sha256"]):
        raise RuntimeError("Parent manifest SHA-256 does not match the approved scope")
    if sha256_file(scope["recipe_manifest"]) != str(scope["recipe_manifest_sha256"]):
        raise RuntimeError("Recipe manifest SHA-256 does not match the approved scope")
    parents = {
        row["parent_id"]: row for row in read_jsonl(scope["parent_manifest"])
    }
    generation_kind = str(scope.get("generation_kind", "base_generation"))
    if generation_kind not in {"base_generation", "mixed_document"}:
        raise ValueError("Unsupported generation_kind")
    recipes = [
        row
        for row in read_jsonl(scope["recipe_manifest"])
        if row["kind"] == generation_kind
    ]
    allowed_ids = set(scope.get("recipe_ids", []))
    if allowed_ids:
        recipes = [row for row in recipes if row["recipe_id"] in allowed_ids]
    limit = int(scope.get("limit", len(recipes)))
    recipes = recipes[:limit]
    expected_recipe_count = int(scope["expected_recipe_count"])
    if len(recipes) != expected_recipe_count:
        raise ValueError("Selected generation recipe count does not match the approved scope")
    output_path = Path(scope["output_path"])
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite completed generation output: {output_path}")
    monitor = ExperimentMonitor.from_scope(
        scope, task_id="generate", disk_path=output_path.parent
    )
    if monitor is not None:
        monitor.update(phase=generation_kind, completed=0, total=len(recipes))
    bridge = MarkLlmBridge(
        MarkLlmSettings(
            repository=repository,
            model_path=model_path,
            model_revision=str(scope["revision"]),
            device=str(scope.get("device", "cuda:0")),
            max_new_tokens=int(scope.get("max_new_tokens", 256)),
            do_sample=bool(scope["do_sample"]),
            temperature=float(scope.get("temperature", 0.8)),
            top_p=float(scope.get("top_p", 0.95)),
            no_repeat_ngram_size=int(scope["no_repeat_ngram_size"]),
            local_files_only=bool(scope.get("local_files_only", True)),
        )
    )
    partial_path = output_path.with_suffix(output_path.suffix + ".partial")
    outputs = _load_approved_generation_partial(
        partial_path,
        expected_sha256=scope.get("resume_partial_sha256"),
        expected_count=scope.get("expected_resumed_documents"),
        recipes=recipes,
        allow_failed_rows=scope.get("resume_explicit_failures") is True,
    )
    resumed_documents = len(outputs)
    completed_ids = {str(row["recipe_id"]) for row in outputs}
    selected_ids = {str(row["recipe_id"]) for row in recipes}
    if not completed_ids.issubset(selected_ids):
        raise RuntimeError("Partial generation output contains an unapproved recipe id")
    if monitor is not None and outputs:
        monitor.update(
            phase=generation_kind, completed=len(outputs), total=len(recipes)
        )
    prompt_characters = int(scope.get("prompt_characters", 1000))
    generation_retry_index = int(scope.get("generation_retry_index", 0))
    if generation_retry_index < 0:
        raise ValueError("generation_retry_index cannot be negative")
    maximum_failure_rate = float(scope.get("maximum_failure_rate", 0.0))
    if not 0.0 <= maximum_failure_rate <= 1.0:
        raise ValueError("maximum_failure_rate must lie in [0, 1]")
    for recipe in recipes:
        if str(recipe["recipe_id"]) in completed_ids:
            continue
        if generation_kind == "base_generation":
            parent = parents[recipe["parent_ids"][0]]
            generation_seed = _base_generation_seed(recipe, generation_retry_index)
            try:
                generated = bridge.generate(
                    str(parent["text"])[:prompt_characters],
                    recipe["watermark_family"],
                    recipe["key_id"],
                    seed=generation_seed,
                )
            except Exception as error:
                _record_recipe_failure(
                    outputs=outputs,
                    partial_path=partial_path,
                    recipe=(
                        {**recipe, "generation_retry_index": generation_retry_index}
                        if generation_retry_index
                        else recipe
                    ),
                    error=error,
                    total=len(recipes),
                    maximum_failure_rate=maximum_failure_rate,
                )
                continue
            row = {
                **recipe,
                "source": parent["source"],
                "language": parent["language"],
                "text": generated,
                "text_sha256": sha256_text(generated),
                "watermark_intervals": []
                if recipe["watermark_family"] is None
                else [[0, len(generated)]],
                "boundary_quality": "exact",
                "generation_seed": generation_seed,
                **(
                    {"generation_retry_index": generation_retry_index}
                    if generation_retry_index
                    else {}
                ),
                "status": "generated",
                "model_revision": str(scope["revision"]),
            }
            detection_keys = scope.get("detection_keys_by_family", {}).get(
                recipe["watermark_family"], []
            )
            if detection_keys:
                row["smoke_detection"] = [
                    {
                        "key_id": key_id,
                        "role": "correct" if key_id == recipe["key_id"] else "wrong",
                        **bridge.detect(generated, recipe["watermark_family"], key_id),
                    }
                    for key_id in detection_keys
                ]
        else:
            if recipe.get("overlap_mode") != "adjacent":
                raise ValueError("Only adjacent mixed documents are approved")
            components = list(recipe["components"])
            if len(components) != len(recipe["parent_ids"]) or len(components) < 2:
                raise ValueError("Mixed recipes require aligned component and parent lists")
            generated_components: list[str] = []
            component_records: list[dict[str, Any]] = []
            component_failed = False
            for component_index, (component, parent_id) in enumerate(
                zip(components, recipe["parent_ids"], strict=True)
            ):
                parent = parents[parent_id]
                family = str(component["watermark_family"])
                key_id = f"{family}_key_{component['key_slot']}"
                generation_seed = int(
                    content_hash(
                        {
                            "seed": recipe["seed"],
                            "recipe_id": recipe["recipe_id"],
                            "component_index": component_index,
                            "key_id": key_id,
                        }
                    )[:16],
                    16,
                ) % (2**31)
                try:
                    generated = bridge.generate(
                        str(parent["text"])[:prompt_characters],
                        family,
                        key_id,
                        seed=generation_seed,
                    )
                except Exception as error:
                    _record_recipe_failure(
                        outputs=outputs,
                        partial_path=partial_path,
                        recipe=recipe,
                        error=error,
                        total=len(recipes),
                        maximum_failure_rate=maximum_failure_rate,
                    )
                    component_failed = True
                    break
                if not generated:
                    raise RuntimeError(
                        f"Mixed generation produced empty component: {recipe['recipe_id']}"
                    )
                generated_components.append(generated)
                component_records.append(
                    {
                        "component_index": component_index,
                        "parent_id": parent_id,
                        "source": parent["source"],
                        "language": parent["language"],
                        "watermark_family": family,
                        "key_id": key_id,
                        "generation_seed": generation_seed,
                        "text_sha256": sha256_text(generated),
                    }
                )
            if component_failed:
                continue
            mixed_text, intervals = _compose_adjacent_components(
                generated_components, str(scope.get("component_separator", "\n\n"))
            )
            for record, interval in zip(component_records, intervals, strict=True):
                record["watermark_interval"] = interval
            languages = [str(record["language"]) for record in component_records]
            row = {
                **recipe,
                "source": "mixed",
                "language": languages[0] if len(set(languages)) == 1 else "mixed",
                "languages": languages,
                "watermark_family": "mixed",
                "watermark_families": [
                    str(record["watermark_family"]) for record in component_records
                ],
                "key_id": None,
                "key_ids": [str(record["key_id"]) for record in component_records],
                "component_records": component_records,
                "text": mixed_text,
                "text_sha256": sha256_text(mixed_text),
                "watermark_intervals": intervals,
                "boundary_quality": "exact",
                "status": "generated",
                "model_revision": str(scope["revision"]),
            }
        outputs.append(row)
        write_jsonl(partial_path, outputs)
        if monitor is not None:
            monitor.update(
                phase=generation_kind, completed=len(outputs), total=len(recipes)
            )
        print(
            json.dumps(
                {"generation_progress": len(outputs), "total": len(recipes)},
                sort_keys=True,
            ),
            flush=True,
        )
    partial_path.replace(output_path)
    return {
        "generated": sum(row.get("status") == "generated" for row in outputs),
        "failed": sum(row.get("status") == "failed" for row in outputs),
        "output_path": str(output_path),
        "runner_sha256": runner_sha256,
        "bridge_sha256": bridge_sha256,
        "generation_kind": generation_kind,
        "model_files_verified": len(verified_files),
        "key_file_sha256": key_file_sha256,
        "code_files_verified": len(verified_code),
        "resumed_documents": resumed_documents,
    }


def _compose_adjacent_components(
    component_texts: list[str], separator: str
) -> tuple[str, list[list[int]]]:
    if len(component_texts) < 2 or any(not text for text in component_texts):
        raise ValueError("At least two non-empty component texts are required")
    text_parts: list[str] = []
    intervals: list[list[int]] = []
    cursor = 0
    for index, component_text in enumerate(component_texts):
        if index:
            text_parts.append(separator)
            cursor += len(separator)
        start = cursor
        text_parts.append(component_text)
        cursor += len(component_text)
        intervals.append([start, cursor])
    return "".join(text_parts), intervals


def _deterministic_attack(text: str, attack_id: str, truncation_fraction: float) -> str:
    if attack_id == "truncation":
        if not 0.0 < truncation_fraction < 1.0:
            raise ValueError("truncation_fraction must lie in (0, 1)")
        return text[: max(1, int(len(text) * truncation_fraction))]
    if attack_id == "copy_edit":
        return " ".join(text.split())
    raise ValueError(f"Unsupported deterministic attack: {attack_id}")


def _attack_generate(config_path: Path, scope: dict[str, Any]) -> dict[str, Any]:
    from cwr_eg.manifest import read_jsonl, write_jsonl
    from cwr_eg.monitoring import ExperimentMonitor

    runner_sha256 = _verify_runner(scope)
    verified_code = _verify_code_files(scope)
    if scope.get("local_files_only") is not True:
        raise ValueError("attack-generate requires local_files_only=true")
    if scope.get("trust_remote_code") is not False:
        raise ValueError("attack-generate requires trust_remote_code=false")
    if scope.get("do_sample") is not False:
        raise ValueError("Approved pilot attacks require do_sample=false")
    if str(scope["dtype"]) != "bfloat16":
        raise ValueError("Approved local attacks require bfloat16")
    if sha256_file(scope["input_path"]) != str(scope["input_sha256"]):
        raise RuntimeError("Attack input SHA-256 does not match the approved scope")
    if sha256_file(scope["recipe_manifest"]) != str(scope["recipe_manifest_sha256"]):
        raise RuntimeError("Attack recipe manifest SHA-256 does not match the approved scope")
    model_path = Path(scope["model_path"]).resolve()
    verified_files = _verify_model_files(model_path, scope.get("model_files"))
    inputs = {str(row["recipe_id"]): row for row in read_jsonl(scope["input_path"])}
    allowed_attacks = set(scope["attack_ids"])
    recipes = [
        row
        for row in read_jsonl(scope["recipe_manifest"])
        if row["kind"] == "matched_attack" and row["attack_id"] in allowed_attacks
    ]
    recipes = recipes[: int(scope["limit"])]
    if len(recipes) != int(scope["expected_recipe_count"]):
        raise ValueError("Selected attack recipe count does not match the approved scope")
    if any(str(row["base_recipe_id"]) not in inputs for row in recipes):
        raise ValueError("At least one attack recipe has no generated base document")
    device = str(scope["device"])
    revision = str(scope["revision"])
    prompts = {
        "paraphrase": "Paraphrase the following text while preserving meaning. Return only the rewritten text:\n\n",
        "translation_roundtrip": "Translate the text to the other language and back, preserving meaning. Return only the final text:\n\n",
    }
    model_attack_ids = set(prompts)
    if not allowed_attacks.issubset(model_attack_ids | {"copy_edit", "truncation"}):
        raise ValueError("Unsupported attack id in approved scope")
    output_path = Path(scope["output_path"])
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite completed attack output: {output_path}")
    partial_path = output_path.with_suffix(output_path.suffix + ".partial")
    outputs = _load_approved_generation_partial(
        partial_path,
        expected_sha256=scope.get("resume_partial_sha256"),
        expected_count=scope.get("expected_resumed_documents"),
        recipes=recipes,
        allow_failed_rows=scope.get("resume_explicit_failures") is True,
    )
    resumed_documents = len(outputs)
    completed_ids = {str(row["recipe_id"]) for row in outputs}
    selected_ids = {str(row["recipe_id"]) for row in recipes}
    if not completed_ids.issubset(selected_ids):
        raise RuntimeError("Partial attack output contains an unapproved recipe id")
    maximum_failure_rate = float(scope.get("maximum_failure_rate", 0.0))
    if not 0.0 <= maximum_failure_rate <= 1.0:
        raise ValueError("maximum_failure_rate must lie in [0, 1]")
    monitor = ExperimentMonitor.from_scope(
        scope, task_id="attack-generate", disk_path=output_path.parent
    )
    if monitor is not None:
        monitor.update(
            phase="matched_attack", completed=len(outputs), total=len(recipes)
        )
    pending_model_attacks = any(
        row["attack_id"] in model_attack_ids and row["recipe_id"] not in completed_ids
        for row in recipes
    )
    torch = None
    tokenizer = None
    model = None
    if pending_model_attacks:
        import torch as torch_module
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch = torch_module
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            revision=revision,
            local_files_only=True,
            trust_remote_code=False,
            use_fast=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            revision=revision,
            local_files_only=True,
            trust_remote_code=False,
            use_safetensors=True,
            torch_dtype=torch.bfloat16,
        ).to(device)
        model.eval()
    for recipe in recipes:
        if str(recipe["recipe_id"]) in completed_ids:
            continue
        try:
            base = inputs[str(recipe["base_recipe_id"])]
            attack_id = str(recipe["attack_id"])
            base_text = str(base["text"])
            if attack_id in {"truncation", "copy_edit"}:
                attacked_text = _deterministic_attack(
                    base_text, attack_id, float(scope["truncation_fraction"])
                )
            else:
                assert torch is not None and tokenizer is not None and model is not None
                prompt = prompts[attack_id] + base_text
                encoded = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=int(scope["maximum_input_tokens"]),
                ).to(device)
                with torch.inference_mode():
                    generated = model.generate(
                        **encoded,
                        max_new_tokens=int(scope["max_new_tokens"]),
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                continuation = generated[0, encoded["input_ids"].shape[1] :]
                attacked_text = tokenizer.decode(
                    continuation, skip_special_tokens=True
                ).strip()
            if not attacked_text:
                raise RuntimeError(f"Attack produced empty text: {recipe['recipe_id']}")
        except Exception as error:
            _record_recipe_failure(
                outputs=outputs,
                partial_path=partial_path,
                recipe=recipe,
                error=error,
                total=len(recipes),
                maximum_failure_rate=maximum_failure_rate,
            )
            continue
        row = {
            **base,
            **recipe,
            "text": attacked_text,
            "text_sha256": sha256_text(attacked_text),
            "watermark_intervals": []
            if base["watermark_family"] is None
            else [[0, len(attacked_text)]],
            "boundary_quality": str(recipe["boundary_quality"]),
            "status": "generated",
        }
        if attack_id in model_attack_ids:
            row["attacker_model_revision"] = revision
        outputs.append(row)
        write_jsonl(partial_path, outputs)
        if monitor is not None:
            monitor.update(
                phase="matched_attack", completed=len(outputs), total=len(recipes)
            )
        print(
            json.dumps(
                {"attack_progress": len(outputs), "total": len(recipes)},
                sort_keys=True,
            ),
            flush=True,
        )
    partial_path.replace(output_path)
    return {
        "attacked": sum(row.get("status") == "generated" for row in outputs),
        "failed": sum(row.get("status") == "failed" for row in outputs),
        "output_path": str(output_path),
        "runner_sha256": runner_sha256,
        "model_files_verified": len(verified_files),
        "code_files_verified": len(verified_code),
        "resumed_documents": resumed_documents,
    }


def _assemble_data(config_path: Path, scope: dict[str, Any]) -> dict[str, Any]:
    from cwr_eg.generated_data import assemble_generated_documents
    from cwr_eg.monitoring import ExperimentMonitor

    runner_sha256 = _verify_runner(scope)
    verified_code = _verify_code_files(scope)
    if sha256_file(scope["recipe_manifest"]) != str(scope["recipe_manifest_sha256"]):
        raise RuntimeError("Recipe manifest SHA-256 does not match the approved scope")
    input_entries = scope.get("inputs")
    if not isinstance(input_entries, list) or not input_entries:
        raise ValueError("assemble-data requires generated input files")
    input_paths: list[str] = []
    for entry in input_entries:
        if sha256_file(entry["path"]) != str(entry["sha256"]):
            raise RuntimeError("Generated input SHA-256 does not match the approved scope")
        input_paths.append(str(entry["path"]))
    monitor = ExperimentMonitor.from_scope(
        scope, task_id="assemble-data", disk_path=Path(scope["output_path"]).parent
    )
    if monitor is not None:
        monitor.update(phase="assemble_generated_data", completed=0, total=1)
    result = assemble_generated_documents(
        recipe_manifest=scope["recipe_manifest"],
        input_paths=input_paths,
        output_path=scope["output_path"],
        feature_documents_path=scope["feature_documents_path"],
    )
    if result["recipes"] != int(scope["expected_recipe_count"]):
        raise RuntimeError("Assembled data has an unexpected recipe count")
    if monitor is not None:
        monitor.update(phase="assemble_generated_data", completed=1, total=1)
    result.update(
        {
            "runner_sha256": runner_sha256,
            "code_files_verified": len(verified_code),
        }
    )
    return result


def _atomic_write_feature_npz(
    path: Path, *, extracted: Any, numpy_module: Any
) -> dict[str, int]:
    payload: dict[str, Any] = {
        "metadata_document_id": numpy_module.asarray(extracted.document_id),
        "metadata_extractor_version": numpy_module.asarray(extracted.extractor_version),
        "metadata_normalization_version": numpy_module.asarray(
            extracted.normalization_version
        ),
    }
    dimensions: dict[str, int] = {}
    for name, view in extracted.views.items():
        if not numpy_module.all(numpy_module.isfinite(view.values)):
            raise RuntimeError(f"Non-finite feature values for {extracted.document_id}:{name}")
        dimensions[name] = int(view.values.shape[1])
        payload[f"{name}_values"] = view.values
        payload[f"{name}_mask"] = view.valid_mask
        payload[f"{name}_offsets"] = numpy_module.asarray(
            [
                (-1, -1)
                if interval is None
                else (interval.char_start, interval.char_end)
                for interval in view.raw_intervals
            ],
            dtype=numpy_module.int32,
        )
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        numpy_module.savez_compressed(handle, **payload)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    return dimensions


def _feature_manifest_entry(
    row: dict[str, Any],
    path: Path,
    *,
    extractor_version: str,
    normalization_version: str,
) -> dict[str, Any]:
    return {
        "recipe_id": row["recipe_id"],
        "parent_ids": row["parent_ids"],
        "split": row["split"],
        "source": row.get("source"),
        "language": row["language"],
        "watermark_family": row.get("watermark_family"),
        "key_id": row.get("key_id"),
        "attack_id": row.get("attack_id"),
        "intervention_id": row.get(
            "attack_id", row.get("watermark_family") or "clean"
        ),
        "boundary_quality": row.get("boundary_quality", "exact"),
        "feature_path": str(path),
        "feature_sha256": sha256_file(path),
        "extractor_version": extractor_version,
        "normalization_version": normalization_version,
    }


def _recover_feature_entry(
    row: dict[str, Any], path: Path, numpy_module: Any
) -> tuple[dict[str, Any], dict[str, int]]:
    with numpy_module.load(path, allow_pickle=False) as payload:
        document_id = str(payload["metadata_document_id"].item())
        if document_id != str(row["recipe_id"]):
            raise RuntimeError(f"Orphan feature belongs to another document: {path}")
        extractor_version = str(payload["metadata_extractor_version"].item())
        normalization_version = str(payload["metadata_normalization_version"].item())
        dimensions = {
            key[: -len("_values")]: int(payload[key].shape[1])
            for key in payload.files
            if key.endswith("_values")
        }
        if not dimensions or any(
            not numpy_module.all(numpy_module.isfinite(payload[key]))
            for key in payload.files
            if key.endswith("_values")
        ):
            raise RuntimeError(f"Orphan feature is incomplete or non-finite: {path}")
    return (
        _feature_manifest_entry(
            row,
            path,
            extractor_version=extractor_version,
            normalization_version=normalization_version,
        ),
        dimensions,
    )


def _verify_feature_resume_manifest(
    manifest_path: Path, expected_sha256: str | None
) -> None:
    if expected_sha256 is None:
        return
    if not manifest_path.is_file():
        raise FileNotFoundError("The approved feature resume manifest is missing")
    if sha256_file(manifest_path) != expected_sha256:
        raise RuntimeError("Feature resume manifest SHA-256 mismatch")


def _extract_features(config_path: Path, scope: dict[str, Any]) -> dict[str, Any]:
    import numpy as np

    from cwr_eg.manifest import read_jsonl, write_jsonl
    from cwr_eg.monitoring import ExperimentMonitor
    from cwr_eg.transformer_features import (
        TransformerFeatureSettings,
        TransformersCausalFeatureExtractor,
    )

    runner_sha256 = _verify_runner(scope)
    if scope.get("local_files_only") is not True:
        raise ValueError("extract-features requires local_files_only=true")
    if scope.get("trust_remote_code") is not False:
        raise ValueError("extract-features requires trust_remote_code=false")
    if str(scope["dtype"]) != "bfloat16":
        raise ValueError("Approved local feature extraction requires bfloat16")
    verified_code = _verify_code_files(scope)
    model_path = Path(scope["model_path"]).resolve()
    verified_files = _verify_model_files(model_path, scope.get("model_files"))
    input_sha256 = sha256_file(scope["input_path"])
    if input_sha256 != str(scope["input_sha256"]):
        raise RuntimeError("Feature input SHA-256 does not match the approved scope")
    rows = read_jsonl(scope["input_path"])
    allowed_ids = set(scope.get("recipe_ids", []))
    if allowed_ids:
        rows = [row for row in rows if row["recipe_id"] in allowed_ids]
    rows = rows[: int(scope.get("limit", len(rows)))]
    if len(rows) != int(scope["expected_document_count"]):
        raise ValueError("Selected feature document count does not match the approved scope")
    if len({str(row["recipe_id"]) for row in rows}) != len(rows):
        raise ValueError("Feature input recipe ids must be unique")
    allowed_splits = {str(item) for item in scope.get("allowed_splits", ["train", "dev"])}
    if not allowed_splits or not allowed_splits.issubset(
        {"train", "dev", "calibration", "test"}
    ):
        raise ValueError("Feature allowed_splits contains an unsupported split")
    if any(str(row["split"]) not in allowed_splits for row in rows):
        raise ValueError("Feature input contains a split outside the approved scope")
    output_dir = Path(scope["output_dir"])
    resume = bool(scope.get("resume", False))
    manifest_path = output_dir / "feature_manifest.jsonl"
    _verify_feature_resume_manifest(
        manifest_path, scope.get("resume_manifest_sha256")
    )
    monitor = ExperimentMonitor.from_scope(
        scope, task_id="extract-features", disk_path=output_dir
    )
    if monitor is not None:
        monitor.update(phase="feature_extraction", completed=0, total=len(rows))
    extractor = TransformersCausalFeatureExtractor(
        TransformerFeatureSettings(
            model_path=model_path,
            revision=str(scope["revision"]),
            device=str(scope.get("device", "cuda:0")),
            maximum_tokens=int(scope.get("maximum_tokens", 1024)),
            local_files_only=bool(scope.get("local_files_only", True)),
            trust_remote_code=bool(scope.get("trust_remote_code", False)),
        )
    )
    output_dir.mkdir(parents=True, exist_ok=resume)
    manifest = read_jsonl(manifest_path) if manifest_path.exists() else []
    approved_ids = {str(row["recipe_id"]) for row in rows}
    if any(str(entry["recipe_id"]) not in approved_ids for entry in manifest):
        raise RuntimeError("Existing feature manifest contains an unapproved recipe id")
    completed: dict[str, dict[str, Any]] = {}
    for entry in manifest:
        recipe_id = str(entry["recipe_id"])
        if recipe_id in completed:
            raise RuntimeError("Existing feature manifest contains duplicate recipe ids")
        feature_path = Path(str(entry["feature_path"]))
        if not feature_path.is_file() or sha256_file(feature_path) != str(
            entry["feature_sha256"]
        ):
            raise RuntimeError(f"Feature hash drift detected during resume: {recipe_id}")
        completed[recipe_id] = entry
    expected_resumed_documents = scope.get("expected_resumed_documents")
    if expected_resumed_documents is not None and len(completed) != int(
        expected_resumed_documents
    ):
        raise RuntimeError("Feature resumed-document count does not match the approved scope")
    view_dims: dict[str, int] | None = None
    for row in rows:
        recipe_id = str(row["recipe_id"])
        path = output_dir / f"{recipe_id}.npz"
        if recipe_id in completed:
            with np.load(path, allow_pickle=False) as payload:
                current_view_dims = {
                    key[: -len("_values")]: int(payload[key].shape[1])
                    for key in payload.files
                    if key.endswith("_values")
                }
            if view_dims is None:
                view_dims = current_view_dims
            elif current_view_dims != view_dims:
                raise RuntimeError("Feature view dimensions changed within resumed files")
        elif path.exists():
            recovered, current_view_dims = _recover_feature_entry(row, path, np)
            completed[recipe_id] = recovered
            if view_dims is None:
                view_dims = current_view_dims
            elif current_view_dims != view_dims:
                raise RuntimeError("Feature view dimensions changed in orphan recovery")
    if completed:
        write_jsonl(
            manifest_path,
            [completed[str(row["recipe_id"])] for row in rows if str(row["recipe_id"]) in completed],
        )
        if monitor is not None:
            monitor.update(
                phase="feature_extraction", completed=len(completed), total=len(rows)
            )

    microbatches = [int(item) for item in scope.get("microbatch_sequence", [1])]
    if (
        not microbatches
        or any(item < 1 for item in microbatches)
        or any(left <= right for left, right in zip(microbatches, microbatches[1:]))
    ):
        raise ValueError("microbatch_sequence must be a strictly decreasing positive list")
    batch_index = 0
    pending = [row for row in rows if str(row["recipe_id"]) not in completed]
    cursor = 0
    last_report = time.monotonic()
    report_increment = max(1, math.ceil(len(rows) / 100))
    while cursor < len(pending):
        microbatch = microbatches[batch_index]
        batch_rows = pending[cursor : cursor + microbatch]
        try:
            extracted_batch = extractor.extract_many(
                [
                    (str(row["recipe_id"]), str(row["text"]), str(row["language"]))
                    for row in batch_rows
                ]
            )
        except extractor.torch.OutOfMemoryError:
            if batch_index + 1 >= len(microbatches):
                raise
            batch_index += 1
            if str(scope.get("device", "")).startswith("cuda"):
                extractor.torch.cuda.empty_cache()
            print(
                json.dumps(
                    {
                        "feature_oom_fallback": microbatches[batch_index],
                        "completed": len(completed),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            continue
        if len(extracted_batch) != len(batch_rows):
            raise RuntimeError("Feature extractor returned an incomplete microbatch")
        for row, extracted in zip(batch_rows, extracted_batch, strict=True):
            recipe_id = str(row["recipe_id"])
            path = output_dir / f"{recipe_id}.npz"
            current_view_dims = _atomic_write_feature_npz(
                path, extracted=extracted, numpy_module=np
            )
            if view_dims is None:
                view_dims = current_view_dims
            elif current_view_dims != view_dims:
                raise RuntimeError("Feature view dimensions changed within the approved run")
            completed[recipe_id] = _feature_manifest_entry(
                row,
                path,
                extractor_version=extracted.extractor_version,
                normalization_version=extracted.normalization_version,
            )
            write_jsonl(
                manifest_path,
                [completed[str(item["recipe_id"])] for item in rows if str(item["recipe_id"]) in completed],
            )
            if monitor is not None:
                monitor.update(
                    phase="feature_extraction",
                    completed=len(completed),
                    total=len(rows),
                )
            now = time.monotonic()
            if len(completed) % report_increment == 0 or now - last_report >= 300:
                print(
                    json.dumps(
                        {
                            "feature_progress": len(completed),
                            "total": len(rows),
                            "microbatch": microbatch,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                last_report = now
        cursor += len(batch_rows)
    manifest = [completed[str(row["recipe_id"])] for row in rows]
    write_jsonl(manifest_path, manifest)
    return {
        "documents": len(manifest),
        "resumed_documents": len(rows) - len(pending),
        "feature_manifest": str(manifest_path),
        "feature_manifest_sha256": sha256_file(manifest_path),
        "input_sha256": input_sha256,
        "runner_sha256": runner_sha256,
        "model_files_verified": len(verified_files),
        "code_files_verified": len(verified_code),
        "view_dims": view_dims,
        "final_microbatch": microbatches[batch_index],
    }


def _tensorize(config_path: Path, scope: dict[str, Any]) -> dict[str, Any]:
    from cwr_eg.monitoring import ExperimentMonitor
    from cwr_eg.tensor_bundle import build_sharded_tensor_bundle, build_tensor_bundle

    runner_sha256 = _verify_runner(scope)
    verified_code = _verify_code_files(scope)
    feature_manifest_sha256 = sha256_file(scope["feature_manifest"])
    if feature_manifest_sha256 != str(scope["feature_manifest_sha256"]):
        raise RuntimeError("Feature manifest SHA-256 does not match the approved scope")
    output_path = Path(scope["output_path"])
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite tensor bundle: {output_path}")
    monitor = ExperimentMonitor.from_scope(
        scope, task_id="tensorize", disk_path=output_path
    )
    if monitor is not None:
        monitor.update(phase="tensorize", completed=0, total=1)
    bundle_format = str(scope.get("bundle_format", "legacy-v0"))
    if bundle_format == "sharded-v1":
        result = build_sharded_tensor_bundle(
            feature_manifest=scope["feature_manifest"],
            output_dir=output_path,
            positions=int(scope["positions"]),
            maximum_batch_examples=int(scope["maximum_batch_examples"]),
            maximum_batches_per_shard=int(scope["maximum_batches_per_shard"]),
            excluded_watermark_families=tuple(
                scope.get("excluded_watermark_families", ())
            ),
        )
    elif bundle_format == "legacy-v0":
        result = build_tensor_bundle(
            feature_manifest=scope["feature_manifest"],
            output_path=output_path,
            positions=int(scope["positions"]),
            maximum_batch_examples=int(scope["maximum_batch_examples"]),
        )
    else:
        raise ValueError("Unsupported tensor bundle format")
    if result["train_batches"] != int(scope["expected_train_batches"]):
        raise RuntimeError("Unexpected Train batch count")
    if result["dev_batches"] != int(scope["expected_dev_batches"]):
        raise RuntimeError("Unexpected Dev batch count")
    if "expected_train_consistency_pairs" in scope and result[
        "train_consistency_pairs"
    ] != int(scope["expected_train_consistency_pairs"]):
        raise RuntimeError("Unexpected Train consistency-pair count")
    if "expected_dev_consistency_pairs" in scope and result[
        "dev_consistency_pairs"
    ] != int(scope["expected_dev_consistency_pairs"]):
        raise RuntimeError("Unexpected Dev consistency-pair count")
    if monitor is not None:
        monitor.update(phase="tensorize", completed=1, total=1)
    result.update(
        {
            "feature_manifest_sha256": feature_manifest_sha256,
            "runner_sha256": runner_sha256,
            "code_files_verified": len(verified_code),
        }
    )
    return result


def _train(config_path: Path, scope: dict[str, Any]) -> dict[str, Any]:
    from cwr_eg.losses import LossWeights
    from cwr_eg.modeling import CwrEgModelConfig
    from cwr_eg.monitoring import ExperimentMonitor
    from cwr_eg.training import TrainingSettings, train_from_tensor_bundle

    runner_sha256 = _verify_runner(scope)
    verified_code = _verify_code_files(scope)
    bundle_path = Path(scope["bundle_path"])
    bundle_hash_path = (
        bundle_path / "bundle_index.json" if bundle_path.is_dir() else bundle_path
    )
    bundle_sha256 = sha256_file(bundle_hash_path)
    if bundle_sha256 != str(scope["bundle_sha256"]):
        raise RuntimeError("Tensor bundle SHA-256 does not match the approved scope")
    output_dir = Path(scope["output_dir"])
    resume_checkpoint = scope.get("resume_checkpoint")
    if output_dir.exists() and resume_checkpoint is None:
        raise FileExistsError(f"Refusing to overwrite training output: {output_dir}")
    if resume_checkpoint is not None and sha256_file(resume_checkpoint) != str(
        scope["resume_checkpoint_sha256"]
    ):
        raise RuntimeError("Resume checkpoint SHA-256 does not match the approved scope")
    config = load_yaml(config_path)
    model_config = CwrEgModelConfig(**scope["model_config"])
    loss_config = config["model"]["losses"]
    loss_config = {**loss_config, **dict(scope.get("loss_overrides", {}))}
    supported_loss_keys = {
        "watermark_weight",
        "null_weight",
        "margin_weight",
        "contrastive_weight",
        "reconstruction_weight",
        "orthogonality_weight",
        "scheme_adv_weight",
        "private_scheme_weight",
        "nuisance_adv_weight",
        "boundary_weight",
        "consistency_weight",
        "variance_floor_weight",
        "grl_scale",
        "invariant_margin",
    }
    if set(loss_config) - supported_loss_keys:
        raise ValueError("Training loss configuration contains an unsupported key")
    weights = LossWeights(
        watermark=float(loss_config["watermark_weight"]),
        null=float(loss_config["null_weight"]),
        margin=float(loss_config["margin_weight"]),
        contrastive=float(loss_config["contrastive_weight"]),
        reconstruction=float(loss_config["reconstruction_weight"]),
        orthogonality=float(loss_config["orthogonality_weight"]),
        scheme_adversary=float(loss_config["scheme_adv_weight"]),
        private_scheme=float(loss_config["private_scheme_weight"]),
        nuisance_adversary=float(loss_config["nuisance_adv_weight"]),
        boundary=float(loss_config["boundary_weight"]),
        consistency=float(loss_config["consistency_weight"]),
        variance_floor=float(loss_config["variance_floor_weight"]),
    )
    settings = TrainingSettings(**scope.get("training_settings", {}))
    monitor = ExperimentMonitor.from_scope(
        scope, task_id="train", disk_path=output_dir
    )
    result = train_from_tensor_bundle(
        bundle_path=scope["bundle_path"],
        output_dir=scope["output_dir"],
        model_config=model_config,
        settings=settings,
        loss_weights=weights,
        device_name=str(scope.get("device", "cuda:0")),
        resume_checkpoint=resume_checkpoint,
        progress_callback=(
            None
            if monitor is None
            else lambda phase, completed, total: monitor.update(
                phase=phase, completed=completed, total=total
            )
        ),
    )
    if monitor is not None:
        monitor.update(phase="training_complete", completed=1, total=1)
    if not math.isfinite(float(result["best_dev_total"])):
        raise RuntimeError("Training produced a non-finite Dev objective")
    result.update(
        {
            "bundle_sha256": bundle_sha256,
            "checkpoint_sha256": sha256_file(result["checkpoint"]),
            "latest_checkpoint_sha256": sha256_file(result["latest_checkpoint"]),
            "training_log_sha256": sha256_file(result["training_log"]),
            "runner_sha256": runner_sha256,
            "code_files_verified": len(verified_code),
        }
    )
    return result


def _score_checkpoint(config_path: Path, scope: dict[str, Any]) -> dict[str, Any]:
    from cwr_eg.monitoring import ExperimentMonitor
    from cwr_eg.scoring import score_checkpoint_features

    runner_sha256 = _verify_runner(scope)
    verified_code = _verify_code_files(scope)
    checkpoint_entries = scope.get("checkpoints")
    if not isinstance(checkpoint_entries, list) or not checkpoint_entries:
        raise ValueError("score-checkpoint requires at least one checkpoint")
    checkpoint_paths: list[str] = []
    for entry in checkpoint_entries:
        path = str(entry["path"])
        if sha256_file(path) != str(entry["sha256"]):
            raise RuntimeError(f"Checkpoint SHA-256 mismatch: {path}")
        checkpoint_paths.append(path)
    feature_manifest_sha256 = sha256_file(scope["feature_manifest"])
    if feature_manifest_sha256 != str(scope["feature_manifest_sha256"]):
        raise RuntimeError("Feature manifest SHA-256 does not match the approved scope")
    documents_sha256 = sha256_file(scope["documents_path"])
    if documents_sha256 != str(scope["documents_sha256"]):
        raise RuntimeError("Scoring document SHA-256 does not match the approved scope")
    output_path = Path(scope["output_path"])
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite checkpoint scores: {output_path}")
    config = load_yaml(config_path)
    monitor = ExperimentMonitor.from_scope(
        scope, task_id="score-checkpoint", disk_path=output_path.parent
    )
    result = score_checkpoint_features(
        checkpoint_paths=checkpoint_paths,
        feature_manifest_path=scope["feature_manifest"],
        documents_path=scope["documents_path"],
        output_path=output_path,
        positions=int(scope["positions"]),
        device_name=str(scope.get("device", "cuda:0")),
        minimum_mapping_coverage=float(
            config["validity"]["minimum_mapping_coverage"]
        ),
        recipe_ids=scope.get("recipe_ids"),
        progress_callback=(
            None
            if monitor is None
            else lambda completed, total: monitor.update(
                phase="checkpoint_scoring", completed=completed, total=total
            )
        ),
    )
    if result["documents"] != int(scope["expected_document_count"]):
        raise RuntimeError("Checkpoint scoring produced an unexpected document count")
    result.update(
        {
            "runner_sha256": runner_sha256,
            "code_files_verified": len(verified_code),
            "feature_manifest_sha256": feature_manifest_sha256,
            "documents_sha256": documents_sha256,
        }
    )
    return result


def _score_registered(config_path: Path, scope: dict[str, Any]) -> dict[str, Any]:
    from cwr_eg.markllm_bridge import MarkLlmBridge, MarkLlmSettings
    from cwr_eg.monitoring import ExperimentMonitor
    from cwr_eg.registered_scoring import score_registered_evidence

    runner_sha256 = _verify_runner(scope)
    verified_code = _verify_code_files(scope)
    score_records_sha256 = sha256_file(scope["score_records_path"])
    if score_records_sha256 != str(scope["score_records_sha256"]):
        raise RuntimeError("Generic score records SHA-256 does not match the approved scope")
    model_path = Path(scope["model_path"]).resolve()
    verified_files = _verify_model_files(model_path, scope.get("model_files"))
    repository = Path(scope["markllm_repository"]).resolve()
    _verify_repository(repository, str(scope["markllm_commit"]))
    key_file_sha256 = _load_private_keys(scope)
    output_path = Path(scope["output_path"])
    monitor = ExperimentMonitor.from_scope(
        scope, task_id="score-registered", disk_path=output_path.parent
    )
    bridge = MarkLlmBridge(
        MarkLlmSettings(
            repository=repository,
            model_path=model_path,
            model_revision=str(scope["revision"]),
            device=str(scope.get("device", "cuda:0")),
            max_new_tokens=1,
            do_sample=False,
            local_files_only=True,
        )
    )
    config = load_yaml(config_path)
    result = score_registered_evidence(
        bridge=bridge,
        score_records_path=scope["score_records_path"],
        output_path=output_path,
        families=scope["families"],
        authorized_key_slots=scope["authorized_key_slots"],
        window_lengths=config["search"]["window_char_lengths"],
        stride_fraction=float(config["search"]["stride_fraction"]),
        candidate_quantile=float(config["search"]["candidate_quantile"]),
        merge_gap_chars=int(config["search"]["merge_gap_chars"]),
        include_scheme_only=bool(scope.get("include_scheme_only", False)),
        progress_callback=(
            None
            if monitor is None
            else lambda completed, total: monitor.update(
                phase="registered_scoring", completed=completed, total=total
            )
        ),
    )
    if result["documents"] != int(scope["expected_document_count"]):
        raise RuntimeError("Registered scoring produced an unexpected document count")
    result.update(
        {
            "runner_sha256": runner_sha256,
            "code_files_verified": len(verified_code),
            "model_files_verified": len(verified_files),
            "score_records_sha256": score_records_sha256,
            "key_file_sha256": key_file_sha256,
        }
    )
    return result


def _prepare_calibration(config_path: Path, scope: dict[str, Any]) -> dict[str, Any]:
    from cwr_eg.calibration_records import build_parent_calibration_records
    from cwr_eg.monitoring import ExperimentMonitor

    runner_sha256 = _verify_runner(scope)
    verified_code = _verify_code_files(scope)
    records_sha256 = sha256_file(scope["scored_documents_path"])
    if records_sha256 != str(scope["scored_documents_sha256"]):
        raise RuntimeError("Scored Calibration documents SHA-256 mismatch")
    output_path = Path(scope["output_path"])
    monitor = ExperimentMonitor.from_scope(
        scope, task_id="prepare-calibration", disk_path=output_path.parent
    )
    if monitor is not None:
        monitor.update(phase="prepare_parent_nulls", completed=0, total=1)
    config = load_yaml(config_path)
    result = build_parent_calibration_records(
        scored_documents_path=scope["scored_documents_path"],
        output_path=output_path,
        window_lengths=config["search"]["window_char_lengths"],
        stride_fraction=float(config["search"]["stride_fraction"]),
        candidate_quantile=float(config["search"]["candidate_quantile"]),
        merge_gap_chars=int(config["search"]["merge_gap_chars"]),
    )
    if result["parents"] != int(scope["expected_parent_count"]):
        raise RuntimeError("Calibration preparation produced an unexpected parent count")
    if monitor is not None:
        monitor.update(phase="prepare_parent_nulls", completed=1, total=1)
    result.update(
        {
            "runner_sha256": runner_sha256,
            "code_files_verified": len(verified_code),
            "scored_documents_sha256": records_sha256,
        }
    )
    return result


def _prepare_evaluation(config_path: Path, scope: dict[str, Any]) -> dict[str, Any]:
    from cwr_eg.evaluation_records import build_evaluation_records
    from cwr_eg.monitoring import ExperimentMonitor

    runner_sha256 = _verify_runner(scope)
    verified_code = _verify_code_files(scope)
    decisions_sha256 = sha256_file(scope["decisions_path"])
    documents_sha256 = sha256_file(scope["documents_path"])
    if decisions_sha256 != str(scope["decisions_sha256"]):
        raise RuntimeError("Test decisions SHA-256 does not match the approved scope")
    if documents_sha256 != str(scope["documents_sha256"]):
        raise RuntimeError("Test documents SHA-256 does not match the approved scope")
    output_path = Path(scope["output_path"])
    monitor = ExperimentMonitor.from_scope(
        scope, task_id="prepare-evaluation", disk_path=output_path.parent
    )
    if monitor is not None:
        monitor.update(phase="prepare_test_metrics", completed=0, total=1)
    result = build_evaluation_records(
        decisions_path=scope["decisions_path"],
        documents_path=scope["documents_path"],
        output_path=output_path,
        authorized_key_slots=scope["authorized_key_slots"],
        held_out_family=scope.get("held_out_family"),
    )
    if result["parents"] != int(scope["expected_parent_count"]):
        raise RuntimeError("Evaluation preparation produced an unexpected parent count")
    if monitor is not None:
        monitor.update(phase="prepare_test_metrics", completed=1, total=1)
    result.update(
        {
            "runner_sha256": runner_sha256,
            "code_files_verified": len(verified_code),
            "decisions_sha256": decisions_sha256,
            "documents_sha256": documents_sha256,
        }
    )
    return result


def _calibrate(config_path: Path, scope: dict[str, Any]) -> dict[str, Any]:
    from cwr_eg.bundle import (
        fit_calibration_bundle_from_records,
        fit_parent_calibration_bundle_from_records,
    )
    from cwr_eg.calibration import CalibrationBundleHeader
    from cwr_eg.monitoring import ExperimentMonitor

    runner_sha256 = _verify_runner(scope)
    verified_code = _verify_code_files(scope)
    records_sha256 = sha256_file(scope["records_path"])
    if records_sha256 != str(scope["records_sha256"]):
        raise RuntimeError("Calibration records SHA-256 does not match the approved scope")
    if Path(scope["output_dir"]).exists():
        raise FileExistsError("Refusing to overwrite calibration output")
    monitor = ExperimentMonitor.from_scope(
        scope, task_id="calibrate", disk_path=scope["output_dir"]
    )
    if monitor is not None:
        monitor.update(phase="parent_calibration", completed=0, total=1)
    config = load_yaml(config_path)
    aggregation_unit = str(scope.get("aggregation_unit", "document"))
    if aggregation_unit == "parent_id":
        path = fit_parent_calibration_bundle_from_records(
            records_path=scope["records_path"],
            output_dir=scope["output_dir"],
            header=CalibrationBundleHeader(**scope["header"]),
            validity_rules=config["validity"],
            minimum_parents_per_stratum=int(scope["minimum_parents_per_stratum"]),
        )
    elif aggregation_unit == "document":
        path = fit_calibration_bundle_from_records(
            records_path=scope["records_path"],
            output_dir=scope["output_dir"],
            header=CalibrationBundleHeader(**scope["header"]),
            validity_rules=config["validity"],
        )
    else:
        raise ValueError("Unsupported calibration aggregation unit")
    manifest_path = path / "calibration_manifest.json"
    null_path = path / "null_distributions.npz"
    if monitor is not None:
        monitor.update(phase="parent_calibration", completed=1, total=1)
    return {
        "calibration_bundle": str(path),
        "records_sha256": records_sha256,
        "calibration_manifest_sha256": sha256_file(manifest_path),
        "null_distributions_sha256": sha256_file(null_path),
        "runner_sha256": runner_sha256,
        "code_files_verified": len(verified_code),
        "aggregation_unit": aggregation_unit,
    }


def _inference_character_scores(row: dict[str, Any]) -> Any:
    scores = row.get("character_scores")
    if scores is None:
        scores = row.get("character_logits")
    if scores is None:
        raise ValueError("Inference score record has no character score vector")
    return scores


def _infer(config_path: Path, scope: dict[str, Any]) -> dict[str, Any]:
    from cwr_eg.bundle import CalibrationBundle
    from cwr_eg.contracts import CharacterInterval, RegisteredEvidence
    from cwr_eg.enums import Applicability, KeyStatus, TailDirection
    from cwr_eg.inference import InferencePipeline, InferenceVersions
    from cwr_eg.manifest import read_jsonl, write_jsonl
    from cwr_eg.monitoring import ExperimentMonitor

    runner_sha256 = _verify_runner(scope)
    verified_code = _verify_code_files(scope)
    score_records_sha256 = sha256_file(scope["score_records_path"])
    if score_records_sha256 != str(scope["score_records_sha256"]):
        raise RuntimeError("Inference score records SHA-256 does not match the approved scope")
    calibration_path = Path(scope["calibration_bundle"])
    calibration_manifest_sha256 = sha256_file(
        calibration_path / "calibration_manifest.json"
    )
    null_distributions_sha256 = sha256_file(
        calibration_path / "null_distributions.npz"
    )
    if calibration_manifest_sha256 != str(scope["calibration_manifest_sha256"]):
        raise RuntimeError("Calibration manifest SHA-256 does not match the approved scope")
    if null_distributions_sha256 != str(scope["null_distributions_sha256"]):
        raise RuntimeError("Calibration null SHA-256 does not match the approved scope")
    output_path = Path(scope["output_path"])
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite inference output: {output_path}")
    config = load_yaml(config_path)
    calibration = CalibrationBundle.load(scope["calibration_bundle"])
    rows = read_jsonl(scope["score_records_path"])
    if len(rows) != int(scope["expected_document_count"]):
        raise ValueError("Inference document count does not match the approved scope")
    allowed_splits = {str(item) for item in scope.get("allowed_splits", ["test"])}
    if not allowed_splits or not allowed_splits.issubset(
        {"train", "dev", "calibration", "test"}
    ):
        raise ValueError("Inference allowed_splits contains an unsupported split")
    if any(str(row.get("split")) not in allowed_splits for row in rows):
        raise ValueError("Inference score records contain a split outside the approved scope")
    monitor = ExperimentMonitor.from_scope(
        scope, task_id="infer", disk_path=output_path.parent
    )
    if monitor is not None:
        monitor.update(phase="inference", completed=0, total=len(rows))
    decisions = []
    for row_index, row in enumerate(rows, start=1):
        declared = tuple(
            RegisteredEvidence(
                detector_id=str(item["detector_id"]),
                scheme_id=str(item["scheme_id"]),
                scheme_family=str(item["scheme_family"]),
                key_id_hash=item.get("key_id_hash"),
                key_status=KeyStatus(item["key_status"]),
                interval=CharacterInterval(int(item["char_start"]), int(item["char_end"])),
                raw_statistic=float(item["raw_statistic"]),
                tail_direction=TailDirection(item["tail_direction"]),
                single_test_p=item.get("single_test_p"),
                adjusted_p=item.get("adjusted_p"),
                applicability=Applicability(item["applicability"]),
                reason_codes=tuple(item.get("reason_codes", ())),
                evidence_strength=item.get("evidence_strength"),
                evidence_transform_version=item.get("evidence_transform_version"),
            )
            for item in row.get("registered_evidence", ())
        )

        def registered_scorer(_text: str, candidate: Any) -> tuple[RegisteredEvidence, ...]:
            return tuple(
                item
                for item in declared
                if item.interval == candidate.interval or "full_text" in item.reason_codes
            )

        pipeline = InferencePipeline(
            calibration=calibration,
            registered_scorer=registered_scorer,
            versions=InferenceVersions(**scope["versions"]),
            alpha=float(config["calibration"]["alpha"]),
            window_lengths=config["search"]["window_char_lengths"],
            stride_fraction=float(config["search"]["stride_fraction"]),
            candidate_quantile=float(config["search"]["candidate_quantile"]),
            merge_gap_chars=int(config["search"]["merge_gap_chars"]),
        )
        decisions.append(
            pipeline.infer_from_character_scores(
            document_id=str(row["document_id"]),
            raw_text=str(row["text"]),
            language=str(row["language"]),
            character_scores=_inference_character_scores(row),
            effective_length=int(row["effective_length"]),
            mapping_coverage=float(row["mapping_coverage"]),
            ).to_dict()
        )
        if monitor is not None:
            monitor.update(
                phase="inference", completed=row_index, total=len(rows)
            )
    write_jsonl(output_path, decisions)
    return {
        "documents": len(decisions),
        "output_path": str(output_path),
        "output_sha256": sha256_file(output_path),
        "score_records_sha256": score_records_sha256,
        "calibration_manifest_sha256": calibration_manifest_sha256,
        "null_distributions_sha256": null_distributions_sha256,
        "runner_sha256": runner_sha256,
        "code_files_verified": len(verified_code),
    }


def _evaluate(config_path: Path, scope: dict[str, Any]) -> dict[str, Any]:
    from cwr_eg.contracts import CharacterInterval
    from cwr_eg.enums import DecisionLabel
    from cwr_eg.manifest import read_jsonl
    from cwr_eg.metrics import EvaluationRecord, cluster_bootstrap_macro_f1, evaluate_records
    from cwr_eg.monitoring import ExperimentMonitor

    runner_sha256 = _verify_runner(scope)
    verified_code = _verify_code_files(scope)
    records_sha256 = sha256_file(scope["records_path"])
    if records_sha256 != str(scope["records_sha256"]):
        raise RuntimeError("Evaluation records SHA-256 does not match the approved scope")
    monitor = ExperimentMonitor.from_scope(
        scope, task_id="evaluate", disk_path=Path(scope["records_path"]).parent
    )
    if monitor is not None:
        monitor.update(phase="metric_evaluation", completed=0, total=1)
    rows = read_jsonl(scope["records_path"])
    records = [
        EvaluationRecord(
            parent_id=str(row["parent_id"]),
            true_label=DecisionLabel(row["true_label"]),
            predicted_label=DecisionLabel(row["predicted_label"]),
            score=float(row["score"]),
            knownness_score=row.get("knownness_score"),
            true_intervals=tuple(CharacterInterval(*item) for item in row.get("true_intervals", [])),
            predicted_intervals=tuple(
                CharacterInterval(*item) for item in row.get("predicted_intervals", [])
            ),
            source=row.get("source"),
            language=row.get("language"),
            watermark_family=row.get("watermark_family"),
            key_id=row.get("key_id"),
            attack_id=row.get("attack_id"),
        )
        for row in rows
    ]
    result = evaluate_records(records, stratify_by=tuple(scope.get("stratify_by", ())))
    if len({record.parent_id for record in records}) >= 2:
        result["macro_f1_cluster_bootstrap"] = cluster_bootstrap_macro_f1(
            records,
            replicates=int(scope.get("bootstrap_replicates", 2000)),
            seed=int(scope.get("seed", 20260813)),
        )
    if monitor is not None:
        monitor.update(phase="metric_evaluation", completed=1, total=1)
    result.update(
        {
            "records_sha256": records_sha256,
            "runner_sha256": runner_sha256,
            "code_files_verified": len(verified_code),
        }
    )
    return result


def _benchmark(config_path: Path, scope: dict[str, Any]) -> dict[str, Any]:
    import torch

    device = str(scope.get("device", "cuda:0"))
    size = int(scope.get("matrix_size", 2048))
    repeats = int(scope.get("repeats", 10))
    values = torch.randn((size, size), device=device, dtype=torch.float16)
    for _ in range(2):
        _ = values @ values
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    for _ in range(repeats):
        _ = values @ values
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    return {
        "device": device,
        "matrix_size": size,
        "repeats": repeats,
        "elapsed_seconds": elapsed,
    }
