from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import time
from typing import Any, Callable

from cwr_eg.config import load_yaml


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


def execute_approved_action(
    action: str, config_path: str | Path, scope: dict[str, Any]
) -> int:
    handlers: dict[str, Callable[[Path, dict[str, Any]], dict[str, Any]]] = {
        "cuda-smoke": _cuda_smoke,
        "model-smoke": _model_smoke,
        "generate": _generate,
        "attack-generate": _attack_generate,
        "extract-features": _extract_features,
        "train": _train,
        "calibrate": _calibrate,
        "infer": _infer,
        "evaluate": _evaluate,
        "benchmark": _benchmark,
    }
    result = handlers[action](Path(config_path), scope)
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


def _model_smoke(config_path: Path, scope: dict[str, Any]) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = str(scope["model_path"])
    revision = str(scope["revision"])
    device = str(scope.get("device", "cuda:0"))
    prompt = str(scope.get("prompt", "Explain text watermarking in one sentence."))
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, revision=revision, local_files_only=bool(scope.get("local_files_only", True))
    )
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        revision=revision,
        local_files_only=bool(scope.get("local_files_only", True)),
        torch_dtype=dtype,
    ).to(device)
    model.eval()
    encoded = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
        forward = model(**encoded)
        generated = model.generate(
            **encoded,
            max_new_tokens=int(scope.get("max_new_tokens", 16)),
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)
    return {
        "model_path": model_path,
        "revision": revision,
        "device": device,
        "input_tokens": int(encoded["input_ids"].shape[1]),
        "logits_finite": bool(torch.isfinite(forward.logits).all().item()),
        "output_tokens": int(generated.shape[1]),
        "generated_text": tokenizer.decode(generated[0], skip_special_tokens=True),
    }


def _generate(config_path: Path, scope: dict[str, Any]) -> dict[str, Any]:
    from cwr_eg.manifest import read_jsonl, write_jsonl
    from cwr_eg.markllm_bridge import MarkLlmBridge, MarkLlmSettings

    parents = {
        row["parent_id"]: row for row in read_jsonl(scope["parent_manifest"])
    }
    recipes = [
        row
        for row in read_jsonl(scope["recipe_manifest"])
        if row["kind"] == "base_generation"
    ]
    allowed_ids = set(scope.get("recipe_ids", []))
    if allowed_ids:
        recipes = [row for row in recipes if row["recipe_id"] in allowed_ids]
    limit = int(scope.get("limit", len(recipes)))
    recipes = recipes[:limit]
    if not recipes:
        raise ValueError("No approved generation recipes were selected")
    bridge = MarkLlmBridge(
        MarkLlmSettings(
            repository=Path(scope["markllm_repository"]),
            model_path=Path(scope["model_path"]),
            model_revision=str(scope["revision"]),
            device=str(scope.get("device", "cuda:0")),
            max_new_tokens=int(scope.get("max_new_tokens", 256)),
            temperature=float(scope.get("temperature", 0.8)),
            top_p=float(scope.get("top_p", 0.95)),
            local_files_only=bool(scope.get("local_files_only", True)),
        )
    )
    outputs: list[dict[str, Any]] = []
    prompt_characters = int(scope.get("prompt_characters", 1000))
    for recipe in recipes:
        parent = parents[recipe["parent_ids"][0]]
        generated = bridge.generate(
            str(parent["text"])[:prompt_characters],
            recipe["watermark_family"],
            recipe["key_id"],
        )
        outputs.append(
            {
                **recipe,
                "source": parent["source"],
                "language": parent["language"],
                "text": generated,
                "status": "generated",
                "model_revision": str(scope["revision"]),
            }
        )
    write_jsonl(scope["output_path"], outputs)
    return {"generated": len(outputs), "output_path": str(scope["output_path"])}


def _attack_generate(config_path: Path, scope: dict[str, Any]) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from cwr_eg.manifest import read_jsonl, write_jsonl

    inputs = read_jsonl(scope["input_path"])
    allowed_ids = set(scope.get("base_recipe_ids", []))
    if allowed_ids:
        inputs = [row for row in inputs if row["recipe_id"] in allowed_ids]
    inputs = inputs[: int(scope.get("limit", len(inputs)))]
    device = str(scope.get("device", "cuda:0"))
    model_path = str(scope["model_path"])
    revision = str(scope["revision"])
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, revision=revision, local_files_only=bool(scope.get("local_files_only", True))
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        revision=revision,
        local_files_only=bool(scope.get("local_files_only", True)),
        torch_dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
    ).to(device)
    model.eval()
    attack_id = str(scope["attack_id"])
    prompts = {
        "paraphrase": "Paraphrase the following text while preserving meaning. Return only the rewritten text:\n\n",
        "translation_roundtrip": "Translate the text to the other language and back, preserving meaning. Return only the final text:\n\n",
        "copy_edit": "Lightly copy-edit the following text without changing its meaning. Return only the edited text:\n\n",
    }
    if attack_id not in prompts:
        raise ValueError(f"Unsupported model-based attack: {attack_id}")
    outputs: list[dict[str, Any]] = []
    for row in inputs:
        prompt = prompts[attack_id] + str(row["text"])
        encoded = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=int(scope.get("max_new_tokens", 512)),
                do_sample=bool(scope.get("do_sample", False)),
                pad_token_id=tokenizer.eos_token_id,
            )
        continuation = generated[0, encoded["input_ids"].shape[1] :]
        outputs.append(
            {
                **row,
                "base_recipe_id": row["recipe_id"],
                "recipe_id": f"attack-{attack_id}-{row['recipe_id']}",
                "attack_id": attack_id,
                "text": tokenizer.decode(continuation, skip_special_tokens=True),
                "boundary_quality": "weak",
                "status": "generated",
                "attacker_model_revision": revision,
            }
        )
    write_jsonl(scope["output_path"], outputs)
    return {"attacked": len(outputs), "output_path": str(scope["output_path"])}


def _extract_features(config_path: Path, scope: dict[str, Any]) -> dict[str, Any]:
    import numpy as np

    from cwr_eg.manifest import read_jsonl, write_jsonl
    from cwr_eg.transformer_features import (
        TransformerFeatureSettings,
        TransformersCausalFeatureExtractor,
    )

    rows = read_jsonl(scope["input_path"])
    allowed_ids = set(scope.get("recipe_ids", []))
    if allowed_ids:
        rows = [row for row in rows if row["recipe_id"] in allowed_ids]
    rows = rows[: int(scope.get("limit", len(rows)))]
    extractor = TransformersCausalFeatureExtractor(
        TransformerFeatureSettings(
            model_path=Path(scope["model_path"]),
            revision=str(scope["revision"]),
            device=str(scope.get("device", "cuda:0")),
            maximum_tokens=int(scope.get("maximum_tokens", 1024)),
            local_files_only=bool(scope.get("local_files_only", True)),
        )
    )
    output_dir = Path(scope["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest: list[dict[str, Any]] = []
    for row in rows:
        extracted = extractor.extract(
            str(row["recipe_id"]), str(row["text"]), str(row["language"])
        )
        path = output_dir / f"{row['recipe_id']}.npz"
        payload: dict[str, Any] = {}
        for name, view in extracted.views.items():
            payload[f"{name}_values"] = view.values
            payload[f"{name}_mask"] = view.valid_mask
            payload[f"{name}_offsets"] = np.asarray(
                [
                    (-1, -1)
                    if interval is None
                    else (interval.char_start, interval.char_end)
                    for interval in view.raw_intervals
                ],
                dtype=np.int32,
            )
        np.savez_compressed(path, **payload)
        manifest.append(
            {
                "recipe_id": row["recipe_id"],
                "parent_ids": row["parent_ids"],
                "split": row["split"],
                "language": row["language"],
                "watermark_family": row.get("watermark_family"),
                "key_id": row.get("key_id"),
                "intervention_id": row.get("attack_id", row.get("watermark_family") or "clean"),
                "boundary_quality": row.get("boundary_quality", "exact"),
                "feature_path": str(path),
                "extractor_version": extracted.extractor_version,
                "normalization_version": extracted.normalization_version,
            }
        )
    manifest_path = output_dir / "feature_manifest.jsonl"
    write_jsonl(manifest_path, manifest)
    return {"documents": len(manifest), "feature_manifest": str(manifest_path)}


def _train(config_path: Path, scope: dict[str, Any]) -> dict[str, Any]:
    from cwr_eg.losses import LossWeights
    from cwr_eg.modeling import CwrEgModelConfig
    from cwr_eg.training import TrainingSettings, train_from_tensor_bundle

    config = load_yaml(config_path)
    model_config = CwrEgModelConfig(**scope["model_config"])
    loss_config = config["model"]["losses"]
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
    return train_from_tensor_bundle(
        bundle_path=scope["bundle_path"],
        output_dir=scope["output_dir"],
        model_config=model_config,
        settings=settings,
        loss_weights=weights,
        device_name=str(scope.get("device", "cuda:0")),
    )


def _calibrate(config_path: Path, scope: dict[str, Any]) -> dict[str, Any]:
    from cwr_eg.bundle import fit_calibration_bundle_from_records
    from cwr_eg.calibration import CalibrationBundleHeader

    config = load_yaml(config_path)
    path = fit_calibration_bundle_from_records(
        records_path=scope["records_path"],
        output_dir=scope["output_dir"],
        header=CalibrationBundleHeader(**scope["header"]),
        validity_rules=config["validity"],
    )
    return {"calibration_bundle": str(path)}


def _infer(config_path: Path, scope: dict[str, Any]) -> dict[str, Any]:
    from cwr_eg.bundle import CalibrationBundle
    from cwr_eg.contracts import CharacterInterval, RegisteredEvidence
    from cwr_eg.enums import Applicability, KeyStatus, TailDirection
    from cwr_eg.inference import InferencePipeline, InferenceVersions
    from cwr_eg.manifest import read_jsonl, write_jsonl

    config = load_yaml(config_path)
    calibration = CalibrationBundle.load(scope["calibration_bundle"])
    rows = read_jsonl(scope["score_records_path"])
    decisions = []
    for row in rows:
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
            )
            for item in row.get("registered_evidence", ())
        )

        def registered_scorer(_text: str, candidate: Any) -> tuple[RegisteredEvidence, ...]:
            return tuple(
                item
                for item in declared
                if item.interval.char_start < candidate.interval.char_end
                and candidate.interval.char_start < item.interval.char_end
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
            character_scores=row["character_scores"],
            effective_length=int(row["effective_length"]),
            mapping_coverage=float(row["mapping_coverage"]),
            ).to_dict()
        )
    write_jsonl(scope["output_path"], decisions)
    return {"documents": len(decisions), "output_path": str(scope["output_path"])}


def _evaluate(config_path: Path, scope: dict[str, Any]) -> dict[str, Any]:
    from cwr_eg.contracts import CharacterInterval
    from cwr_eg.enums import DecisionLabel
    from cwr_eg.manifest import read_jsonl
    from cwr_eg.metrics import EvaluationRecord, cluster_bootstrap_macro_f1, evaluate_records

    rows = read_jsonl(scope["records_path"])
    records = [
        EvaluationRecord(
            parent_id=str(row["parent_id"]),
            true_label=DecisionLabel(row["true_label"]),
            predicted_label=DecisionLabel(row["predicted_label"]),
            score=float(row["score"]),
            true_intervals=tuple(CharacterInterval(*item) for item in row.get("true_intervals", [])),
            predicted_intervals=tuple(
                CharacterInterval(*item) for item in row.get("predicted_intervals", [])
            ),
        )
        for row in rows
    ]
    result = evaluate_records(records)
    if len({record.parent_id for record in records}) >= 2:
        result["macro_f1_cluster_bootstrap"] = cluster_bootstrap_macro_f1(
            records,
            replicates=int(scope.get("bootstrap_replicates", 2000)),
            seed=int(scope.get("seed", 20260813)),
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
