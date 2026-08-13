from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
from torch import Tensor

from cwr_eg.hashing import sha256_file
from cwr_eg.losses import LossWeights, cwr_eg_objective
from cwr_eg.modeling import CwrEgModel, CwrEgModelConfig


@dataclass(frozen=True, slots=True)
class TrainingSettings:
    epochs: int = 20
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0
    grl_scale: float = 1.0
    seed: int = 20260813


def _to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    return value


def _validate_batches(batches: Iterable[Mapping[str, Any]]) -> None:
    required = {
        "split",
        "views",
        "valid_mask",
        "watermark_labels",
        "scheme_labels",
        "positive_mask",
        "negative_mask",
    }
    for index, batch in enumerate(batches):
        missing = required - set(batch)
        if missing:
            raise ValueError(f"Training batch {index} misses {sorted(missing)}")
        if batch["split"] not in {"train", "dev"}:
            raise ValueError("Training bundles cannot contain Calibration or Test batches")


@torch.inference_mode()
def fit_train_only_null_prototype(
    model: CwrEgModel,
    train_batches: Iterable[Mapping[str, Any]],
    device: torch.device,
) -> Tensor:
    model.eval()
    clean: list[Tensor] = []
    for batch in train_batches:
        if batch["split"] != "train":
            raise ValueError("Null prototype can only use Train batches")
        moved = _to_device(dict(batch), device)
        outputs = model(moved["views"], moved["valid_mask"], grl_scale=0.0)
        selected = moved["watermark_labels"] <= 0.5
        if torch.any(selected):
            clean.append(outputs["z_inv"][selected].detach())
    if not clean:
        raise ValueError("At least one Train clean example is required for null prototype")
    return torch.cat(clean, dim=0).mean(dim=0)


def _epoch(
    model: CwrEgModel,
    batches: list[Mapping[str, Any]],
    *,
    device: torch.device,
    null_prototype: Tensor,
    loss_weights: LossWeights,
    grl_scale: float,
    optimizer: torch.optim.Optimizer | None,
    gradient_clip_norm: float,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals: dict[str, float] = {}
    examples = 0
    for batch in batches:
        moved = _to_device(dict(batch), device)
        moved["null_prototype"] = null_prototype
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            outputs = model(
                moved["views"], moved["valid_mask"], grl_scale=grl_scale
            )
            total, components = cwr_eg_objective(outputs, moved, loss_weights)
            if training:
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                optimizer.step()
        batch_size = int(moved["watermark_labels"].shape[0])
        examples += batch_size
        totals["total"] = totals.get("total", 0.0) + float(total.detach()) * batch_size
        for name, value in components.items():
            totals[name] = totals.get(name, 0.0) + float(value.detach()) * batch_size
    if not examples:
        raise ValueError("Epoch received no examples")
    return {name: value / examples for name, value in totals.items()}


def train_from_tensor_bundle(
    *,
    bundle_path: str | Path,
    output_dir: str | Path,
    model_config: CwrEgModelConfig,
    settings: TrainingSettings,
    loss_weights: LossWeights,
    device_name: str,
) -> dict[str, Any]:
    torch.manual_seed(settings.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(settings.seed)
    device = torch.device(device_name)
    payload = torch.load(bundle_path, map_location="cpu", weights_only=True)
    train_batches = list(payload["train_batches"])
    dev_batches = list(payload["dev_batches"])
    _validate_batches(train_batches)
    _validate_batches(dev_batches)
    if any(batch["split"] != "train" for batch in train_batches):
        raise ValueError("train_batches contains a non-Train batch")
    if any(batch["split"] != "dev" for batch in dev_batches):
        raise ValueError("dev_batches contains a non-Dev batch")

    model = CwrEgModel(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay
    )
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=False)
    log_path = target / "training.jsonl"
    best_dev = float("inf")
    best_epoch = -1
    for epoch in range(settings.epochs):
        prototype = fit_train_only_null_prototype(model, train_batches, device)
        train_metrics = _epoch(
            model,
            train_batches,
            device=device,
            null_prototype=prototype,
            loss_weights=loss_weights,
            grl_scale=settings.grl_scale,
            optimizer=optimizer,
            gradient_clip_norm=settings.gradient_clip_norm,
        )
        dev_metrics = _epoch(
            model,
            dev_batches,
            device=device,
            null_prototype=prototype,
            loss_weights=loss_weights,
            grl_scale=0.0,
            optimizer=None,
            gradient_clip_norm=settings.gradient_clip_norm,
        )
        with log_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(
                json.dumps(
                    {"epoch": epoch, "train": train_metrics, "dev": dev_metrics},
                    sort_keys=True,
                )
                + "\n"
            )
        if dev_metrics["total"] < best_dev:
            best_dev = dev_metrics["total"]
            best_epoch = epoch
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "model_config": asdict(model_config),
                    "null_prototype": prototype.cpu(),
                    "training_settings": asdict(settings),
                    "loss_weights": asdict(loss_weights),
                    "source_bundle_sha256": sha256_file(bundle_path),
                    "best_epoch": best_epoch,
                    "best_dev_total": best_dev,
                },
                target / "best_checkpoint.pt",
            )
    return {
        "best_epoch": best_epoch,
        "best_dev_total": best_dev,
        "checkpoint": str(target / "best_checkpoint.pt"),
        "training_log": str(log_path),
    }
