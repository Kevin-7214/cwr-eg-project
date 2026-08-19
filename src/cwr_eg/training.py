from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from collections.abc import Callable
from typing import Any, Iterable, Mapping

import torch
from torch import Tensor

from cwr_eg.hashing import sha256_file
from cwr_eg.losses import LossWeights, cwr_eg_objective
from cwr_eg.modeling import CwrEgModel, CwrEgModelConfig
from cwr_eg.tensor_bundle import iter_sharded_batches, load_sharded_bundle_index


@dataclass(frozen=True, slots=True)
class TrainingSettings:
    epochs: int = 20
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0
    grl_scale: float = 1.0
    seed: int = 20260813
    minimum_epochs: int = 1
    early_stopping_patience: int = 0
    deterministic_algorithms: bool = True

    def __post_init__(self) -> None:
        if self.epochs < 1 or not 1 <= self.minimum_epochs <= self.epochs:
            raise ValueError("Training requires 1 <= minimum_epochs <= epochs")
        if self.early_stopping_patience < 0:
            raise ValueError("early_stopping_patience cannot be negative")


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
    batches: Iterable[Mapping[str, Any]],
    *,
    device: torch.device,
    null_prototype: Tensor,
    loss_weights: LossWeights,
    grl_scale: float,
    optimizer: torch.optim.Optimizer | None,
    gradient_clip_norm: float,
    batch_callback: Callable[[int], None] | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals: dict[str, float] = {}
    examples = 0
    for batch_index, batch in enumerate(batches, start=1):
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
        if batch_callback is not None:
            batch_callback(batch_index)
    if not examples:
        raise ValueError("Epoch received no examples")
    return {name: value / examples for name, value in totals.items()}


def _bundle_hash(path: str | Path) -> str:
    source = Path(path)
    if source.is_dir() or source.name == "bundle_index.json":
        index_path, _ = load_sharded_bundle_index(source)
        return sha256_file(index_path)
    return sha256_file(source)


def _batch_factories(
    path: str | Path,
) -> tuple[
    Callable[[], Iterable[Mapping[str, Any]]],
    Callable[[], Iterable[Mapping[str, Any]]],
    int,
    int,
]:
    source = Path(path)
    if source.is_dir() or source.name == "bundle_index.json":
        _, index = load_sharded_bundle_index(source)
        return (
            lambda: iter_sharded_batches(source, "train"),
            lambda: iter_sharded_batches(source, "dev"),
            int(index["splits"]["train"]["batches"]),
            int(index["splits"]["dev"]["batches"]),
        )
    payload = torch.load(source, map_location="cpu", weights_only=True)
    train_batches = list(payload["train_batches"])
    dev_batches = list(payload["dev_batches"])
    return (
        lambda: iter(train_batches),
        lambda: iter(dev_batches),
        len(train_batches),
        len(dev_batches),
    )


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def train_from_tensor_bundle(
    *,
    bundle_path: str | Path,
    output_dir: str | Path,
    model_config: CwrEgModelConfig,
    settings: TrainingSettings,
    loss_weights: LossWeights,
    device_name: str,
    resume_checkpoint: str | Path | None = None,
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> dict[str, Any]:
    torch.manual_seed(settings.seed)
    torch.use_deterministic_algorithms(settings.deterministic_algorithms)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = settings.deterministic_algorithms
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(settings.seed)
    device = torch.device(device_name)
    train_factory, dev_factory, train_batch_count, dev_batch_count = _batch_factories(
        bundle_path
    )
    _validate_batches(train_factory())
    _validate_batches(dev_factory())
    if any(batch["split"] != "train" for batch in train_factory()):
        raise ValueError("train_batches contains a non-Train batch")
    if any(batch["split"] != "dev" for batch in dev_factory()):
        raise ValueError("dev_batches contains a non-Dev batch")

    model = CwrEgModel(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay
    )
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=resume_checkpoint is not None)
    log_path = target / "training.jsonl"
    best_dev = float("inf")
    best_epoch = -1
    bad_epochs = 0
    start_epoch = 0
    source_bundle_hash = _bundle_hash(bundle_path)
    if resume_checkpoint is not None:
        resume_payload = torch.load(
            resume_checkpoint, map_location="cpu", weights_only=True
        )
        if str(resume_payload["source_bundle_sha256"]) != source_bundle_hash:
            raise RuntimeError("Resume checkpoint belongs to another tensor bundle")
        if resume_payload["model_config"] != asdict(model_config):
            raise RuntimeError("Resume checkpoint model configuration changed")
        model.load_state_dict(resume_payload["state_dict"])
        optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
        best_dev = float(resume_payload["best_dev_total"])
        best_epoch = int(resume_payload["best_epoch"])
        bad_epochs = int(resume_payload["bad_epochs"])
        start_epoch = int(resume_payload["next_epoch"])
        torch.set_rng_state(resume_payload["torch_rng_state"])
        if torch.cuda.is_available() and resume_payload.get("cuda_rng_state_all") is not None:
            torch.cuda.set_rng_state_all(resume_payload["cuda_rng_state_all"])

    stopped_early = False
    completed_epochs = start_epoch
    batches_per_epoch = train_batch_count + dev_batch_count
    total_progress_batches = settings.epochs * batches_per_epoch
    for epoch in range(start_epoch, settings.epochs):
        prototype = fit_train_only_null_prototype(model, train_factory(), device)
        train_metrics = _epoch(
            model,
            train_factory(),
            device=device,
            null_prototype=prototype,
            loss_weights=loss_weights,
            grl_scale=settings.grl_scale,
            optimizer=optimizer,
            gradient_clip_norm=settings.gradient_clip_norm,
            batch_callback=(
                None
                if progress_callback is None
                else lambda count, current_epoch=epoch: progress_callback(
                    "train",
                    current_epoch * batches_per_epoch + count,
                    total_progress_batches,
                )
            ),
        )
        prototype = fit_train_only_null_prototype(model, train_factory(), device)
        dev_metrics = _epoch(
            model,
            dev_factory(),
            device=device,
            null_prototype=prototype,
            loss_weights=loss_weights,
            grl_scale=0.0,
            optimizer=None,
            gradient_clip_norm=settings.gradient_clip_norm,
            batch_callback=(
                None
                if progress_callback is None
                else lambda count, current_epoch=epoch: progress_callback(
                    "dev",
                    current_epoch * batches_per_epoch + train_batch_count + count,
                    total_progress_batches,
                )
            ),
        )
        if any(
            not torch.isfinite(torch.tensor(value)).item()
            for value in (*train_metrics.values(), *dev_metrics.values())
        ):
            raise RuntimeError("Training produced a non-finite loss")
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
            bad_epochs = 0
            _atomic_torch_save(
                {
                    "state_dict": model.state_dict(),
                    "model_config": asdict(model_config),
                    "null_prototype": prototype.cpu(),
                    "training_settings": asdict(settings),
                    "loss_weights": asdict(loss_weights),
                    "source_bundle_sha256": source_bundle_hash,
                    "best_epoch": best_epoch,
                    "best_dev_total": best_dev,
                },
                target / "best_checkpoint.pt",
            )
        else:
            bad_epochs += 1
        completed_epochs = epoch + 1
        latest = {
            "state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_config": asdict(model_config),
            "null_prototype": prototype.cpu(),
            "training_settings": asdict(settings),
            "loss_weights": asdict(loss_weights),
            "source_bundle_sha256": source_bundle_hash,
            "best_epoch": best_epoch,
            "best_dev_total": best_dev,
            "bad_epochs": bad_epochs,
            "next_epoch": completed_epochs,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None,
        }
        _atomic_torch_save(latest, target / "latest_checkpoint.pt")
        if (
            settings.early_stopping_patience > 0
            and completed_epochs >= settings.minimum_epochs
            and bad_epochs >= settings.early_stopping_patience
        ):
            stopped_early = True
            break
    return {
        "best_epoch": best_epoch,
        "best_dev_total": best_dev,
        "checkpoint": str(target / "best_checkpoint.pt"),
        "latest_checkpoint": str(target / "latest_checkpoint.pt"),
        "training_log": str(log_path),
        "epochs_completed": completed_epochs,
        "stopped_early": stopped_early,
        "source_bundle_sha256": source_bundle_hash,
    }
