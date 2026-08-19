from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from cwr_eg.hashing import content_hash


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Top-level YAML configuration must be an object")
    return payload


def config_hash(path: str | Path) -> str:
    return content_hash(load_yaml(path))


def validate_pilot_config(config: dict[str, Any]) -> None:
    validate_experiment_config(config)
    if config.get("profile", "pilot") != "pilot":
        raise ValueError("validate_pilot_config accepts only the pilot profile")
    counts = config["data"]["split_counts"]
    if counts != {"train": 16, "dev": 8, "calibration": 4, "test": 4}:
        raise ValueError("Pilot split counts must remain 16/8/4/4")


def validate_experiment_config(config: dict[str, Any]) -> None:
    required = {
        "protocol_version",
        "seed",
        "data",
        "model",
        "search",
        "registered",
        "calibration",
        "validity",
        "decision",
        "evaluation",
    }
    missing = required - set(config)
    if missing:
        raise ValueError(f"Missing configuration sections: {sorted(missing)}")
    counts = config["data"]["split_counts"]
    if set(counts) != {"train", "dev", "calibration", "test"}:
        raise ValueError("Split counts must define Train, Dev, Calibration, and Test")
    if any(int(value) < 1 for value in counts.values()):
        raise ValueError("Every split must contain at least one parent")
    alpha = float(config["calibration"]["alpha"])
    if alpha != 0.01:
        raise ValueError("Protocol fixes document-level alpha at 0.01")
    labels = list(config["decision"]["labels"])
    expected = [
        "known_scheme_known_key",
        "known_scheme_unknown_key",
        "suspected_unknown_scheme",
        "uncertain",
        "none",
    ]
    if labels != expected:
        raise ValueError("The five decision labels or their order changed")

    profile = str(config.get("profile", "pilot"))
    if profile == "rtx5060-24h-intermediate":
        expected_counts = {"train": 300, "dev": 100, "calibration": 200, "test": 200}
        if counts != expected_counts:
            raise ValueError("Intermediate split counts must remain 300/100/200/200")
        if int(config["seed"]) != 20260815:
            raise ValueError("Intermediate seed must remain 20260815")
        tensor = config.get("tensor_bundle", {})
        if tensor.get("format") != "sharded-v1" or int(
            tensor.get("maximum_batches_per_shard", 0)
        ) != 16:
            raise ValueError("Intermediate tensor bundle must use sharded-v1 with 16 batches")
        training = config.get("training", {})
        frozen_training = {
            "positions": 256,
            "batch_size": 20,
            "maximum_epochs": 20,
            "minimum_epochs": 5,
            "early_stopping_patience": 4,
        }
        for key, expected_value in frozen_training.items():
            if int(training.get(key, -1)) != expected_value:
                raise ValueError(f"Intermediate training value changed: {key}")
