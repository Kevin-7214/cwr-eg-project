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
    if counts != {"train": 16, "dev": 8, "calibration": 4, "test": 4}:
        raise ValueError("Pilot split counts must remain 16/8/4/4")
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
