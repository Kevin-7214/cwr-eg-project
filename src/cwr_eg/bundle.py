from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from cwr_eg.calibration import CalibrationBundleHeader, empirical_upper_p
from cwr_eg.hashing import content_hash, sha256_file


@dataclass(slots=True)
class CalibrationBundle:
    header: CalibrationBundleHeader
    generic_null_by_stratum: dict[str, np.ndarray]
    registered_null_by_stratum: dict[str, np.ndarray]
    gap_null_by_stratum: dict[str, np.ndarray]
    validity_rules: dict[str, Any]

    def __post_init__(self) -> None:
        for collection in (
            self.generic_null_by_stratum,
            self.registered_null_by_stratum,
            self.gap_null_by_stratum,
        ):
            for stratum, values in collection.items():
                array = np.sort(np.asarray(values, dtype=np.float64))
                if array.ndim != 1 or not len(array) or not np.all(np.isfinite(array)):
                    raise ValueError(f"Invalid null distribution for {stratum}")
                collection[stratum] = array

    def p_value(self, route: str, stratum: str, value: float) -> float:
        routes = {
            "generic": self.generic_null_by_stratum,
            "registered": self.registered_null_by_stratum,
            "gap": self.gap_null_by_stratum,
        }
        if route not in routes:
            raise ValueError(f"Unsupported calibration route: {route}")
        if stratum not in routes[route]:
            raise KeyError(f"unsupported_calibration_stratum:{stratum}")
        return empirical_upper_p(value, routes[route][stratum])

    def save(self, directory: str | Path) -> Path:
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=False)
        arrays: dict[str, np.ndarray] = {}
        manifest: dict[str, Any] = {
            "header": asdict(self.header),
            "validity_rules": self.validity_rules,
            "routes": {},
        }
        for route, collection in (
            ("generic", self.generic_null_by_stratum),
            ("registered", self.registered_null_by_stratum),
            ("gap", self.gap_null_by_stratum),
        ):
            route_keys: dict[str, str] = {}
            for index, (stratum, values) in enumerate(sorted(collection.items())):
                key = f"{route}_{index}"
                arrays[key] = values
                route_keys[stratum] = key
            manifest["routes"][route] = route_keys
        np.savez_compressed(target / "null_distributions.npz", **arrays)
        manifest["null_distributions_sha256"] = sha256_file(
            target / "null_distributions.npz"
        )
        manifest["bundle_content_hash"] = content_hash(manifest)
        (target / "calibration_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return target

    @classmethod
    def load(
        cls,
        directory: str | Path,
        *,
        expected_header: CalibrationBundleHeader | None = None,
    ) -> "CalibrationBundle":
        source = Path(directory)
        with (source / "calibration_manifest.json").open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        npz_path = source / "null_distributions.npz"
        if sha256_file(npz_path) != manifest["null_distributions_sha256"]:
            raise ValueError("calibration_bundle_checksum_mismatch")
        header = CalibrationBundleHeader(**manifest["header"])
        if expected_header is not None:
            header.require_compatible(expected_header)
        with np.load(npz_path, allow_pickle=False) as arrays:
            route_values = {
                route: {
                    stratum: arrays[key]
                    for stratum, key in manifest["routes"][route].items()
                }
                for route in ("generic", "registered", "gap")
            }
        return cls(
            header=header,
            generic_null_by_stratum=route_values["generic"],
            registered_null_by_stratum=route_values["registered"],
            gap_null_by_stratum=route_values["gap"],
            validity_rules=manifest["validity_rules"],
        )


def fit_calibration_bundle_from_records(
    *,
    records_path: str | Path,
    output_dir: str | Path,
    header: CalibrationBundleHeader,
    validity_rules: dict[str, Any],
) -> Path:
    route_values: dict[str, dict[str, list[float]]] = {
        "generic": {},
        "registered": {},
        "gap": {},
    }
    with Path(records_path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("split") != "calibration":
                raise ValueError(f"Non-Calibration row at line {line_number}")
            if not row.get("full_search_executed", False):
                raise ValueError("Calibration requires complete candidate/search execution")
            route = str(row["route"])
            stratum = str(row["stratum"])
            route_values[route].setdefault(stratum, []).append(float(row["maximum_statistic"]))
    if not all(route_values[route] for route in route_values):
        raise ValueError("All three evidence routes require calibration null records")
    bundle = CalibrationBundle(
        header=header,
        generic_null_by_stratum={
            key: np.asarray(value) for key, value in route_values["generic"].items()
        },
        registered_null_by_stratum={
            key: np.asarray(value) for key, value in route_values["registered"].items()
        },
        gap_null_by_stratum={
            key: np.asarray(value) for key, value in route_values["gap"].items()
        },
        validity_rules=validity_rules,
    )
    return bundle.save(output_dir)
