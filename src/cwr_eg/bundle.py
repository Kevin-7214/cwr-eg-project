from __future__ import annotations

from dataclasses import asdict, dataclass, field
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
    metadata: dict[str, Any] = field(default_factory=dict)

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
            "metadata": self.metadata,
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
            metadata=dict(manifest.get("metadata", {})),
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


def fit_parent_calibration_bundle_from_records(
    *,
    records_path: str | Path,
    output_dir: str | Path,
    header: CalibrationBundleHeader,
    validity_rules: dict[str, Any],
    minimum_parents_per_stratum: int = 100,
) -> Path:
    if minimum_parents_per_stratum < 1:
        raise ValueError("minimum_parents_per_stratum must be positive")
    grouped: dict[tuple[str, str, str], list[float]] = {}
    parent_strata: dict[str, str] = {}
    with Path(records_path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("split") != "calibration":
                raise ValueError(f"Non-Calibration row at line {line_number}")
            if row.get("is_null_descendant") is not True:
                raise ValueError("Parent calibration accepts only declared null descendants")
            if not row.get("full_search_executed", False):
                raise ValueError("Calibration requires complete candidate/search execution")
            route = str(row["route"])
            if route not in {"generic", "registered", "gap"}:
                raise ValueError(f"Unsupported evidence route: {route}")
            parent_id = str(row.get("parent_id", ""))
            stratum = str(row["stratum"])
            if not parent_id:
                raise ValueError("Parent calibration requires parent_id")
            previous_stratum = parent_strata.setdefault(parent_id, stratum)
            if previous_stratum != stratum:
                raise ValueError("A calibration parent cannot cross null strata")
            value = float(row["maximum_statistic"])
            if not np.isfinite(value):
                raise ValueError("Calibration maximum statistics must be finite")
            grouped.setdefault((route, stratum, parent_id), []).append(value)

    route_values: dict[str, dict[str, list[float]]] = {
        "generic": {},
        "registered": {},
        "gap": {},
    }
    parent_sets: dict[tuple[str, str], set[str]] = {}
    for (route, stratum, parent_id), values in grouped.items():
        route_values[route].setdefault(stratum, []).append(max(values))
        parent_sets.setdefault((route, stratum), set()).add(parent_id)
    strata = sorted({stratum for _, stratum in parent_sets})
    if not strata:
        raise ValueError("Parent calibration records are empty")
    for stratum in strata:
        expected_parents: set[str] | None = None
        for route in ("generic", "registered", "gap"):
            parents = parent_sets.get((route, stratum), set())
            if len(parents) < minimum_parents_per_stratum:
                raise ValueError(
                    f"Calibration stratum {stratum}:{route} has only {len(parents)} parents"
                )
            if expected_parents is None:
                expected_parents = parents
            elif parents != expected_parents:
                raise ValueError("All routes must use identical parent sets per stratum")
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
        metadata={
            "aggregation_unit": "parent_id",
            "descendant_aggregation": "maximum",
            "null_descendants": ["clean", "attacked-clean"],
            "minimum_parents_per_stratum": minimum_parents_per_stratum,
            "parent_counts_by_stratum": {
                stratum: len(parent_sets[("generic", stratum)]) for stratum in strata
            },
            "minimum_empirical_p_by_stratum": {
                stratum: 1.0 / (len(parent_sets[("generic", stratum)]) + 1)
                for stratum in strata
            },
        },
    )
    return bundle.save(output_dir)
