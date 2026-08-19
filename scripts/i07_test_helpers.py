from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from cwr_eg.baselines import (
    MahalanobisBaseline,
    energy_ood,
    fit_logistic_regression,
    fit_one_class_svm,
    maximum_softmax_ood,
    prototype_distance,
)
from cwr_eg.bundle import CalibrationBundle
from cwr_eg.candidates import generate_candidates, refine_candidate
from cwr_eg.hashing import sha256_file, sha256_text
from cwr_eg.manifest import read_jsonl, write_jsonl

try:
    from analyze_i05_dev import FAMILIES, FAMILY_TO_LABEL, _classification_metrics
except ModuleNotFoundError:
    from scripts.analyze_i05_dev import FAMILIES, FAMILY_TO_LABEL, _classification_metrics


def _single_parent(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if len(row["parent_ids"]) == 1]


def _label(row: dict[str, Any]) -> int:
    return FAMILY_TO_LABEL[row.get("watermark_family") or "clean"]


def _direct(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([row["direct_statistical_features"] for row in rows], dtype=np.float64)


def _embedding(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([row["invariant_embedding"] for row in rows], dtype=np.float64)


def _logits(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray(
        [[row["mechanism_logits"][family] for family in FAMILIES[1:]] for row in rows],
        dtype=np.float64,
    )


def _binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    return float(roc_auc_score((labels != 0).astype(np.int64), scores))


def fit_baseline_bundle(score_path: Path, output_path: Path) -> dict[str, Any]:
    rows = [row for row in _single_parent(read_jsonl(score_path)) if row["split"] == "train"]
    labels = np.asarray([_label(row) for row in rows], dtype=np.int64)
    direct = _direct(rows)
    embeddings = _embedding(rows)
    logits = _logits(rows)

    direct_candidates = []
    for index in range(direct.shape[1]):
        auc = _binary_auc(labels, direct[:, index])
        direct_candidates.append((max(auc, 1.0 - auc), index, 1.0 if auc >= 0.5 else -1.0))
    _, direct_index, direct_sign = max(direct_candidates)

    direct_scaler = StandardScaler().fit(direct)
    logistic = fit_logistic_regression(direct_scaler.transform(direct), labels, seed=20260815)
    mlp = MLPClassifier(
        hidden_layer_sizes=(128,),
        max_iter=500,
        random_state=20260815,
        early_stopping=True,
    ).fit(direct_scaler.transform(direct), labels)
    mahalanobis = MahalanobisBaseline.fit(embeddings, labels)
    clean_embeddings = embeddings[labels == 0]
    one_class_svm = fit_one_class_svm(clean_embeddings)
    clean_prototype = clean_embeddings.mean(axis=0, keepdims=True)

    family_predictions = np.argmax(logits, axis=1) + 1
    watermark_logits = np.asarray([row["watermark_logit"] for row in rows], dtype=np.float64)
    thresholds = np.unique(np.quantile(watermark_logits, np.linspace(0.0, 1.0, 201)))
    best_threshold = float(thresholds[0])
    best_f1 = -1.0
    for threshold in thresholds:
        predictions = family_predictions.copy()
        predictions[watermark_logits < threshold] = 0
        score = f1_score(labels, predictions, labels=np.arange(5), average="macro", zero_division=0)
        if score > best_f1:
            best_threshold = float(threshold)
            best_f1 = float(score)
    payload = {
        "version": "i07-frozen-baselines-v1",
        "train_score_sha256": sha256_file(score_path),
        "train_examples": len(rows),
        "direct_index": direct_index,
        "direct_sign": direct_sign,
        "direct_scaler": direct_scaler,
        "logistic": logistic,
        "mlp": mlp,
        "mahalanobis": mahalanobis,
        "one_class_svm": one_class_svm,
        "clean_prototype": clean_prototype,
        "watermark_logit_threshold": best_threshold,
        "train_macro_f1_at_threshold": best_f1,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, output_path)
    return {
        "path": str(output_path).replace("\\", "/"),
        "sha256": sha256_file(output_path),
        "version": payload["version"],
        "train_examples": len(rows),
    }


def evaluate_baseline_bundle(bundle_path: Path, test_score_path: Path) -> dict[str, Any]:
    bundle = joblib.load(bundle_path)
    rows = _single_parent(read_jsonl(test_score_path))
    labels = np.asarray([_label(row) for row in rows], dtype=np.int64)
    direct = _direct(rows)
    scaled = bundle["direct_scaler"].transform(direct)
    embeddings = _embedding(rows)
    logits = _logits(rows)
    logistic_predictions = bundle["logistic"].predict(scaled)
    mlp_predictions = bundle["mlp"].predict(scaled)
    differences = embeddings[:, None, :] - bundle["mahalanobis"].means[None, :, :]
    squared = np.einsum(
        "nkd,df,nkf->nk",
        differences,
        bundle["mahalanobis"].precision,
        differences,
    )
    mahalanobis_predictions = np.asarray(bundle["mahalanobis"].labels)[np.argmin(squared, axis=1)]
    mechanism_predictions = np.argmax(logits, axis=1) + 1
    watermark_logits = np.asarray([row["watermark_logit"] for row in rows], dtype=np.float64)
    mechanism_predictions[watermark_logits < bundle["watermark_logit_threshold"]] = 0
    generic = np.asarray([row["generic_residual_score"] for row in rows], dtype=np.float64)
    direct_scores = bundle["direct_sign"] * direct[:, bundle["direct_index"]]
    return {
        "examples": len(rows),
        "direct_statistics": {"watermark_auc": _binary_auc(labels, direct_scores)},
        "logistic_regression": _classification_metrics(labels, logistic_predictions),
        "maximum_softmax": {
            "clean_ood_auc": float(roc_auc_score((labels == 0).astype(np.int64), maximum_softmax_ood(logits)))
        },
        "energy": {
            "clean_ood_auc": float(roc_auc_score((labels == 0).astype(np.int64), energy_ood(logits)))
        },
        "mahalanobis": {
            **_classification_metrics(labels, mahalanobis_predictions),
            "watermark_auc": _binary_auc(labels, bundle["mahalanobis"].score(embeddings)),
        },
        "one_class_svm": {
            "watermark_auc": _binary_auc(labels, -bundle["one_class_svm"].score_samples(embeddings))
        },
        "prototype_distance": {
            "watermark_auc": _binary_auc(labels, prototype_distance(embeddings, bundle["clean_prototype"]))
        },
        "generic_only": {"watermark_auc": _binary_auc(labels, generic)},
        "mechanism_head": _classification_metrics(labels, mechanism_predictions),
        "direct_feature_mlp": _classification_metrics(labels, mlp_predictions),
    }


def mask_registered_records(
    source_path: Path,
    output_path: Path,
    *,
    authorized_key_slots: tuple[str, ...],
    excluded_family: str | None = None,
) -> dict[str, Any]:
    allowed_hashes = {
        "sha256:" + sha256_text(f"{family}_key_{slot}")
        for family in FAMILIES[1:]
        if family != excluded_family
        for slot in authorized_key_slots
    }
    output = []
    for row in read_jsonl(source_path):
        evidence = [
            item
            for item in row.get("registered_evidence", ())
            if item["scheme_family"] != excluded_family
            and (
                item["key_status"] == "scheme_only"
                or item.get("key_id_hash") in allowed_hashes
            )
        ]
        search = dict(row["registered_search_space"])
        search["authorized_key_slots"] = list(authorized_key_slots)
        search["families"] = [family for family in search["families"] if family != excluded_family]
        search["tests"] = len(evidence)
        output.append(
            {
                **row,
                "authorization_scenario": "_and_".join(authorized_key_slots),
                "registered_evidence": evidence,
                "registered_search_space": search,
            }
        )
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite masked registered records: {output_path}")
    write_jsonl(output_path, output)
    return {"documents": len(output), "output_sha256": sha256_file(output_path)}


def merge_generic_with_registered(
    generic_path: Path,
    registered_path: Path,
    output_path: Path,
    *,
    excluded_family: str,
) -> dict[str, Any]:
    registered = {str(row["recipe_id"]): row for row in read_jsonl(registered_path)}
    output = []
    for generic in read_jsonl(generic_path):
        recipe_id = str(generic["recipe_id"])
        source = registered[recipe_id]
        evidence = [
            item
            for item in source["registered_evidence"]
            if item["scheme_family"] != excluded_family
        ]
        search = dict(source["registered_search_space"])
        search["families"] = [family for family in search["families"] if family != excluded_family]
        search["tests"] = len(evidence)
        output.append(
            {
                **generic,
                "authorization_scenario": source["authorization_scenario"],
                "registered_evidence": evidence,
                "registered_search_space": search,
            }
        )
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite LOFO merged records: {output_path}")
    write_jsonl(output_path, output)
    return {"documents": len(output), "output_sha256": sha256_file(output_path)}


def calibrated_document_scores(
    rows: list[dict[str, Any]], bundle: CalibrationBundle, search: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    generic_scores = []
    registered_scores = []
    labels = []
    for row in rows:
        candidates = [
            refine_candidate(candidate, row["character_logits"])
            for candidate in generate_candidates(
                row["character_logits"],
                tuple(search["window_char_lengths"]),
                float(search["stride_fraction"]),
                float(search["candidate_quantile"]),
                int(search["merge_gap_chars"]),
            )
        ]
        generic_max = max((item.raw_score for item in candidates), default=0.0)
        registered_max = max(
            (float(item.get("evidence_strength", item["raw_statistic"])) for item in row.get("registered_evidence", ())),
            default=0.0,
        )
        stratum = f"{row['language']}:all"
        generic_scores.append(1.0 - bundle.p_value("generic", stratum, generic_max))
        registered_scores.append(1.0 - bundle.p_value("registered", stratum, registered_max))
        labels.append(0 if row.get("watermark_family") is None else 1)
    return (
        np.asarray(labels, dtype=np.int64),
        np.asarray(generic_scores, dtype=np.float64),
        np.asarray(registered_scores, dtype=np.float64),
    )


def select_fusion_weight(
    rows: list[dict[str, Any]], bundle: CalibrationBundle, search: dict[str, Any]
) -> dict[str, Any]:
    labels, generic, registered = calibrated_document_scores(rows, bundle, search)
    weights = np.linspace(0.0, 1.0, 21)
    candidates = [
        {
            "weight_generic": float(weight),
            "dev_auc": float(roc_auc_score(labels, weight * generic + (1.0 - weight) * registered)),
        }
        for weight in weights
    ]
    selected = max(candidates, key=lambda item: (item["dev_auc"], -abs(item["weight_generic"] - 0.5)))
    return {
        "registered_only_dev_auc": float(roc_auc_score(labels, registered)),
        "candidates": candidates,
        "selected": selected,
    }
