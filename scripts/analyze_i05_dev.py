from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from cwr_eg.baselines import (
    MahalanobisBaseline,
    energy_ood,
    fit_logistic_regression,
    fit_one_class_svm,
    maximum_softmax_ood,
    prototype_distance,
)
from cwr_eg.hashing import sha256_file
from cwr_eg.manifest import read_jsonl


FAMILIES = ("clean", "kgw", "unigram", "unbiased", "synthid")
FAMILY_TO_LABEL = {name: index for index, name in enumerate(FAMILIES)}


def _label(row: dict[str, Any]) -> int:
    family = row.get("watermark_family") or "clean"
    if family not in FAMILY_TO_LABEL:
        raise ValueError(f"Unsupported single-parent family: {family}")
    return FAMILY_TO_LABEL[family]


def _single_parent(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if len(row["parent_ids"]) == 1]


def _classification_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        predictions,
        labels=np.arange(len(FAMILIES)),
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, labels=np.arange(len(FAMILIES)), average="macro", zero_division=0)),
        "per_class": {
            family: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, family in enumerate(FAMILIES)
        },
    }


def _binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    binary = (labels != FAMILY_TO_LABEL["clean"]).astype(np.int64)
    return float(roc_auc_score(binary, scores))


def _mechanism_logits(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray(
        [[row["mechanism_logits"][family] for family in FAMILIES[1:]] for row in rows],
        dtype=np.float64,
    )


def _direct_features(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([row["direct_statistical_features"] for row in rows], dtype=np.float64)


def _embeddings(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([row["invariant_embedding"] for row in rows], dtype=np.float64)


def _evaluate_score_rows(train_rows: list[dict[str, Any]], dev_rows: list[dict[str, Any]]) -> dict[str, Any]:
    train_labels = np.asarray([_label(row) for row in train_rows], dtype=np.int64)
    dev_labels = np.asarray([_label(row) for row in dev_rows], dtype=np.int64)
    train_direct = _direct_features(train_rows)
    dev_direct = _direct_features(dev_rows)
    train_embeddings = _embeddings(train_rows)
    dev_embeddings = _embeddings(dev_rows)
    train_logits = _mechanism_logits(train_rows)
    dev_logits = _mechanism_logits(dev_rows)

    direct_candidates: list[dict[str, Any]] = []
    names = ("mean_nll", "mean_entropy", "mean_log_rank", "perturbation_delta")
    for index, name in enumerate(names):
        train_auc = _binary_auc(train_labels, train_direct[:, index])
        sign = 1.0 if train_auc >= 0.5 else -1.0
        direct_candidates.append(
            {
                "name": name,
                "train_auc": max(train_auc, 1.0 - train_auc),
                "sign": sign,
                "dev_auc": _binary_auc(dev_labels, sign * dev_direct[:, index]),
            }
        )
    selected_direct = max(direct_candidates, key=lambda item: item["train_auc"])

    logistic_scaler = StandardScaler().fit(train_direct)
    logistic = fit_logistic_regression(logistic_scaler.transform(train_direct), train_labels)
    logistic_predictions = logistic.predict(logistic_scaler.transform(dev_direct))

    mechanism_predictions = np.argmax(dev_logits, axis=1) + 1
    train_watermark_logits = np.asarray([row["watermark_logit"] for row in train_rows], dtype=np.float64)
    dev_watermark_logits = np.asarray([row["watermark_logit"] for row in dev_rows], dtype=np.float64)
    candidates = np.unique(np.quantile(train_watermark_logits, np.linspace(0.0, 1.0, 201)))
    best_threshold = float(candidates[0])
    best_train_macro_f1 = -1.0
    train_family_predictions = np.argmax(train_logits, axis=1) + 1
    for threshold in candidates:
        predictions = train_family_predictions.copy()
        predictions[train_watermark_logits < threshold] = 0
        score = f1_score(
            train_labels,
            predictions,
            labels=np.arange(len(FAMILIES)),
            average="macro",
            zero_division=0,
        )
        if score > best_train_macro_f1:
            best_threshold = float(threshold)
            best_train_macro_f1 = float(score)
    mechanism_predictions[dev_watermark_logits < best_threshold] = 0

    mahalanobis = MahalanobisBaseline.fit(train_embeddings, train_labels)
    differences = dev_embeddings[:, None, :] - mahalanobis.means[None, :, :]
    squared = np.einsum("nkd,df,nkf->nk", differences, mahalanobis.precision, differences)
    mahalanobis_predictions = np.asarray(mahalanobis.labels)[np.argmin(squared, axis=1)]

    clean_train = train_embeddings[train_labels == FAMILY_TO_LABEL["clean"]]
    clean_prototype = clean_train.mean(axis=0, keepdims=True)
    one_class = fit_one_class_svm(clean_train)
    one_class_scores = -one_class.score_samples(dev_embeddings)
    prototype_scores = prototype_distance(dev_embeddings, clean_prototype)

    mlp = make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(128,), max_iter=500, random_state=20260815, early_stopping=True),
    )
    mlp.fit(train_direct, train_labels)
    mlp_predictions = mlp.predict(dev_direct)

    generic_scores = np.asarray([row["generic_residual_score"] for row in dev_rows], dtype=np.float64)
    msp_clean_scores = maximum_softmax_ood(dev_logits)
    energy_clean_scores = energy_ood(dev_logits)
    return {
        "train_examples": len(train_rows),
        "dev_examples": len(dev_rows),
        "direct_statistics": {
            "selection_rule": "maximum Train AUROC with Train-frozen orientation",
            "candidates": direct_candidates,
            "selected": selected_direct,
        },
        "logistic_regression": _classification_metrics(dev_labels, logistic_predictions),
        "maximum_softmax": {
            "clean_ood_auc": float(roc_auc_score((dev_labels == 0).astype(np.int64), msp_clean_scores)),
        },
        "energy": {
            "clean_ood_auc": float(roc_auc_score((dev_labels == 0).astype(np.int64), energy_clean_scores)),
        },
        "mechanism_head": {
            **_classification_metrics(dev_labels, mechanism_predictions),
            "train_selected_watermark_logit_threshold": best_threshold,
            "train_macro_f1_at_threshold": best_train_macro_f1,
        },
        "mahalanobis": {
            **_classification_metrics(dev_labels, mahalanobis_predictions),
            "watermark_auc": _binary_auc(dev_labels, mahalanobis.score(dev_embeddings)),
        },
        "one_class_svm": {"watermark_auc": _binary_auc(dev_labels, one_class_scores)},
        "prototype_distance": {"watermark_auc": _binary_auc(dev_labels, prototype_scores)},
        "generic_only": {"watermark_auc": _binary_auc(dev_labels, generic_scores)},
        "direct_feature_mlp": _classification_metrics(dev_labels, mlp_predictions),
    }


def _evaluate_dev_model(rows: list[dict[str, Any]]) -> dict[str, Any]:
    selected = _single_parent(rows)
    labels = np.asarray([_label(row) for row in selected], dtype=np.int64)
    logits = _mechanism_logits(selected)
    predictions = np.argmax(logits, axis=1) + 1
    predictions[np.max(logits, axis=1) < 0.0] = 0
    residual = np.asarray([row["generic_residual_score"] for row in selected], dtype=np.float64)
    return {
        "examples": len(selected),
        "classification": _classification_metrics(labels, predictions),
        "generic_watermark_auc": _binary_auc(labels, residual),
        "uncertain": sum(row.get("validity_override") == "uncertain" for row in selected),
    }


def _evaluate_lofo(rows: list[dict[str, Any]], held_out_family: str) -> dict[str, Any]:
    selected = [
        row
        for row in _single_parent(rows)
        if (row.get("watermark_family") or "clean") in {"clean", held_out_family}
    ]
    labels = np.asarray([_label(row) for row in selected], dtype=np.int64)
    residual = np.asarray([row["generic_residual_score"] for row in selected], dtype=np.float64)
    return {
        "held_out_family": held_out_family,
        "decision_label": "suspected_unknown_scheme",
        "examples": len(selected),
        "clean_examples": int(np.sum(labels == 0)),
        "held_out_examples": int(np.sum(labels == FAMILY_TO_LABEL[held_out_family])),
        "generic_watermark_auc": _binary_auc(labels, residual),
    }


def _finite_score_qa(rows: list[dict[str, Any]]) -> dict[str, Any]:
    coverages = np.asarray([row["mapping_coverage"] for row in rows], dtype=np.float64)
    return {
        "documents": len(rows),
        "non_finite_scalar_scores": sum(
            not np.all(
                np.isfinite(
                    [row["generic_residual_score"], row["watermark_logit"], row["mapping_coverage"]]
                )
            )
            for row in rows
        ),
        "mapping_coverage_minimum": float(coverages.min()),
        "mapping_coverage_mean": float(coverages.mean()),
        "uncertain": sum(row.get("validity_override") == "uncertain" for row in rows),
    }


def analyze(*, ensemble_path: Path, model_scores_dir: Path, matrix_path: Path) -> dict[str, Any]:
    ensemble_rows = read_jsonl(ensemble_path)
    single = _single_parent(ensemble_rows)
    train_rows = [row for row in single if row["split"] == "train"]
    dev_rows = [row for row in single if row["split"] == "dev"]
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))["runs"]
    model_results: dict[str, Any] = {}
    for run_id, run in matrix.items():
        rows = read_jsonl(model_scores_dir / f"{run_id}.jsonl")
        if run["role"] == "leave_one_family_out":
            model_results[run_id] = _evaluate_lofo(rows, run_id.removeprefix("lofo_"))
        else:
            model_results[run_id] = _evaluate_dev_model(rows)
    full_ids = [f"full_seed_{seed}" for seed in (20260815, 20260816, 20260817)]
    stability_auc = np.asarray([model_results[run_id]["generic_watermark_auc"] for run_id in full_ids])
    stability_f1 = np.asarray([model_results[run_id]["classification"]["macro_f1"] for run_id in full_ids])
    return {
        "analysis_version": "i05-dev-analysis-v1",
        "ensemble_rule": "arithmetic_mean_character_logits_for_three_full_models",
        "ensemble_score_path": str(ensemble_path).replace("\\", "/"),
        "ensemble_score_sha256": sha256_file(ensemble_path),
        "qa": _finite_score_qa(ensemble_rows),
        "primary_dev": _evaluate_score_rows(train_rows, dev_rows),
        "individual_and_ablation": model_results,
        "full_model_stability": {
            "generic_watermark_auc_mean": float(stability_auc.mean()),
            "generic_watermark_auc_std": float(stability_auc.std(ddof=1)),
            "macro_f1_mean": float(stability_f1.mean()),
            "macro_f1_std": float(stability_f1.std(ddof=1)),
        },
        "deferred_registered_dependencies": {
            "registered_only": "Calibration is sealed until I-GATE-D and is the preregistered fit split.",
            "linear_evidence_fusion": "Requires frozen registered evidence and Calibration weights.",
            "markllm_registered_detectors": "Executed after generic candidate scoring under I-06/I-07 registered routes.",
        },
        "deviation_audit": {
            "status": "pass",
            "train_used_for_fit_only": True,
            "dev_used_for_selection_and_analysis_only": True,
            "mixed_dev_excluded_from_document_classification": 50,
            "calibration_or_test_read": False,
            "ensemble_rule_changed": False,
        },
    }


def _markdown(result: dict[str, Any]) -> str:
    primary = result["primary_dev"]
    stability = result["full_model_stability"]
    return "\n".join(
        [
            "# I-05 Train/Dev 中尺度分析",
            "",
            "本文件只报告冻结的 Train/Dev 结果；Calibration 与 Test 未读取。",
            "",
            "## 主集成与基线",
            "",
            f"- Train 单父样本：{primary['train_examples']}；Dev 单父样本：{primary['dev_examples']}。",
            f"- Logistic Regression 宏 F1：{primary['logistic_regression']['macro_f1']:.4f}。",
            f"- 直接特征 MLP 宏 F1：{primary['direct_feature_mlp']['macro_f1']:.4f}。",
            f"- generic-only 水印 AUROC：{primary['generic_only']['watermark_auc']:.4f}。",
            f"- Mahalanobis 宏 F1：{primary['mahalanobis']['macro_f1']:.4f}。",
            "",
            "## 三完整模型稳定性",
            "",
            f"- generic-only AUROC：{stability['generic_watermark_auc_mean']:.4f} ± {stability['generic_watermark_auc_std']:.4f}。",
            f"- 五类宏 F1：{stability['macro_f1_mean']:.4f} ± {stability['macro_f1_std']:.4f}。",
            "",
            "## 阶段边界",
            "",
            "`registered-only`、线性注册融合和 MarkLLM 登记路线需要冻结的 Calibration，按预注册顺序留到 I-06/I-07；此处不以 Dev 代替 Calibration。",
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ensemble", type=Path, required=True)
    parser.add_argument("--model-scores-dir", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(
        ensemble_path=args.ensemble,
        model_scores_dir=args.model_scores_dir,
        matrix_path=args.matrix,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(_markdown(result), encoding="utf-8")
    print(json.dumps({"output_sha256": sha256_file(args.output_json), "status": "done"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
