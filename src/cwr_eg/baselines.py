from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


def direct_statistical_features(
    nll: Sequence[float],
    entropy: Sequence[float],
    log_rank: Sequence[float],
    perturbed_nll: Sequence[float] | None = None,
) -> np.ndarray:
    arrays = [np.asarray(values, dtype=np.float64) for values in (nll, entropy, log_rank)]
    if any(array.ndim != 1 or not len(array) for array in arrays):
        raise ValueError("Direct statistics require non-empty one-dimensional arrays")
    curvature = 0.0
    if perturbed_nll is not None:
        perturbed = np.asarray(perturbed_nll, dtype=np.float64)
        if perturbed.ndim != 1 or not len(perturbed):
            raise ValueError("perturbed_nll must be non-empty and one-dimensional")
        curvature = float(perturbed.mean() - arrays[0].mean())
    return np.asarray(
        [arrays[0].mean(), arrays[1].mean(), arrays[2].mean(), curvature],
        dtype=np.float64,
    )


def maximum_softmax_ood(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    shifted = values - values.max(axis=-1, keepdims=True)
    probabilities = np.exp(shifted) / np.exp(shifted).sum(axis=-1, keepdims=True)
    return 1.0 - probabilities.max(axis=-1)


def energy_ood(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    values = np.asarray(logits, dtype=np.float64) / temperature
    maximum = values.max(axis=-1, keepdims=True)
    return -temperature * (
        maximum.squeeze(-1) + np.log(np.exp(values - maximum).sum(axis=-1))
    )


def prototype_distance(
    embeddings: np.ndarray, prototypes: np.ndarray
) -> np.ndarray:
    values = np.asarray(embeddings, dtype=np.float64)
    centers = np.asarray(prototypes, dtype=np.float64)
    if values.ndim != 2 or centers.ndim != 2 or values.shape[1] != centers.shape[1]:
        raise ValueError("Embeddings and prototypes require compatible feature dimensions")
    return np.linalg.norm(values[:, None, :] - centers[None, :, :], axis=-1).min(axis=1)


@dataclass(frozen=True, slots=True)
class MahalanobisBaseline:
    labels: tuple[int, ...]
    means: np.ndarray
    precision: np.ndarray

    @classmethod
    def fit(
        cls, embeddings: np.ndarray, labels: Sequence[int], ridge: float = 1e-5
    ) -> "MahalanobisBaseline":
        values = np.asarray(embeddings, dtype=np.float64)
        targets = np.asarray(labels)
        unique = tuple(int(value) for value in np.unique(targets))
        if values.ndim != 2 or targets.shape != (values.shape[0],) or len(unique) < 2:
            raise ValueError("Mahalanobis fit requires aligned samples from at least two classes")
        means = np.stack([values[targets == label].mean(axis=0) for label in unique])
        residuals = np.concatenate(
            [values[targets == label] - means[index] for index, label in enumerate(unique)],
            axis=0,
        )
        covariance = np.cov(residuals, rowvar=False) + ridge * np.eye(values.shape[1])
        return cls(unique, means, np.linalg.pinv(covariance))

    def score(self, embeddings: np.ndarray) -> np.ndarray:
        values = np.asarray(embeddings, dtype=np.float64)
        differences = values[:, None, :] - self.means[None, :, :]
        squared = np.einsum("nkd,df,nkf->nk", differences, self.precision, differences)
        return np.sqrt(np.clip(squared.min(axis=1), 0.0, None))


def fit_logistic_regression(
    features: np.ndarray, labels: Sequence[int], seed: int = 20260813
) -> Any:
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(max_iter=2000, random_state=seed, class_weight="balanced")
    return model.fit(np.asarray(features, dtype=np.float64), np.asarray(labels))


def fit_one_class_svm(features: np.ndarray, nu: float = 0.05) -> Any:
    from sklearn.svm import OneClassSVM

    if not 0.0 < nu < 1.0:
        raise ValueError("nu must lie in (0, 1)")
    return OneClassSVM(kernel="rbf", gamma="scale", nu=nu).fit(
        np.asarray(features, dtype=np.float64)
    )


def fit_xgboost(features: np.ndarray, labels: Sequence[int], seed: int = 20260813) -> Any:
    from xgboost import XGBClassifier

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        eval_metric="logloss",
    )
    return model.fit(np.asarray(features, dtype=np.float64), np.asarray(labels))


def linear_evidence_fusion(generic_score: np.ndarray, registered_score: np.ndarray, weight: float) -> np.ndarray:
    if not 0.0 <= weight <= 1.0:
        raise ValueError("weight must lie in [0, 1]")
    generic = np.asarray(generic_score, dtype=np.float64)
    registered = np.asarray(registered_score, dtype=np.float64)
    if generic.shape != registered.shape:
        raise ValueError("Evidence arrays must be aligned")
    return weight * generic + (1.0 - weight) * registered


def direct_fusion_mlp(input_dim: int, hidden_dim: int = 128):
    import torch.nn as nn

    if input_dim < 1 or hidden_dim < 1:
        raise ValueError("MLP dimensions must be positive")
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, 1),
    )
