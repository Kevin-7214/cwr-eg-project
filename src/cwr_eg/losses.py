from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor
from torch.nn import functional as F


def paired_contrastive_loss(
    embeddings: Tensor,
    positive_mask: Tensor,
    negative_mask: Tensor,
    pair_weights: Tensor | None = None,
    temperature: float = 0.1,
) -> Tensor:
    if embeddings.ndim != 2 or temperature <= 0.0:
        raise ValueError("Invalid embeddings or temperature")
    count = embeddings.shape[0]
    if positive_mask.shape != (count, count) or negative_mask.shape != (count, count):
        raise ValueError("Pair masks must have shape (batch, batch)")
    candidates = (positive_mask | negative_mask).bool()
    identity = torch.eye(count, dtype=torch.bool, device=embeddings.device)
    candidates &= ~identity
    positives = positive_mask.bool() & candidates
    normalized = F.normalize(embeddings, dim=-1)
    logits = normalized @ normalized.T / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    exp_logits = torch.exp(logits) * candidates.to(logits.dtype)
    denominator = exp_logits.sum(dim=1).clamp_min(torch.finfo(logits.dtype).tiny)
    log_probability = logits - torch.log(denominator).unsqueeze(1)
    weights = positives.to(logits.dtype)
    if pair_weights is not None:
        if pair_weights.shape != (count, count):
            raise ValueError("pair_weights must have shape (batch, batch)")
        weights = weights * pair_weights
    positive_weight = weights.sum(dim=1)
    valid = positive_weight > 0
    if not torch.any(valid):
        return embeddings.sum() * 0.0
    per_anchor = -(log_probability * weights).sum(dim=1) / positive_weight.clamp_min(1.0)
    return per_anchor[valid].mean()


def centered_orthogonality_loss(z_inv: Tensor, z_priv: Tensor) -> Tensor:
    if z_inv.ndim != 2 or z_priv.ndim != 2 or z_inv.shape[0] != z_priv.shape[0]:
        raise ValueError("Representations must be aligned two-dimensional tensors")
    if z_inv.shape[0] < 2:
        return (z_inv.sum() + z_priv.sum()) * 0.0
    inv_centered = z_inv - z_inv.mean(dim=0, keepdim=True)
    priv_centered = z_priv - z_priv.mean(dim=0, keepdim=True)
    covariance = inv_centered.T @ priv_centered / (z_inv.shape[0] - 1)
    return covariance.square().mean()


def variance_floor_loss(values: Tensor, floor: float = 0.1) -> Tensor:
    if values.ndim != 2 or floor <= 0.0:
        raise ValueError("Invalid representation or variance floor")
    if values.shape[0] < 2:
        return values.sum() * 0.0
    standard_deviation = values.std(dim=0, unbiased=True)
    return F.relu(floor - standard_deviation).square().mean()


def null_and_margin_loss(
    z_inv: Tensor,
    watermark_labels: Tensor,
    null_prototype: Tensor,
    margin: float = 1.5,
    sample_weights: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    if z_inv.ndim != 2 or null_prototype.shape != (z_inv.shape[1],) or margin <= 0.0:
        raise ValueError("Invalid invariant representation, prototype, or margin")
    labels = watermark_labels > 0.5
    distances = torch.linalg.vector_norm(z_inv - null_prototype.unsqueeze(0), dim=-1)
    weights = torch.ones_like(distances) if sample_weights is None else sample_weights
    if weights.shape != distances.shape:
        raise ValueError("sample_weights must match batch size")

    def weighted_mean(values: Tensor, selected: Tensor) -> Tensor:
        if not torch.any(selected):
            return z_inv.sum() * 0.0
        selected_weights = weights[selected]
        return (values[selected] * selected_weights).sum() / selected_weights.sum().clamp_min(1e-8)

    null_loss = weighted_mean(distances.square(), ~labels)
    margin_loss = weighted_mean(F.relu(margin - distances).square(), labels)
    return null_loss, margin_loss


def boundary_loss(
    logits: Tensor,
    targets: Tensor,
    valid_mask: Tensor,
    quality_weights: Tensor | None = None,
    epsilon: float = 1e-6,
) -> Tensor:
    if logits.shape != targets.shape or logits.shape != valid_mask.shape:
        raise ValueError("Boundary logits, targets, and mask must share shape")
    mask = valid_mask.to(logits.dtype)
    if quality_weights is not None:
        if quality_weights.shape != (logits.shape[0],):
            raise ValueError("Boundary quality weights must have shape (batch,)")
        mask = mask * quality_weights.to(logits.dtype).unsqueeze(1)
    denominator = mask.sum().clamp_min(1.0)
    bce = (
        F.binary_cross_entropy_with_logits(logits, targets.to(logits.dtype), reduction="none")
        * mask
    ).sum() / denominator
    probabilities = torch.sigmoid(logits) * mask
    target_values = targets.to(logits.dtype) * mask
    intersection = (probabilities * target_values).sum(dim=1)
    dice = (2.0 * intersection + epsilon) / (
        probabilities.sum(dim=1) + target_values.sum(dim=1) + epsilon
    )
    return bce + (1.0 - dice).mean()


def attack_consistency_loss(
    z_inv: Tensor,
    watermark_logits: Tensor,
    pair_indices: Tensor,
    preserves_label: Tensor,
) -> Tensor:
    if pair_indices.ndim != 2 or pair_indices.shape[1] != 2:
        raise ValueError("pair_indices must have shape (pairs, 2)")
    if preserves_label.shape != (pair_indices.shape[0],):
        raise ValueError("preserves_label must align with consistency pairs")
    selected = pair_indices[preserves_label.bool()]
    if not len(selected):
        return z_inv.sum() * 0.0
    left, right = selected[:, 0], selected[:, 1]
    representation = (z_inv[left] - z_inv[right]).square().sum(dim=1).mean()
    left_probability = torch.sigmoid(watermark_logits[left]).clamp(1e-6, 1.0 - 1e-6)
    right_probability = torch.sigmoid(watermark_logits[right]).clamp(1e-6, 1.0 - 1e-6)
    midpoint = 0.5 * (left_probability + right_probability)

    def bernoulli_kl(first: Tensor, second: Tensor) -> Tensor:
        return first * torch.log(first / second) + (1.0 - first) * torch.log(
            (1.0 - first) / (1.0 - second)
        )

    divergence = 0.5 * (
        bernoulli_kl(left_probability, midpoint)
        + bernoulli_kl(right_probability, midpoint)
    ).mean()
    return representation + divergence


@dataclass(frozen=True, slots=True)
class LossWeights:
    watermark: float = 1.0
    null: float = 1.0
    margin: float = 1.0
    contrastive: float = 0.2
    reconstruction: float = 0.1
    orthogonality: float = 0.05
    scheme_adversary: float = 0.1
    private_scheme: float = 0.1
    nuisance_adversary: float = 0.1
    boundary: float = 0.5
    consistency: float = 0.2
    variance_floor: float = 0.01


def cwr_eg_objective(
    outputs: Mapping[str, Tensor | Mapping[str, Tensor]],
    batch: Mapping[str, Tensor],
    weights: LossWeights = LossWeights(),
) -> tuple[Tensor, dict[str, Tensor]]:
    z_inv = outputs["z_inv"]
    z_priv = outputs["z_priv"]
    assert isinstance(z_inv, Tensor) and isinstance(z_priv, Tensor)
    watermark_labels = batch["watermark_labels"].to(z_inv.dtype)
    positive_examples = watermark_labels > 0.5
    watermark = F.binary_cross_entropy_with_logits(
        outputs["watermark_logits"], watermark_labels
    )
    margin_value = batch.get("invariant_margin", 1.5)
    invariant_margin = (
        float(margin_value.item()) if isinstance(margin_value, Tensor) else float(margin_value)
    )
    null_loss, margin_loss = null_and_margin_loss(
        z_inv,
        watermark_labels,
        batch["null_prototype"],
        margin=invariant_margin,
        sample_weights=batch.get("sample_weights"),
    )
    contrastive = paired_contrastive_loss(
        z_inv,
        batch["positive_mask"].bool(),
        batch["negative_mask"].bool(),
        batch.get("pair_weights"),
    )
    orthogonality = centered_orthogonality_loss(z_inv, z_priv)
    reconstruction_map = outputs["reconstructions"]
    target_map = outputs["reconstruction_targets"]
    assert isinstance(reconstruction_map, Mapping) and isinstance(target_map, Mapping)
    reconstruction = torch.stack(
        [F.mse_loss(reconstruction_map[name], target_map[name]) for name in reconstruction_map]
    ).mean()
    if torch.any(positive_examples):
        scheme_adversary = F.cross_entropy(
            outputs["scheme_adv_logits"][positive_examples],
            batch["scheme_labels"][positive_examples],
        )
        private_scheme = F.cross_entropy(
            outputs["private_scheme_logits"][positive_examples],
            batch["scheme_labels"][positive_examples],
        )
    else:
        scheme_adversary = z_inv.sum() * 0.0
        private_scheme = z_priv.sum() * 0.0
    nuisance_map = outputs["nuisance_logits"]
    assert isinstance(nuisance_map, Mapping)
    nuisance_losses: list[Tensor] = []
    for name, logits in nuisance_map.items():
        label_key = f"nuisance_{name}_labels"
        mask_key = f"nuisance_{name}_mask"
        if label_key not in batch:
            continue
        selected = batch.get(mask_key, torch.ones_like(batch[label_key], dtype=torch.bool)).bool()
        if torch.any(selected):
            nuisance_losses.append(F.cross_entropy(logits[selected], batch[label_key][selected]))
    nuisance = torch.stack(nuisance_losses).mean() if nuisance_losses else z_inv.sum() * 0.0
    boundary = (
        boundary_loss(
            outputs["character_logits"],
            batch["boundary_targets"],
            batch["boundary_mask"],
            batch.get("boundary_quality_weights"),
        )
        if "boundary_targets" in batch
        else z_inv.sum() * 0.0
    )
    consistency = (
        attack_consistency_loss(
            z_inv,
            outputs["watermark_logits"],
            batch["consistency_pairs"],
            batch["consistency_preserves_label"],
        )
        if "consistency_pairs" in batch
        else z_inv.sum() * 0.0
    )
    variance = variance_floor_loss(z_inv) + variance_floor_loss(z_priv)
    components = {
        "watermark": watermark,
        "null": null_loss,
        "margin": margin_loss,
        "contrastive": contrastive,
        "reconstruction": reconstruction,
        "orthogonality": orthogonality,
        "scheme_adversary": scheme_adversary,
        "private_scheme": private_scheme,
        "nuisance_adversary": nuisance,
        "boundary": boundary,
        "consistency": consistency,
        "variance_floor": variance,
    }
    total = (
        weights.watermark * watermark
        + weights.null * null_loss
        + weights.margin * margin_loss
        + weights.contrastive * contrastive
        + weights.reconstruction * reconstruction
        + weights.orthogonality * orthogonality
        + weights.scheme_adversary * scheme_adversary
        + weights.private_scheme * private_scheme
        + weights.nuisance_adversary * nuisance
        + weights.boundary * boundary
        + weights.consistency * consistency
        + weights.variance_floor * variance
    )
    return total, components
