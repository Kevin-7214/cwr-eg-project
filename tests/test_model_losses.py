from __future__ import annotations

import numpy as np
import torch

from cwr_eg.counterfactual import build_pairing_masks
from cwr_eg.losses import LossWeights, centered_orthogonality_loss, cwr_eg_objective
from cwr_eg.modeling import CwrEgModel, CwrEgModelConfig, gradient_reverse


def test_gradient_reverse_changes_only_backward_direction() -> None:
    values = torch.tensor([1.0, 2.0], requires_grad=True)
    reversed_values = gradient_reverse(values, 0.5)
    assert torch.equal(reversed_values, values)
    reversed_values.sum().backward()
    assert torch.equal(values.grad, torch.tensor([-0.5, -0.5]))


def test_orthogonality_returns_differentiable_zero_for_singleton() -> None:
    invariant = torch.ones((1, 2), requires_grad=True)
    private = torch.ones((1, 3), requires_grad=True)
    loss = centered_orthogonality_loss(invariant, private)
    loss.backward()
    assert loss.item() == 0.0


def test_synthetic_cpu_forward_objective_and_backward() -> None:
    torch.manual_seed(7)
    config = CwrEgModelConfig(
        view_dims={"proxy": 4, "representation": 6, "perturbation": 4, "validity": 4},
        hidden_dim=12,
        invariant_dim=5,
        private_dim=4,
        scheme_classes=4,
        nuisance_classes={"language": 2},
        dropout=0.0,
    )
    model = CwrEgModel(config)
    batch_size, positions = 4, 7
    views = {
        name: torch.randn(batch_size, positions, dimension)
        for name, dimension in config.view_dims.items()
    }
    valid_mask = torch.ones((batch_size, positions), dtype=torch.bool)
    outputs = model(views, valid_mask, grl_scale=0.5)
    pairing = build_pairing_masks(
        ["a", "a", "b", "b"], ["clean", "kgw", "clean", "unigram"]
    )
    batch = {
        "watermark_labels": torch.tensor([0.0, 1.0, 0.0, 1.0]),
        "scheme_labels": torch.tensor([0, 0, 0, 1]),
        "positive_mask": torch.tensor(pairing.positive_mask.tolist()),
        "negative_mask": torch.tensor(pairing.negative_mask.tolist()),
        "pair_weights": torch.tensor(pairing.pair_weights.tolist()),
        "null_prototype": torch.zeros(config.invariant_dim),
        "nuisance_language_labels": torch.tensor([0, 0, 1, 1]),
        "boundary_targets": torch.randint(0, 2, (batch_size, positions)).float(),
        "boundary_mask": valid_mask,
        "consistency_pairs": torch.tensor([[0, 1], [2, 3]]),
        "consistency_preserves_label": torch.tensor([False, True]),
    }
    total, components = cwr_eg_objective(outputs, batch, LossWeights())
    assert torch.isfinite(total)
    assert {"null", "margin", "boundary", "consistency"} <= set(components)
    total.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())
