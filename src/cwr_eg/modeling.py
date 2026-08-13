from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import torch
from torch import Tensor, nn


class _GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx: object, inputs: Tensor, scale: float) -> Tensor:
        ctx.scale = scale
        return inputs.view_as(inputs)

    @staticmethod
    def backward(ctx: object, gradients: Tensor) -> tuple[Tensor, None]:
        return -float(ctx.scale) * gradients, None


def gradient_reverse(inputs: Tensor, scale: float) -> Tensor:
    return _GradientReverse.apply(inputs, float(scale))


@dataclass(frozen=True, slots=True)
class CwrEgModelConfig:
    view_dims: Mapping[str, int]
    hidden_dim: int = 256
    invariant_dim: int = 128
    private_dim: int = 128
    scheme_classes: int = 4
    nuisance_classes: Mapping[str, int] = field(default_factory=dict)
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if not self.view_dims or any(value < 1 for value in self.view_dims.values()):
            raise ValueError("Every declared view requires a positive dimension")
        if any(value < 2 for value in self.nuisance_classes.values()):
            raise ValueError("Each nuisance adversary requires at least two classes")
        for value in (
            self.hidden_dim,
            self.invariant_dim,
            self.private_dim,
            self.scheme_classes,
        ):
            if value < 1:
                raise ValueError("Model dimensions and class counts must be positive")


class CwrEgModel(nn.Module):
    def __init__(self, config: CwrEgModelConfig) -> None:
        super().__init__()
        self.config = config
        self.view_names = tuple(config.view_dims)
        self.view_projectors = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(dimension, config.hidden_dim),
                    nn.LayerNorm(config.hidden_dim),
                    nn.GELU(),
                )
                for name, dimension in config.view_dims.items()
            }
        )
        fused_dim = config.hidden_dim * len(self.view_names)
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.invariant_head = nn.Linear(config.hidden_dim, config.invariant_dim)
        self.private_head = nn.Linear(config.hidden_dim, config.private_dim)
        joint_dim = config.invariant_dim + config.private_dim
        self.reconstruction_heads = nn.ModuleDict(
            {name: nn.Linear(joint_dim, dimension) for name, dimension in config.view_dims.items()}
        )
        self.watermark_head = nn.Linear(config.invariant_dim, 1)
        self.residual_head = nn.Linear(config.invariant_dim, 1)
        self.character_head = nn.Linear(config.hidden_dim, 1)
        self.scheme_adversary = nn.Linear(config.invariant_dim, config.scheme_classes)
        self.private_scheme_head = nn.Linear(config.private_dim, config.scheme_classes)
        self.nuisance_adversaries = nn.ModuleDict(
            {
                name: nn.Linear(config.invariant_dim, classes)
                for name, classes in config.nuisance_classes.items()
            }
        )

    @staticmethod
    def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
        weights = mask.to(values.dtype).unsqueeze(-1)
        denominator = weights.sum(dim=1).clamp_min(1.0)
        return (values * weights).sum(dim=1) / denominator

    def forward(
        self,
        views: Mapping[str, Tensor],
        valid_mask: Tensor,
        *,
        grl_scale: float = 1.0,
    ) -> dict[str, Tensor | dict[str, Tensor]]:
        if tuple(views) != self.view_names:
            raise ValueError(
                f"Expected ordered views {self.view_names}, received {tuple(views)}"
            )
        projected: list[Tensor] = []
        pooled_targets: dict[str, Tensor] = {}
        expected_shape: tuple[int, int] | None = None
        for name in self.view_names:
            values = views[name]
            if values.ndim != 3:
                raise ValueError(f"View {name} must have shape (batch, positions, dims)")
            shape = (values.shape[0], values.shape[1])
            if expected_shape is None:
                expected_shape = shape
            elif shape != expected_shape:
                raise ValueError("All views must share batch and position axes")
            projected.append(self.view_projectors[name](values))
            pooled_targets[name] = self._masked_mean(values, valid_mask)
        fused_positions = self.fusion(torch.cat(projected, dim=-1))
        pooled = self._masked_mean(fused_positions, valid_mask)
        z_inv = self.invariant_head(pooled)
        z_priv = self.private_head(pooled)
        joint = torch.cat((z_inv, z_priv), dim=-1)
        reconstructions = {
            name: self.reconstruction_heads[name](joint) for name in self.view_names
        }
        reversed_invariant = gradient_reverse(z_inv, grl_scale)
        return {
            "z_inv": z_inv,
            "z_priv": z_priv,
            "watermark_logits": self.watermark_head(z_inv).squeeze(-1),
            "residual_score": self.residual_head(z_inv).squeeze(-1),
            "character_logits": self.character_head(fused_positions).squeeze(-1),
            "scheme_adv_logits": self.scheme_adversary(
                reversed_invariant
            ),
            "private_scheme_logits": self.private_scheme_head(z_priv),
            "nuisance_logits": {
                name: head(reversed_invariant)
                for name, head in self.nuisance_adversaries.items()
            },
            "reconstructions": reconstructions,
            "reconstruction_targets": pooled_targets,
        }
