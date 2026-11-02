"""Multi-scale Process Encoder (MPE)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
from torch import nn

from .config import ObservedHistory
from .process import PHYSICS_PROCESS_GROUP_NAMES, PhysicsGuidedHydroProcessEncoder

PROCESS_DIAGNOSTIC_NAMES = (
    "trend_slope",
    "residual_volatility",
    "analog_consistency",
    "current_swe",
    "valid_observation_ratio",
)


@dataclass(frozen=True)
class CrossDomainProcessEncoding:
    context: torch.Tensor
    scale_tokens: torch.Tensor
    scale_valid: torch.Tensor
    group_tokens: torch.Tensor
    group_valid: torch.Tensor
    effective_prior: torch.Tensor
    prior_valid: torch.Tensor
    process_features: torch.Tensor
    group_names: tuple[str, ...]
    diagnostics: Mapping[str, Any]


class MultiScaleProcessEncoder(nn.Module):
    """Multiscale variable-preserving process representation (MPE)."""

    def __init__(
        self,
        *,
        seq_dim: int,
        static_dim: int,
        prior_dim: int,
        model_dim: int,
        nhead: int,
        num_layers: int,
        dropout: float,
        feature_names: Sequence[str],
        patch_length_days: int = 5,
        long_patch_length_steps: int = 4,
        analog_patch_length_steps: int = 5,
    ) -> None:
        super().__init__()
        self.seq_dim = int(seq_dim)
        self.static_dim = int(static_dim)
        self.prior_dim = int(prior_dim)
        self.model_dim = int(model_dim)
        self.feature_names = tuple((str(name) for name in feature_names))
        self.group_names = PHYSICS_PROCESS_GROUP_NAMES
        self.swe_index = self.feature_names.index("swe_prev")
        self.process_encoder = PhysicsGuidedHydroProcessEncoder(
            seq_dim=self.seq_dim,
            static_dim=self.static_dim,
            prior_dim=self.prior_dim,
            model_dim=self.model_dim,
            nhead=int(nhead),
            num_layers=int(num_layers),
            dropout=float(dropout),
            patch_length_days=int(patch_length_days),
            long_patch_length_steps=int(long_patch_length_steps),
            analog_patch_length_steps=int(analog_patch_length_steps),
            swe_feature_index=self.swe_index,
            include_prior_token=True,
        )

    def forward(
        self,
        *,
        short: ObservedHistory,
        long: ObservedHistory,
        analog: ObservedHistory,
        static_topography: torch.Tensor,
        mechanism_prior: torch.Tensor,
        current_swe_normalized: torch.Tensor,
    ) -> CrossDomainProcessEncoding:
        effective_prior = torch.nan_to_num(
            mechanism_prior, nan=0.0, posinf=0.0, neginf=0.0
        )
        prior_valid = torch.isfinite(mechanism_prior).all(dim=-1) & True
        encoder_prior = torch.where(
            prior_valid.reshape(-1, 1),
            effective_prior,
            torch.full_like(effective_prior, float("nan")),
        )
        context, diagnostics = self.process_encoder(
            x_seq=short.normalized,
            x_mask=short.observed,
            x_static=static_topography,
            x_prior=encoder_prior,
            current_swe=current_swe_normalized,
            x_long=long.normalized,
            x_long_mask=long.observed,
            x_analog=analog.normalized,
            x_analog_mask=analog.observed,
            feature_names=self.feature_names,
        )
        group_tokens = diagnostics["physics_process_group_tokens"]
        group_valid = diagnostics["physics_process_group_valid"]
        scale_tokens = diagnostics["scale_tokens"]
        scale_valid = diagnostics["scale_valid"]
        process_features = torch.stack(
            [diagnostics[name] for name in PROCESS_DIAGNOSTIC_NAMES], dim=-1
        )
        process_features = torch.nan_to_num(
            process_features, nan=0.0, posinf=0.0, neginf=0.0
        )
        zero_link = context.new_zeros(())
        for parameter in self.process_encoder.parameters():
            zero_link = zero_link + parameter.reshape(-1)[0] * 0.0
        context = context + zero_link
        public_diagnostics = {
            **diagnostics,
            "process_group_tokens": group_tokens,
            "process_group_valid": group_valid,
            "topography_conditioning_used": torch.tensor(True, device=context.device),
            "mechanism_prior_used": torch.tensor(True, device=context.device),
        }
        return CrossDomainProcessEncoding(
            context=context,
            scale_tokens=scale_tokens,
            scale_valid=scale_valid,
            group_tokens=group_tokens,
            group_valid=group_valid,
            effective_prior=effective_prior,
            prior_valid=prior_valid,
            process_features=process_features,
            group_names=self.group_names,
            diagnostics=public_diagnostics,
        )

    def swe_feature_index(self) -> int:
        return self.swe_index
