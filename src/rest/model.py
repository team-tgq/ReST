"""ReST: Reliable Spatiotemporal SWE forecasting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import nn

from .amrm import AdaptiveMissingnessReliabilityModule, ReliabilityEncoding
from .config import ARCHITECTURE_REVISION, ReSTConfig, ReSTInput
from .mpe import CrossDomainProcessEncoding, MultiScaleProcessEncoder
from .sdef import ScaleConsistentDecomposedExpertForecasting, WindowScaleState


@dataclass(frozen=True)
class ReSTOutput:
    prediction: torch.Tensor
    raw_prediction: torch.Tensor
    base: torch.Tensor
    trend: torch.Tensor
    residual: torch.Tensor
    expert_forecast: torch.Tensor
    expert_base: torch.Tensor
    expert_trend: torch.Tensor
    expert_residual: torch.Tensor
    router_weights: torch.Tensor
    router_load_balance_loss: torch.Tensor
    auxiliary_loss: torch.Tensor
    reliability: ReliabilityEncoding
    process: CrossDomainProcessEncoding
    scale: WindowScaleState
    swe_free_anchor_mm: torch.Tensor
    swe_free_anchor_used: torch.Tensor
    swe_free_anchor_attention: torch.Tensor
    effective_anchor_swe_mm: torch.Tensor
    decomposition_used: torch.Tensor
    diagnostics: Mapping[str, torch.Tensor]


class ReST(nn.Module):
    """ReST composed of AMRM, MPE, and SDEF."""

    architecture_revision = ARCHITECTURE_REVISION

    def __init__(self, seq_dim: int, static_dim: int, config: ReSTConfig) -> None:
        super().__init__()
        if not isinstance(config, ReSTConfig):
            raise TypeError("config must be ReSTConfig")
        if int(seq_dim) != len(config.feature_names):
            raise ValueError(
                f"seq_dim={seq_dim} does not match feature schema length={len(config.feature_names)}"
            )
        self.seq_dim = int(seq_dim)
        self.static_dim = int(static_dim)
        self.config = config
        self.architecture_revision = config.architecture_revision
        self.amrm = AdaptiveMissingnessReliabilityModule(
            model_dim=config.model_dim,
            static_dim=self.static_dim,
            prior_dim=config.prior_dim,
            feature_names=config.feature_names,
            config=config.amrm,
            replay_config=config.structured_replay,
            dropout=config.dropout,
        )
        self.mpe = MultiScaleProcessEncoder(
            seq_dim=self.seq_dim,
            static_dim=self.static_dim,
            prior_dim=config.prior_dim,
            model_dim=config.model_dim,
            nhead=config.nhead,
            num_layers=config.encoder_num_layers,
            dropout=config.dropout,
            feature_names=config.feature_names,
            patch_length_days=config.patch_length_days,
            long_patch_length_steps=config.long_patch_length_steps,
            analog_patch_length_steps=config.analog_patch_length_steps,
        )
        self.sdef = ScaleConsistentDecomposedExpertForecasting(
            seq_dim=self.seq_dim,
            model_dim=config.model_dim,
            prior_dim=config.prior_dim,
            horizon=config.horizon,
            hidden_dim=config.hidden_dim,
            num_experts=config.num_experts,
            router_top_k=config.router_top_k,
            router_prior_strength=config.router_prior_strength,
            dropout=config.dropout,
            window_scale_eps=config.window_scale_eps,
        )

    def forward(self, inputs: ReSTInput) -> ReSTOutput:
        if not isinstance(inputs, ReSTInput):
            raise TypeError("inputs must be ReSTInput")
        inputs.validate(
            feature_dim=self.seq_dim,
            static_dim=self.static_dim,
            prior_dim=self.config.prior_dim,
        )
        short, replay_diagnostics = self.amrm.replay_short(inputs.short)
        scale = self.sdef.prepare_inputs(short, inputs.long, inputs.analog)
        reliability = self.amrm.encode(
            short=scale.short, long=scale.long, analog=scale.analog
        )
        process = self.mpe(
            short=scale.short,
            long=scale.long,
            analog=scale.analog,
            static_topography=inputs.static_topography,
            mechanism_prior=inputs.mechanism_prior,
            current_swe_normalized=reliability.safe_current_swe_normalized,
        )
        context, adaptation_diagnostics = self.amrm.adapt_context(
            context=process.context,
            reliability=reliability,
            short=scale.short,
            group_tokens=process.group_tokens,
            group_valid=process.group_valid,
        )
        swe_free = self.amrm.infer_swe_free_anchor(
            group_tokens=process.group_tokens,
            group_valid=process.group_valid,
            group_names=process.group_names,
            static_topography=inputs.static_topography,
            mechanism_prior=process.effective_prior,
            mechanism_prior_valid=process.prior_valid,
            reliability=reliability,
        )
        forecast = self.sdef.forecast(
            context=context,
            process=process,
            reliability=reliability,
            swe_free=swe_free,
            scale=scale,
            swe_index=self.mpe.swe_feature_index(),
        )
        anchor_used = ~reliability.has_any_swe_history & swe_free.evidence_valid
        effective_anchor = torch.where(
            anchor_used, swe_free.anchor_mm, reliability.effective_anchor_swe_mm
        ).clamp_min(0.0)
        auxiliary_loss = (
            swe_free.auxiliary_loss * self.config.amrm.swe_free_anchor_aux_loss_weight
        )
        diagnostics = {
            **replay_diagnostics,
            **reliability.diagnostics,
            **process.diagnostics,
            **adaptation_diagnostics,
            **forecast.diagnostics,
            "swe_free_anchor_evidence_valid": swe_free.evidence_valid,
            "swe_free_anchor_supervised_count": swe_free.supervised_count,
        }
        return ReSTOutput(
            prediction=forecast.prediction
            if self.training
            else forecast.prediction.clamp_min(0.0),
            raw_prediction=forecast.prediction,
            base=forecast.base,
            trend=forecast.trend,
            residual=forecast.residual,
            expert_forecast=forecast.expert_forecast,
            expert_base=forecast.expert_base,
            expert_trend=forecast.expert_trend,
            expert_residual=forecast.expert_residual,
            router_weights=forecast.router_weights,
            router_load_balance_loss=forecast.router_load_balance_loss,
            auxiliary_loss=auxiliary_loss,
            reliability=reliability,
            process=process,
            scale=scale,
            swe_free_anchor_mm=swe_free.anchor_mm,
            swe_free_anchor_used=anchor_used,
            swe_free_anchor_attention=swe_free.attention,
            effective_anchor_swe_mm=effective_anchor,
            decomposition_used=forecast.decomposition_used,
            diagnostics=diagnostics,
        )
