"""Adaptive Missingness Reliability Module (AMRM)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
from torch import nn

from .config import AMRMConfig, ObservedHistory, StructuredReplayConfig
from .reliability import (
    RELIABILITY_STATE_DIM,
    GapAwareSWEStateAnchor,
    HydroMissingnessStateEncoder,
    ReliabilityAwareGroupDependencyAdapter,
    ReliabilityAwarePatchAdapter,
    StructuredMissingnessReplay,
    SWEFreeCrossVariableAnchor,
)


@dataclass(frozen=True)
class ReliabilityEncoding:
    context: torch.Tensor
    state_features: torch.Tensor
    safe_current_swe_mm: torch.Tensor
    safe_current_swe_normalized: torch.Tensor
    effective_anchor_swe_mm: torch.Tensor
    has_any_swe_history: torch.Tensor
    issue_day_swe_valid: torch.Tensor
    last_valid_swe_age_days: torch.Tensor
    diagnostics: Mapping[str, Any]


@dataclass(frozen=True)
class SWEFreeAnchorOutput:
    anchor_mm: torch.Tensor
    attention: torch.Tensor
    evidence_valid: torch.Tensor
    auxiliary_loss: torch.Tensor
    supervised_count: torch.Tensor


class AdaptiveMissingnessReliabilityModule(nn.Module):
    """Observation-aware context and conditional SWE anchors (AMRM)."""

    def __init__(
        self,
        *,
        model_dim: int,
        static_dim: int,
        prior_dim: int,
        feature_names: Sequence[str],
        config: AMRMConfig,
        replay_config: StructuredReplayConfig,
        dropout: float,
    ) -> None:
        super().__init__()
        self.model_dim = int(model_dim)
        self.feature_names = tuple((str(name) for name in feature_names))
        self.config = config
        self.replay = StructuredMissingnessReplay(
            swe_tail_dropout_prob=replay_config.swe_tail_dropout_prob,
            feature_block_dropout_prob=replay_config.feature_block_dropout_prob,
            max_tail_days=replay_config.max_tail_days,
            max_feature_block_days=replay_config.max_feature_block_days,
        )
        self.anchor = GapAwareSWEStateAnchor(
            hidden_dim=config.gap_hidden_dim,
            max_gap_correction_mm=config.max_gap_correction_mm,
        )
        self.state_encoder = HydroMissingnessStateEncoder(
            model_dim=model_dim, hidden_dim=config.state_hidden_dim, dropout=dropout
        )
        self.patch_encoder = ReliabilityAwarePatchAdapter(
            model_dim=model_dim,
            state_dim=RELIABILITY_STATE_DIM,
            patch_bins=8,
            hidden_dim=config.state_hidden_dim,
            dropout=dropout,
        )
        self.process_adapter = ReliabilityAwareGroupDependencyAdapter(model_dim)
        self.swe_free_anchor = SWEFreeCrossVariableAnchor(
            model_dim=model_dim,
            static_dim=static_dim,
            prior_dim=prior_dim,
            hidden_dim=config.swe_free_anchor_hidden_dim,
            smooth_l1_beta_mm=config.swe_free_anchor_smooth_l1_beta_mm,
        )

    def replay_short(
        self, short: ObservedHistory
    ) -> tuple[ObservedHistory, Mapping[str, Any]]:
        normalized, observed, raw, diagnostics = self.replay(
            x_seq=short.normalized,
            x_mask=short.observed,
            x_seq_raw=short.raw,
            feature_names=self.feature_names,
        )
        if raw is None:
            raise AssertionError(
                "structured replay must preserve the raw short-history mirror"
            )
        return (ObservedHistory(normalized, raw, observed), diagnostics)

    def encode(
        self, *, short: ObservedHistory, long: ObservedHistory, analog: ObservedHistory
    ) -> ReliabilityEncoding:
        anchor = self.anchor(
            x_seq=short.normalized,
            x_mask=short.observed,
            x_seq_raw=short.raw,
            current_swe=None,
            current_swe_raw=None,
            current_swe_normalized=None,
            feature_names=self.feature_names,
        )
        context, features, state_diagnostics = self.state_encoder(
            x_seq=short.normalized,
            x_mask=short.observed,
            x_long=long.normalized,
            x_long_mask=long.observed,
            x_analog=analog.normalized,
            x_analog_mask=analog.observed,
            anchor=anchor,
            feature_names=self.feature_names,
        )
        diagnostics = {**anchor, **state_diagnostics}
        return ReliabilityEncoding(
            context=context,
            state_features=features,
            safe_current_swe_mm=anchor["safe_current_swe"],
            safe_current_swe_normalized=anchor["safe_current_swe_normalized"],
            effective_anchor_swe_mm=anchor["effective_anchor_swe"],
            has_any_swe_history=anchor["has_any_swe_history"].to(torch.bool),
            issue_day_swe_valid=anchor["issue_day_swe_valid"].to(torch.bool),
            last_valid_swe_age_days=anchor["last_valid_swe_age_days"],
            diagnostics=diagnostics,
        )

    def adapt_context(
        self,
        *,
        context: torch.Tensor,
        reliability: ReliabilityEncoding,
        short: ObservedHistory,
        group_tokens: torch.Tensor,
        group_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, Mapping[str, Any]]:
        patch, patch_diagnostics = self.patch_encoder(
            x_seq=short.normalized,
            x_mask=short.observed,
            state_features=reliability.state_features,
            feature_names=self.feature_names,
        )
        adapted = context + reliability.context + patch
        process_context, attention = self.process_adapter(
            group_tokens, group_valid, adapted
        )
        adapted = adapted + process_context
        return (
            adapted,
            {
                **patch_diagnostics,
                "reliability_state_context": reliability.context,
                "reliability_patch_context": patch,
                "reliability_process_context": process_context,
                "reliability_process_attention": attention,
            },
        )

    def infer_swe_free_anchor(
        self,
        *,
        group_tokens: torch.Tensor,
        group_valid: torch.Tensor,
        group_names: Sequence[str],
        static_topography: torch.Tensor,
        mechanism_prior: torch.Tensor,
        mechanism_prior_valid: torch.Tensor,
        reliability: ReliabilityEncoding,
    ) -> SWEFreeAnchorOutput:
        safe_prior = torch.where(
            mechanism_prior_valid.reshape(-1, 1),
            mechanism_prior,
            torch.full_like(mechanism_prior, float("nan")),
        )
        output = self.swe_free_anchor(
            group_tokens=group_tokens,
            group_valid=group_valid,
            group_names=tuple(group_names),
            x_static=static_topography,
            mechanism_prior=safe_prior,
            current_swe_raw=reliability.safe_current_swe_mm,
            issue_day_swe_valid=reliability.issue_day_swe_valid,
        )
        auxiliary_loss = output["swe_free_anchor_aux_loss"]
        if not self.training:
            auxiliary_loss = auxiliary_loss.new_zeros(())
        return SWEFreeAnchorOutput(
            anchor_mm=output["swe_free_anchor_mm"],
            attention=output["swe_free_anchor_attention"],
            evidence_valid=output["swe_free_anchor_evidence_valid"],
            auxiliary_loss=auxiliary_loss,
            supervised_count=output["swe_free_anchor_supervised_count"],
        )
