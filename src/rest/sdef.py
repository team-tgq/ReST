"""Scale-consistent Decomposed Expert Forecasting (SDEF)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn.functional as Fnn
from torch import nn

from .amrm import ReliabilityEncoding, SWEFreeAnchorOutput
from .config import ObservedHistory
from .mpe import CrossDomainProcessEncoding
from .reliability import RELIABILITY_STATE_DIM

_NETWORK_INPUT_LIMIT = 1000000.0


@dataclass(frozen=True)
class WindowScaleState:
    short: ObservedHistory
    long: ObservedHistory
    analog: ObservedHistory
    mean: torch.Tensor
    std: torch.Tensor
    count: torch.Tensor
    used: bool


@dataclass(frozen=True)
class DecomposedForecast:
    prediction: torch.Tensor
    base: torch.Tensor
    trend: torch.Tensor
    residual: torch.Tensor
    expert_forecast: torch.Tensor
    expert_base: torch.Tensor
    expert_trend: torch.Tensor
    expert_residual: torch.Tensor
    router_weights: torch.Tensor
    router_load_balance_loss: torch.Tensor
    decomposition_used: torch.Tensor
    diagnostics: Mapping[str, torch.Tensor]


def _masked_standardize(
    history: ObservedHistory, *, eps: float
) -> tuple[ObservedHistory, torch.Tensor, torch.Tensor, torch.Tensor]:
    valid = history.observed.to(torch.bool) & torch.isfinite(history.raw)
    clean = torch.where(
        valid,
        torch.nan_to_num(history.raw, nan=0.0, posinf=0.0, neginf=0.0),
        torch.zeros_like(history.raw),
    )
    weights = valid.to(clean.dtype)
    count = weights.sum(dim=1)
    mean_raw = clean.sum(dim=1) / count.clamp_min(1.0)
    centered = torch.where(
        valid, clean - mean_raw.unsqueeze(1), torch.zeros_like(clean)
    )
    std_raw = torch.sqrt(centered.square().sum(dim=1) / count.clamp_min(1.0))
    usable = (
        (count >= 2)
        & torch.isfinite(mean_raw)
        & torch.isfinite(std_raw)
        & (std_raw > eps)
    )
    mean = torch.where(usable, mean_raw, torch.zeros_like(mean_raw)).detach()
    std = torch.where(usable, std_raw, torch.ones_like(std_raw)).detach()
    normalized = torch.where(
        valid,
        ((clean - mean.unsqueeze(1)) / std.unsqueeze(1)).clamp(
            -_NETWORK_INPUT_LIMIT, _NETWORK_INPUT_LIMIT
        ),
        torch.zeros_like(clean),
    )
    return (
        ObservedHistory(normalized, history.raw, history.observed),
        mean,
        std,
        count.detach(),
    )


class _ProcessRouter(nn.Module):
    def __init__(
        self,
        *,
        context_dim: int,
        prior_dim: int,
        diagnostics_dim: int,
        hidden_dim: int,
        num_experts: int,
        prior_strength: float,
        top_k: int,
    ) -> None:
        super().__init__()
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.prior_strength = float(prior_strength)
        self.net = nn.Sequential(
            nn.Linear(context_dim + prior_dim + diagnostics_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_experts),
        )
        self.prior_to_logits = nn.Linear(prior_dim, num_experts, bias=False)
        self.diagnostics_to_logits = nn.Linear(diagnostics_dim, num_experts, bias=False)

    def forward(
        self, context: torch.Tensor, prior: torch.Tensor, diagnostics: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        logits = self.net(torch.cat([context, prior, diagnostics], dim=-1))
        logits = logits + self.prior_strength * self.prior_to_logits(prior)
        logits = logits + self.diagnostics_to_logits(diagnostics)
        dense = torch.softmax(logits, dim=-1)
        top_logits, top_indices = torch.topk(logits, k=self.top_k, dim=-1)
        top_weights = torch.softmax(top_logits, dim=-1)
        weights = torch.zeros_like(logits).scatter(
            dim=-1, index=top_indices, src=top_weights
        )
        load = (weights > 0).to(weights.dtype).mean(dim=0)
        importance = dense.mean(dim=0)
        balance = self.num_experts * torch.sum(load * importance)
        return (
            weights,
            {
                "router_logits": logits,
                "router_dense_probs": dense,
                "router_top_indices": top_indices,
                "router_top_weights": top_weights,
                "router_load_balance_loss": balance,
            },
        )


class _ScaleAwareExpert(nn.Module):
    def __init__(
        self, *, model_dim: int, hidden_dim: int, horizon: int, dropout: float
    ) -> None:
        super().__init__()
        self.trend_head = nn.Sequential(
            nn.Linear(model_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, horizon),
        )
        self.residual_head = nn.Sequential(
            nn.Linear(model_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, horizon),
        )

    def forward(
        self,
        context: torch.Tensor,
        base: torch.Tensor,
        scale_tokens: torch.Tensor,
        scale_valid: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        tokens = torch.where(
            scale_valid.unsqueeze(-1), scale_tokens, torch.zeros_like(scale_tokens)
        )
        trend = self.trend_head(
            torch.cat([context, tokens[:, 1], tokens[:, 2]], dim=-1)
        )
        residual = self.residual_head(torch.cat([context, tokens[:, 0]], dim=-1))
        expanded_base = base.view(-1, 1).expand_as(trend)
        return {
            "forecast": expanded_base + trend + residual,
            "base": expanded_base,
            "trend": trend,
            "residual": residual,
        }


class _ProcessExperts(nn.Module):
    def __init__(
        self,
        *,
        model_dim: int,
        hidden_dim: int,
        horizon: int,
        num_experts: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.experts = nn.ModuleList(
            [
                _ScaleAwareExpert(
                    model_dim=model_dim,
                    hidden_dim=hidden_dim,
                    horizon=horizon,
                    dropout=dropout,
                )
                for _ in range(num_experts)
            ]
        )

    def forward(
        self,
        context: torch.Tensor,
        base: torch.Tensor,
        scale_tokens: torch.Tensor,
        scale_valid: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        outputs = [
            expert(context, base, scale_tokens, scale_valid) for expert in self.experts
        ]
        return {
            name: torch.stack([output[name] for output in outputs], dim=1)
            for name in ("forecast", "base", "trend", "residual")
        }


class ScaleConsistentDecomposedExpertForecasting(nn.Module):
    """SDEF: scale control, conditional routing, experts and raw-mm BTR recovery."""

    def __init__(
        self,
        *,
        seq_dim: int,
        model_dim: int,
        prior_dim: int,
        horizon: int,
        hidden_dim: int,
        num_experts: int,
        router_top_k: int,
        router_prior_strength: float,
        dropout: float,
        window_scale_eps: float,
    ) -> None:
        super().__init__()
        self.seq_dim = int(seq_dim)
        self.model_dim = int(model_dim)
        self.horizon = int(horizon)
        self.num_experts = int(num_experts)
        self.window_scale_eps = float(window_scale_eps)
        diagnostics_dim = 5 + RELIABILITY_STATE_DIM
        self.router = _ProcessRouter(
            context_dim=model_dim,
            prior_dim=prior_dim,
            diagnostics_dim=diagnostics_dim,
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            prior_strength=router_prior_strength,
            top_k=router_top_k,
        )
        self.experts = _ProcessExperts(
            model_dim=model_dim,
            hidden_dim=hidden_dim,
            horizon=horizon,
            num_experts=num_experts,
            dropout=dropout,
        )
        self.learned_base_head = nn.Sequential(
            nn.Linear(model_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def prepare_inputs(
        self, short: ObservedHistory, long: ObservedHistory, analog: ObservedHistory
    ) -> WindowScaleState:
        scaled = [
            _masked_standardize(short, eps=self.window_scale_eps),
            _masked_standardize(long, eps=self.window_scale_eps),
            _masked_standardize(analog, eps=self.window_scale_eps),
        ]
        return WindowScaleState(
            short=scaled[0][0],
            long=scaled[1][0],
            analog=scaled[2][0],
            mean=torch.stack([value[1] for value in scaled], dim=1),
            std=torch.stack([value[2] for value in scaled], dim=1),
            count=torch.stack([value[3] for value in scaled], dim=1),
            used=True,
        )

    def forecast(
        self,
        *,
        context: torch.Tensor,
        process: CrossDomainProcessEncoding,
        reliability: ReliabilityEncoding,
        swe_free: SWEFreeAnchorOutput,
        scale: WindowScaleState,
        swe_index: int,
    ) -> DecomposedForecast:
        context.shape[0]
        if (
            self.router is None
            or self.experts is None
            or self.learned_base_head is None
        ):
            raise AssertionError("decomposed components are not initialized")
        router_features = torch.cat(
            [process.process_features, reliability.state_features], dim=-1
        )
        router_weights, router_diagnostics = self.router(
            context, process.effective_prior, router_features
        )
        swe_mean = scale.mean[:, 0, swe_index]
        swe_std = scale.std[:, 0, swe_index]
        normalized_anchor = torch.where(
            reliability.has_any_swe_history,
            (reliability.effective_anchor_swe_mm - swe_mean) / swe_std,
            torch.zeros_like(swe_mean),
        )
        learned_base = Fnn.softplus(self.learned_base_head(context).reshape(-1))
        expert_normalized = self.experts(
            context,
            torch.where(
                reliability.has_any_swe_history, normalized_anchor, learned_base
            ),
            process.scale_tokens,
            process.scale_valid,
        )
        mean = swe_mean[:, None, None]
        std = swe_std[:, None, None]
        expert_base = (mean + std * expert_normalized["base"]).clamp_min(0.0)
        expert_trend = std * expert_normalized["trend"]
        expert_residual = std * expert_normalized["residual"]
        anchor_used = ~reliability.has_any_swe_history & swe_free.evidence_valid
        expert_base = torch.where(
            anchor_used[:, None, None],
            swe_free.anchor_mm[:, None, None].expand_as(expert_base),
            expert_base,
        )
        expert_forecast = expert_base + expert_trend + expert_residual
        weights = router_weights.unsqueeze(-1)
        base = torch.sum(expert_base * weights, dim=1)
        trend = torch.sum(expert_trend * weights, dim=1)
        residual = torch.sum(expert_residual * weights, dim=1)
        prediction = base + trend + residual
        return DecomposedForecast(
            prediction=prediction,
            base=base,
            trend=trend,
            residual=residual,
            expert_forecast=expert_forecast,
            expert_base=expert_base,
            expert_trend=expert_trend,
            expert_residual=expert_residual,
            router_weights=router_weights,
            router_load_balance_loss=router_diagnostics["router_load_balance_loss"],
            decomposition_used=torch.tensor(True, device=context.device),
            diagnostics={
                **router_diagnostics,
                "window_scale_control_used": torch.tensor(True, device=context.device),
                "window_scale_mean": scale.mean,
                "window_scale_std": scale.std,
                "window_scale_valid_count": scale.count,
                "process_diagnostics": router_features,
            },
        )
