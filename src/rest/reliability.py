"""Observation reliability, gap correction, SWE-free anchors, and training augmentation."""

from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn.functional as Fnn
from torch import nn

from .observations import SWE_FEATURE_NAMES, SWE_VALUE_FEATURE_NAMES, get_last_valid_swe

RELIABILITY_STATE_NAMES = (
    "issue_day_swe_valid",
    "has_any_swe_history",
    "swe_history_valid_ratio",
    "last_valid_swe_age_ratio",
    "trailing_missing_ratio",
    "longest_missing_run_ratio",
    "short_valid_ratio",
    "long_valid_ratio",
    "analog_valid_ratio",
    "short_swe_valid_ratio",
    "long_swe_valid_ratio",
    "analog_swe_valid_ratio",
    "forcing_valid_ratio",
    "microwave_valid_ratio",
    "short_mask_transition_ratio",
    "short_recent_swe_valid_ratio",
)
RELIABILITY_STATE_DIM = len(RELIABILITY_STATE_NAMES)


def _lower_names(feature_names: Iterable[str] | None) -> list[str]:
    return [str(name).lower() for name in feature_names or []]


def _indices(
    feature_names: Iterable[str] | None, candidates: Iterable[str], feature_dim: int
) -> list[int]:
    names = _lower_names(feature_names)[: int(feature_dim)]
    candidate_set = {str(name).lower() for name in candidates}
    return [index for index, name in enumerate(names) if name in candidate_set]


def resolve_missingness_feature_groups(
    feature_names: Iterable[str] | None, feature_dim: int
) -> dict[str, list[int]]:
    dimension = int(feature_dim)
    swe_value = _indices(feature_names, SWE_VALUE_FEATURE_NAMES, dimension) or (
        [0] if dimension > 0 else []
    )
    swe_indicator = _indices(
        feature_names, SWE_FEATURE_NAMES.difference(SWE_VALUE_FEATURE_NAMES), dimension
    )
    swe_all = sorted(set(swe_value).union(swe_indicator))
    time_features = _indices(feature_names, {"day_sin", "day_cos"}, dimension)
    microwave = _indices(feature_names, {"tb_19ve", "tb_37ve", "tb_diff"}, dimension)
    thermal = _indices(feature_names, {"tmax", "tmin"}, dimension)
    precipitation = _indices(feature_names, {"precipitation"}, dimension)
    radiation = _indices(feature_names, {"shortwave_radiation"}, dimension)
    turbulent_moisture = _indices(
        feature_names,
        {
            "wind_velocity",
            "relative_humidity_max",
            "relative_humidity_min",
            "specific_humidity",
        },
        dimension,
    )
    protected = set(swe_all).union(time_features)
    sensor = [index for index in range(dimension) if index not in protected]
    return {
        "swe_value": swe_value,
        "swe_indicator": swe_indicator,
        "swe_all": swe_all,
        "time": time_features,
        "microwave": microwave,
        "thermal": thermal,
        "precipitation": precipitation,
        "radiation": radiation,
        "turbulent_moisture": turbulent_moisture,
        "forcing": sensor,
        "sensor": sensor,
    }


def _valid_mask(
    values: torch.Tensor | None,
    mask: torch.Tensor | None,
    batch_size: int,
    time_steps: int,
    feature_dim: int,
    device: torch.device,
) -> torch.Tensor:
    if values is None:
        return torch.zeros(
            batch_size, time_steps, feature_dim, dtype=torch.bool, device=device
        )
    finite = torch.isfinite(values)
    if mask is None:
        return finite
    return mask.to(device=device, dtype=torch.bool) & finite


def _ratio(valid: torch.Tensor, indices: list[int] | None = None) -> torch.Tensor:
    selected = valid if indices is None else valid[..., indices]
    if selected.shape[-1] == 0:
        return valid.new_zeros(valid.shape[0], dtype=torch.float32)
    return selected.to(dtype=torch.float32).reshape(selected.shape[0], -1).mean(dim=-1)


def _masked_ratio_for_scale(
    values: torch.Tensor | None,
    mask: torch.Tensor | None,
    indices: list[int] | None,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    if values is None:
        return torch.zeros(batch_size, dtype=torch.float32, device=device)
    valid = _valid_mask(
        values,
        mask,
        int(values.shape[0]),
        int(values.shape[1]),
        int(values.shape[2]),
        device,
    )
    return _ratio(valid, indices)


def _longest_false_run(valid: torch.Tensor) -> torch.Tensor:
    if valid.shape[1] == 0:
        return torch.zeros(valid.shape[0], dtype=torch.float32, device=valid.device)
    run = torch.zeros(valid.shape[0], dtype=torch.float32, device=valid.device)
    longest = torch.zeros_like(run)
    for step in range(int(valid.shape[1])):
        run = torch.where(valid[:, step], torch.zeros_like(run), run + 1.0)
        longest = torch.maximum(longest, run)
    return longest


class GapAwareSWEStateAnchor(nn.Module):
    def __init__(
        self, hidden_dim: int = 32, max_gap_correction_mm: float = 100.0
    ) -> None:
        super().__init__()
        self.max_gap_correction_mm = float(max_gap_correction_mm)
        self.gap_projection = nn.Sequential(
            nn.Linear(7, int(hidden_dim)), nn.GELU(), nn.Linear(int(hidden_dim), 1)
        )
        final = self.gap_projection[-1]
        if isinstance(final, nn.Linear):
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)

    @staticmethod
    def _tail_mean(
        values: torch.Tensor,
        valid: torch.Tensor,
        tail: torch.Tensor,
        indices: list[int],
    ) -> torch.Tensor:
        if not indices:
            return values.new_zeros(values.shape[0])
        selected = values[..., indices]
        selected_valid = valid[..., indices] & tail.unsqueeze(-1)
        clean = torch.where(
            selected_valid,
            torch.nan_to_num(selected, nan=0.0, posinf=0.0, neginf=0.0),
            torch.zeros_like(selected),
        )
        count = (
            selected_valid.to(dtype=clean.dtype).reshape(clean.shape[0], -1).sum(dim=-1)
        )
        return clean.reshape(clean.shape[0], -1).sum(dim=-1) / count.clamp_min(1.0)

    def forward(
        self,
        x_seq: torch.Tensor,
        x_mask: torch.Tensor | None,
        x_seq_raw: torch.Tensor | None,
        current_swe: torch.Tensor | None,
        current_swe_raw: torch.Tensor | None,
        current_swe_normalized: torch.Tensor | None,
        feature_names: list[str] | tuple[str, ...] | None,
    ) -> dict[str, torch.Tensor]:
        batch_size, time_steps, feature_dim = map(int, x_seq.shape)
        device, dtype = (x_seq.device, x_seq.dtype)
        normalized_history = get_last_valid_swe(x_seq, x_mask, feature_names)
        raw_sequence = x_seq_raw if x_seq_raw is not None else x_seq
        raw_history = get_last_valid_swe(raw_sequence, x_mask, feature_names)
        normalized_last = normalized_history["swe_history_last_valid_index"]
        raw_last = raw_history["swe_history_last_valid_index"]
        has_any = normalized_last >= 0
        if x_seq_raw is not None:
            has_any = has_any & (raw_last >= 0)
        issue_day_valid = has_any & (normalized_last == time_steps - 1)
        if x_seq_raw is not None:
            issue_day_valid = issue_day_valid & (raw_last == time_steps - 1)
        supplied_normalized = (
            current_swe_normalized
            if current_swe_normalized is not None
            else current_swe
            if current_swe_raw is None
            else None
        )
        supplied_raw = (
            current_swe_raw
            if current_swe_raw is not None
            else current_swe
            if x_seq_raw is None
            else None
        )

        def _finite_supplied(
            value: torch.Tensor | None, fallback: torch.Tensor
        ) -> torch.Tensor:
            if value is None:
                return fallback
            reshaped = value.reshape(-1).to(device=device, dtype=fallback.dtype)
            finite = torch.isfinite(reshaped)
            return torch.where(issue_day_valid & finite, reshaped, fallback)

        safe_normalized = torch.where(
            has_any,
            _finite_supplied(
                supplied_normalized, normalized_history["safe_current_swe"]
            ),
            torch.zeros_like(normalized_history["safe_current_swe"]),
        )
        safe_raw = torch.where(
            has_any,
            _finite_supplied(supplied_raw, raw_history["safe_current_swe"]),
            torch.zeros_like(raw_history["safe_current_swe"]),
        )
        feature_groups = resolve_missingness_feature_groups(feature_names, feature_dim)
        swe_idx = feature_groups["swe_value"]
        normalized_valid = _valid_mask(
            x_seq, x_mask, batch_size, time_steps, feature_dim, device
        )
        swe_valid = normalized_valid[..., swe_idx].any(dim=-1)
        positions = torch.arange(time_steps, device=device, dtype=torch.long).reshape(
            1, -1
        )
        tail = positions > normalized_last.reshape(-1, 1)
        age = torch.where(
            has_any,
            (time_steps - 1 - normalized_last).clamp_min(0),
            torch.full_like(normalized_last, time_steps),
        )
        trailing = age.to(dtype=torch.float32)
        longest = _longest_false_run(swe_valid)
        forcing_indices = feature_groups["forcing"]
        precipitation_indices = feature_groups["precipitation"]
        thermal_indices = feature_groups["thermal"]
        radiation_indices = feature_groups["radiation"]
        microwave_indices = feature_groups["microwave"]
        forcing_values = x_seq
        forcing_valid = normalized_valid
        forcing_mean = self._tail_mean(
            forcing_values, forcing_valid, tail, forcing_indices
        )
        precipitation_mean = self._tail_mean(
            forcing_values, forcing_valid, tail, precipitation_indices
        )
        thermal_mean = self._tail_mean(
            forcing_values, forcing_valid, tail, thermal_indices
        )
        radiation_mean = self._tail_mean(
            forcing_values, forcing_valid, tail, radiation_indices
        )
        microwave_mean = self._tail_mean(
            forcing_values, forcing_valid, tail, microwave_indices
        )
        gap_active = has_any & (age > 0)
        tail_forcing_ratio = _ratio(forcing_valid & tail.unsqueeze(-1), forcing_indices)
        gap_features = torch.stack(
            [
                age.to(dtype=dtype) / max(1.0, float(time_steps)),
                trailing.to(dtype=dtype) / max(1.0, float(time_steps)),
                precipitation_mean.to(dtype=dtype),
                thermal_mean.to(dtype=dtype),
                radiation_mean.to(dtype=dtype),
                microwave_mean.to(dtype=dtype),
                tail_forcing_ratio.to(dtype=dtype),
            ],
            dim=-1,
        )
        gap_features = torch.nan_to_num(gap_features, nan=0.0, posinf=0.0, neginf=0.0)
        gap_delta = (
            torch.tanh(self.gap_projection(gap_features).reshape(-1))
            * self.max_gap_correction_mm
        )
        gap_delta = torch.where(gap_active, gap_delta, torch.zeros_like(gap_delta))
        effective_anchor = torch.relu(safe_raw + gap_delta)
        effective_anchor = torch.where(
            has_any, effective_anchor, torch.zeros_like(effective_anchor)
        )
        return {
            "safe_current_swe": torch.nan_to_num(
                safe_raw, nan=0.0, posinf=0.0, neginf=0.0
            ),
            "safe_current_swe_normalized": torch.nan_to_num(
                safe_normalized, nan=0.0, posinf=0.0, neginf=0.0
            ),
            "effective_anchor_swe": torch.nan_to_num(
                effective_anchor, nan=0.0, posinf=0.0, neginf=0.0
            ),
            "gap_forcing_delta_mm": torch.nan_to_num(
                gap_delta, nan=0.0, posinf=0.0, neginf=0.0
            ),
            "current_swe_valid": has_any,
            "has_any_swe_history": has_any,
            "issue_day_swe_valid": issue_day_valid,
            "swe_history_valid_ratio": swe_valid.to(dtype=dtype).mean(dim=-1),
            "swe_history_last_valid_index": normalized_last,
            "last_valid_swe_age_days": age.to(dtype=dtype),
            "trailing_missing_length": trailing,
            "longest_missing_run": longest.to(dtype=dtype),
            "gap_tail_forcing_ratio": tail_forcing_ratio.to(dtype=dtype),
            "gap_tail_has_swe_gap": gap_active.to(dtype=dtype),
            "gap_tail_forcing_mean": forcing_mean.to(dtype=dtype),
        }


class HydroMissingnessStateEncoder(nn.Module):
    feature_names = RELIABILITY_STATE_NAMES

    def __init__(
        self, model_dim: int, hidden_dim: int = 64, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(RELIABILITY_STATE_DIM, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(model_dim)),
            nn.LayerNorm(int(model_dim)),
        )

    def forward(
        self,
        x_seq: torch.Tensor,
        x_mask: torch.Tensor | None,
        x_long: torch.Tensor | None,
        x_long_mask: torch.Tensor | None,
        x_analog: torch.Tensor | None,
        x_analog_mask: torch.Tensor | None,
        anchor: dict[str, torch.Tensor],
        feature_names: list[str] | tuple[str, ...] | None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        batch_size, time_steps, feature_dim = map(int, x_seq.shape)
        device, dtype = (x_seq.device, x_seq.dtype)
        short_valid = _valid_mask(
            x_seq, x_mask, batch_size, time_steps, feature_dim, device
        )
        feature_groups = resolve_missingness_feature_groups(feature_names, feature_dim)
        swe_indices = feature_groups["swe_value"]
        forcing_indices = feature_groups["forcing"]
        microwave_indices = feature_groups["microwave"]
        swe_valid = short_valid[..., swe_indices]
        transition = torch.zeros(batch_size, device=device, dtype=torch.float32)
        if time_steps > 1:
            transition = (
                swe_valid.to(dtype=torch.float32).any(dim=-1).to(dtype=torch.float32)
            )
            transition = (transition[:, 1:] - transition[:, :-1]).abs().mean(dim=-1)
        recent_steps = min(5, time_steps)
        recent_swe_ratio = (
            swe_valid[:, -recent_steps:].to(dtype=torch.float32).mean(dim=(1, 2))
            if recent_steps > 0
            else torch.zeros(batch_size, device=device)
        )
        short_ratio = _ratio(short_valid)
        long_ratio = _masked_ratio_for_scale(
            x_long, x_long_mask, None, batch_size, device
        )
        analog_ratio = _masked_ratio_for_scale(
            x_analog, x_analog_mask, None, batch_size, device
        )
        short_swe_ratio = _ratio(short_valid, swe_indices)
        long_swe_ratio = _masked_ratio_for_scale(
            x_long, x_long_mask, swe_indices, batch_size, device
        )
        analog_swe_ratio = _masked_ratio_for_scale(
            x_analog, x_analog_mask, swe_indices, batch_size, device
        )
        forcing_ratio = _ratio(short_valid, forcing_indices)
        microwave_ratio = _ratio(short_valid, microwave_indices)
        scale = max(1.0, float(time_steps))
        features = torch.stack(
            [
                anchor["issue_day_swe_valid"].to(dtype=torch.float32),
                anchor["has_any_swe_history"].to(dtype=torch.float32),
                anchor["swe_history_valid_ratio"].to(dtype=torch.float32),
                anchor["last_valid_swe_age_days"].to(dtype=torch.float32) / scale,
                anchor["trailing_missing_length"].to(dtype=torch.float32) / scale,
                anchor["longest_missing_run"].to(dtype=torch.float32) / scale,
                short_ratio,
                long_ratio,
                analog_ratio,
                short_swe_ratio,
                long_swe_ratio,
                analog_swe_ratio,
                forcing_ratio,
                microwave_ratio,
                transition,
                recent_swe_ratio,
            ],
            dim=-1,
        ).to(device=device, dtype=dtype)
        features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        context = self.projection(features)
        diagnostics = {
            name: features[:, index]
            for index, name in enumerate(RELIABILITY_STATE_NAMES)
        }
        diagnostics["reliability_state_used"] = torch.tensor(
            True, dtype=torch.bool, device=device
        )
        return (context, features, diagnostics)


class ReliabilityAwarePatchAdapter(nn.Module):
    def __init__(
        self,
        model_dim: int,
        state_dim: int = RELIABILITY_STATE_DIM,
        patch_bins: int = 8,
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.patch_bins = max(1, int(patch_bins))
        input_dim = int(state_dim) + 3 * self.patch_bins
        self.projection = nn.Sequential(
            nn.Linear(input_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(model_dim)),
            nn.LayerNorm(int(model_dim)),
        )

    def forward(
        self,
        x_seq: torch.Tensor,
        x_mask: torch.Tensor | None,
        state_features: torch.Tensor,
        feature_names: list[str] | tuple[str, ...] | None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch_size, time_steps, feature_dim = map(int, x_seq.shape)
        valid = _valid_mask(
            x_seq, x_mask, batch_size, time_steps, feature_dim, x_seq.device
        )
        overall = valid.to(dtype=x_seq.dtype).mean(dim=-1)
        swe_indices = _indices(feature_names, SWE_VALUE_FEATURE_NAMES, feature_dim) or [
            0
        ]
        swe = valid[..., swe_indices].to(dtype=x_seq.dtype).mean(dim=-1)
        transition = torch.zeros_like(swe)
        if time_steps > 1:
            transition = (swe[:, 1:] - swe[:, :-1]).abs()
            transition = Fnn.pad(transition, (1, 0), value=0.0)
        patch_features = torch.cat(
            [
                Fnn.adaptive_avg_pool1d(overall.unsqueeze(1), self.patch_bins).squeeze(
                    1
                ),
                Fnn.adaptive_avg_pool1d(swe.unsqueeze(1), self.patch_bins).squeeze(1),
                Fnn.adaptive_avg_pool1d(
                    transition.unsqueeze(1), self.patch_bins
                ).squeeze(1),
            ],
            dim=-1,
        )
        adapter_input = torch.cat(
            [state_features.to(dtype=x_seq.dtype), patch_features], dim=-1
        )
        context = self.projection(
            torch.nan_to_num(adapter_input, nan=0.0, posinf=0.0, neginf=0.0)
        )
        diagnostics = {
            "short_patch_valid_ratio": patch_features[:, : self.patch_bins],
            "short_patch_swe_valid_ratio": patch_features[
                :, self.patch_bins : 2 * self.patch_bins
            ],
            "short_patch_mask_transition": patch_features[:, 2 * self.patch_bins :],
            "reliability_patch_context": context,
        }
        return (context, diagnostics)


class ReliabilityAwareGroupDependencyAdapter(nn.Module):
    def __init__(self, model_dim: int) -> None:
        super().__init__()
        self.query_projection = nn.Linear(int(model_dim), int(model_dim), bias=False)
        self.key_projection = nn.Linear(int(model_dim), int(model_dim), bias=False)
        self.value_projection = nn.Linear(int(model_dim), int(model_dim), bias=False)
        self.output_projection = nn.Sequential(
            nn.Linear(int(model_dim), int(model_dim)), nn.LayerNorm(int(model_dim))
        )

    def forward(
        self,
        group_tokens: torch.Tensor,
        group_valid: torch.Tensor,
        query_context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if group_tokens.ndim != 4 or group_valid.ndim != 3:
            raise ValueError(
                "group_tokens must be [B,scales,groups,D] and group_valid must be [B,scales,groups]."
            )
        batch_size, scale_count, group_count, model_dim = map(int, group_tokens.shape)
        tokens = group_tokens.reshape(batch_size, scale_count * group_count, model_dim)
        valid = group_valid.reshape(batch_size, scale_count * group_count).to(
            dtype=torch.bool
        )
        any_valid = valid.any(dim=-1)
        safe_valid = valid.clone()
        if safe_valid.shape[-1] > 0:
            safe_valid[~any_valid, 0] = True
        query = self.query_projection(query_context).unsqueeze(1)
        keys = self.key_projection(
            torch.nan_to_num(tokens, nan=0.0, posinf=0.0, neginf=0.0)
        )
        values = self.value_projection(
            torch.nan_to_num(tokens, nan=0.0, posinf=0.0, neginf=0.0)
        )
        scores = torch.sum(query * keys, dim=-1) / math.sqrt(max(1, model_dim))
        scores = scores.masked_fill(~safe_valid, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        weights = torch.where(valid, weights, torch.zeros_like(weights))
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        pooled = torch.sum(values * weights.unsqueeze(-1), dim=1)
        output = self.output_projection(pooled)
        output = torch.where(any_valid.unsqueeze(-1), output, torch.zeros_like(output))
        weights = torch.where(
            any_valid.unsqueeze(-1), weights, torch.zeros_like(weights)
        )
        return (output, weights)


class SWEFreeCrossVariableAnchor(nn.Module):
    def __init__(
        self,
        model_dim: int,
        static_dim: int,
        prior_dim: int = 5,
        hidden_dim: int = 64,
        smooth_l1_beta_mm: float = 50.0,
    ) -> None:
        super().__init__()
        self.model_dim = int(model_dim)
        self.smooth_l1_beta_mm = float(smooth_l1_beta_mm)
        if self.smooth_l1_beta_mm <= 0.0:
            raise ValueError("smooth_l1_beta_mm must be positive")
        self.query = nn.Parameter(torch.empty(self.model_dim))
        nn.init.normal_(self.query, mean=0.0, std=0.02)
        self.key_projection = nn.Linear(self.model_dim, self.model_dim, bias=False)
        self.value_projection = nn.Linear(self.model_dim, self.model_dim, bias=False)
        self.static_projection = nn.Sequential(
            nn.Linear(int(static_dim), self.model_dim),
            nn.GELU(),
            nn.LayerNorm(self.model_dim),
        )
        self.prior_projection = nn.Sequential(
            nn.Linear(int(prior_dim), self.model_dim),
            nn.GELU(),
            nn.LayerNorm(self.model_dim),
        )
        self.anchor_head = nn.Sequential(
            nn.Linear(self.model_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), 1),
        )

    def forward(
        self,
        *,
        group_tokens: torch.Tensor,
        group_valid: torch.Tensor,
        group_names: list[str] | tuple[str, ...],
        x_static: torch.Tensor,
        mechanism_prior: torch.Tensor,
        current_swe_raw: torch.Tensor | None,
        issue_day_swe_valid: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if group_tokens.ndim != 4 or group_valid.ndim != 3:
            raise ValueError(
                "group_tokens/group_valid must be [B,scales,groups,D]/[B,scales,groups]"
            )
        batch_size, scale_count, group_count, model_dim = map(int, group_tokens.shape)
        if model_dim != self.model_dim or len(group_names) != group_count:
            raise ValueError(
                "group token dimensions or group_names do not match the anchor configuration"
            )
        non_swe_indices = [
            index for index, name in enumerate(group_names) if str(name) != "snow_state"
        ]
        if len(non_swe_indices) != group_count - 1 or "snow_state" not in group_names:
            raise ValueError("group_names must contain exactly one snow_state group")
        process_tokens = group_tokens[:, :, non_swe_indices, :].reshape(
            batch_size, scale_count * len(non_swe_indices), model_dim
        )
        process_valid = (
            group_valid[:, :, non_swe_indices]
            .reshape(batch_size, scale_count * len(non_swe_indices))
            .to(dtype=torch.bool)
        )
        static_valid = torch.isfinite(x_static).all(dim=-1, keepdim=True)
        prior_valid = torch.isfinite(mechanism_prior).all(dim=-1, keepdim=True)
        static_token = self.static_projection(
            torch.nan_to_num(x_static, nan=0.0, posinf=0.0, neginf=0.0)
        ).unsqueeze(1)
        prior_token = self.prior_projection(
            torch.nan_to_num(mechanism_prior, nan=0.0, posinf=0.0, neginf=0.0)
        ).unsqueeze(1)
        tokens = torch.cat([process_tokens, static_token, prior_token], dim=1)
        valid = torch.cat([process_valid, static_valid, prior_valid], dim=1)
        any_valid = valid.any(dim=-1)
        safe_valid = valid.clone()
        safe_valid[~any_valid, 0] = True
        clean_tokens = torch.nan_to_num(tokens, nan=0.0, posinf=0.0, neginf=0.0)
        keys = self.key_projection(clean_tokens)
        values = self.value_projection(clean_tokens)
        scores = torch.sum(keys * self.query.reshape(1, 1, -1), dim=-1) / math.sqrt(
            self.model_dim
        )
        scores = scores.masked_fill(~safe_valid, torch.finfo(scores.dtype).min)
        attention = torch.softmax(scores, dim=-1)
        attention = torch.where(valid, attention, torch.zeros_like(attention))
        attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        pooled = torch.sum(values * attention.unsqueeze(-1), dim=1)
        pooled = torch.where(any_valid.unsqueeze(-1), pooled, torch.zeros_like(pooled))
        anchor_mm = Fnn.softplus(self.anchor_head(pooled).reshape(-1))
        anchor_mm = torch.nan_to_num(anchor_mm, nan=0.0, posinf=1000000.0, neginf=0.0)
        if current_swe_raw is None:
            target = anchor_mm.new_zeros(batch_size)
            supervised = torch.zeros(
                batch_size, dtype=torch.bool, device=anchor_mm.device
            )
        else:
            target = current_swe_raw.reshape(-1).to(
                device=anchor_mm.device, dtype=anchor_mm.dtype
            )
            supervised = issue_day_swe_valid.reshape(-1).to(
                device=anchor_mm.device, dtype=torch.bool
            )
            supervised = supervised & torch.isfinite(target)
        clean_target = torch.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
        per_sample = Fnn.smooth_l1_loss(
            anchor_mm, clean_target, reduction="none", beta=self.smooth_l1_beta_mm
        )
        supervised_float = supervised.to(dtype=anchor_mm.dtype)
        supervised_count = supervised_float.sum()
        auxiliary_loss = torch.sum(
            per_sample * supervised_float
        ) / supervised_count.clamp_min(1.0)
        return {
            "swe_free_anchor_mm": anchor_mm,
            "swe_free_anchor_attention": attention,
            "swe_free_anchor_evidence_valid": any_valid,
            "swe_free_anchor_aux_loss": auxiliary_loss,
            "swe_free_anchor_supervised_count": supervised_count,
        }


class StructuredMissingnessReplay(nn.Module):
    def __init__(
        self,
        swe_tail_dropout_prob: float = 0.0,
        feature_block_dropout_prob: float = 0.0,
        max_tail_days: int = 7,
        max_feature_block_days: int = 5,
    ) -> None:
        super().__init__()
        self.swe_tail_dropout_prob = float(swe_tail_dropout_prob)
        self.feature_block_dropout_prob = float(feature_block_dropout_prob)
        self.max_tail_days = max(1, int(max_tail_days))
        self.max_feature_block_days = max(1, int(max_feature_block_days))

    @staticmethod
    def _apply_keep(
        values: torch.Tensor | None, mask: torch.Tensor | None, keep: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if values is None:
            return (None, mask)
        if mask is None:
            mask = torch.isfinite(values).to(dtype=values.dtype)
        new_mask = mask.to(device=values.device) * keep.to(
            device=values.device, dtype=mask.dtype
        )
        clean = torch.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        new_values = torch.where(new_mask > 0, clean, torch.zeros_like(clean))
        return (new_values, new_mask)

    def forward(
        self,
        x_seq: torch.Tensor,
        x_mask: torch.Tensor | None,
        x_seq_raw: torch.Tensor | None,
        feature_names: list[str] | tuple[str, ...] | None,
    ) -> tuple[
        torch.Tensor, torch.Tensor | None, torch.Tensor | None, dict[str, torch.Tensor]
    ]:
        batch_size, time_steps, feature_dim = map(int, x_seq.shape)
        device = x_seq.device
        if x_mask is None:
            base_mask = torch.isfinite(x_seq).to(dtype=x_seq.dtype)
        else:
            base_mask = x_mask
        keep = torch.ones(
            batch_size, time_steps, feature_dim, device=device, dtype=torch.bool
        )
        feature_groups = resolve_missingness_feature_groups(feature_names, feature_dim)
        swe_indices = feature_groups["swe_all"]
        blocked_feature_indices = list(feature_groups["forcing"])
        swe_drop = torch.zeros(batch_size, device=device, dtype=x_seq.dtype)
        feature_drop = torch.zeros(batch_size, device=device, dtype=x_seq.dtype)
        if self.training:
            if self.swe_tail_dropout_prob > 0.0 and time_steps > 1:
                selected = torch.rand(batch_size, device=device) < min(
                    max(self.swe_tail_dropout_prob, 0.0), 1.0
                )
                max_tail = min(self.max_tail_days, time_steps - 1)
                tail_lengths = torch.randint(
                    1, max_tail + 1, (batch_size,), device=device
                )
                positions = torch.arange(
                    time_steps, device=device, dtype=torch.long
                ).reshape(1, -1)
                tail_start = time_steps - tail_lengths
                tail_drop = selected.reshape(-1, 1) & (
                    positions >= tail_start.reshape(-1, 1)
                )
                swe_columns = torch.zeros(feature_dim, device=device, dtype=torch.bool)
                swe_columns[swe_indices] = True
                keep = keep & ~(tail_drop.unsqueeze(-1) & swe_columns.reshape(1, 1, -1))
                swe_drop = selected.to(dtype=x_seq.dtype)
            if (
                blocked_feature_indices
                and self.feature_block_dropout_prob > 0.0
                and (time_steps > 1)
            ):
                selected = torch.rand(
                    batch_size, len(blocked_feature_indices), device=device
                ) < min(max(self.feature_block_dropout_prob, 0.0), 1.0)
                length = min(self.max_feature_block_days, time_steps)
                starts = torch.randint(
                    0,
                    max(1, time_steps - length + 1),
                    (batch_size, len(blocked_feature_indices)),
                    device=device,
                )
                positions = torch.arange(
                    time_steps, device=device, dtype=torch.long
                ).reshape(1, -1, 1)
                block_drop = (
                    selected.unsqueeze(1)
                    & (positions >= starts.unsqueeze(1))
                    & (positions < (starts + length).unsqueeze(1))
                )
                keep[:, :, blocked_feature_indices] = (
                    keep[:, :, blocked_feature_indices] & ~block_drop
                )
                feature_drop = selected.to(dtype=x_seq.dtype).mean(dim=-1)
        new_x, new_mask = self._apply_keep(x_seq, base_mask, keep)
        new_raw, _ = (
            self._apply_keep(
                x_seq_raw, base_mask if x_seq_raw is not None else None, keep
            )
            if x_seq_raw is not None
            else (None, None)
        )
        diagnostics = {
            "structured_mask_replay_applied": torch.tensor(
                self.training, device=device
            ),
            "structured_swe_tail_drop_sample": swe_drop,
            "structured_feature_block_drop_ratio": feature_drop,
            "structured_mask_keep_ratio": keep.to(dtype=x_seq.dtype)
            .reshape(batch_size, -1)
            .mean(dim=-1),
            "structured_mask_replay_mode": "structured",
        }
        assert new_x is not None
        return (new_x, new_mask, new_raw, diagnostics)
