"""Variable-preserving patch encoders and SWE-centered process attention."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, log

import torch
import torch.nn.functional as Fnn
from torch import nn

from .attention import multihead_attention_bool_mask_forward

SCALE_SHORT_INDEX = 0
SCALE_ANALOG_INDEX = 2
PHYSICS_SCALE_NAMES = ("short", "long", "analog")
STATIC_TOPOGRAPHY_FEATURE_NAMES = {
    "elevation",
    "latitude",
    "longitude",
    "landcover_code",
    "southness",
}
PHYSICS_PROCESS_GROUP_RULES: dict[str, tuple[str, ...]] = {
    "snow_state": ("swe_prev", "swe_prev_observed"),
    "seasonal_time": ("day_sin", "day_cos"),
    "thermal_energy": ("tmax", "tmin"),
    "precipitation_mass": ("precipitation",),
    "radiative_energy": ("shortwave_radiation",),
    "turbulent_moisture": (
        "wind_velocity",
        "relative_humidity_max",
        "relative_humidity_min",
        "specific_humidity",
    ),
    "microwave_observation": ("tb_19ve", "tb_37ve", "tb_diff"),
}
PHYSICS_PROCESS_GROUP_NAMES = tuple(PHYSICS_PROCESS_GROUP_RULES.keys()) + (
    "other_forcing",
)


@dataclass(frozen=True)
class PhysicsProcessGrouping:
    group_names: tuple[str, ...]
    group_indices: dict[str, tuple[int, ...]]
    group_feature_slots: dict[str, tuple[int, ...]]


class PhysicsProcessFeatureGrouper:
    def __init__(
        self,
        group_rules: dict[str, tuple[str, ...]] | None = None,
        static_names: set[str] | None = None,
    ) -> None:
        self.group_rules = group_rules or PHYSICS_PROCESS_GROUP_RULES
        self.group_names = tuple(self.group_rules.keys()) + ("other_forcing",)
        self.static_names = {
            str(name).lower()
            for name in static_names or STATIC_TOPOGRAPHY_FEATURE_NAMES
        }
        self._feature_to_group: dict[str, str] = {}
        self._feature_to_slot: dict[str, int] = {}
        for group_name, feature_names in self.group_rules.items():
            for slot, feature_name in enumerate(feature_names):
                self._feature_to_group[str(feature_name).lower()] = group_name
                self._feature_to_slot[str(feature_name).lower()] = int(slot)

    def group(
        self,
        feature_names: list[str] | tuple[str, ...] | None,
        seq_dim: int | None = None,
    ) -> PhysicsProcessGrouping:
        indices: dict[str, list[int]] = {
            group_name: [] for group_name in self.group_names
        }
        feature_slots: dict[str, list[int]] = {
            group_name: [] for group_name in self.group_names
        }
        if feature_names is None:
            feature_count = int(seq_dim or 0)
            indices["other_forcing"] = list(range(feature_count))
            feature_slots["other_forcing"] = list(range(feature_count))
            return PhysicsProcessGrouping(
                self.group_names,
                {name: tuple(values) for name, values in indices.items()},
                {name: tuple(values) for name, values in feature_slots.items()},
            )
        names = [str(name) for name in feature_names]
        feature_count = int(seq_dim if seq_dim is not None else len(names))
        for idx, raw_name in enumerate(names[:feature_count]):
            name = raw_name.lower()
            if name in self.static_names:
                continue
            group_name = self._feature_to_group.get(name, "other_forcing")
            indices[group_name].append(int(idx))
            feature_slots[group_name].append(self._feature_to_slot.get(name, int(idx)))
        return PhysicsProcessGrouping(
            self.group_names,
            {name: tuple(values) for name, values in indices.items()},
            {name: tuple(values) for name, values in feature_slots.items()},
        )


class PhysicsGroupedVariableDSWEmbedding(nn.Module):
    def __init__(
        self,
        patch_length: int,
        model_dim: int,
        num_variable_slots: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.patch_length = max(1, int(patch_length))
        self.model_dim = int(model_dim)
        self.num_variable_slots = max(1, int(num_variable_slots))
        self.patch_projection = nn.Sequential(
            nn.Linear(self.patch_length, self.model_dim), nn.GELU(), nn.Dropout(dropout)
        )
        self.variable_embedding = nn.Embedding(self.num_variable_slots, self.model_dim)
        self.valid_ratio_projection = nn.Linear(1, self.model_dim)
        self.token_norm = nn.LayerNorm(self.model_dim)
        self.token_dropout = nn.Dropout(dropout)

    @staticmethod
    def _patch_position_encoding(
        patch_count: int, model_dim: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        if patch_count <= 0:
            return torch.zeros(0, model_dim, device=device, dtype=dtype)
        positions = torch.arange(
            patch_count, device=device, dtype=torch.float32
        ).unsqueeze(1)
        frequency = torch.exp(
            torch.arange(0, model_dim, 2, device=device, dtype=torch.float32)
            * (-log(10000.0) / max(1, model_dim))
        )
        angles = positions * frequency.unsqueeze(0)
        encoding = torch.zeros(
            patch_count, model_dim, device=device, dtype=torch.float32
        )
        encoding[:, 0::2] = torch.sin(angles)
        encoding[:, 1::2] = torch.cos(angles[:, : encoding[:, 1::2].shape[1]])
        return encoding.to(dtype=dtype)

    def forward(
        self,
        group_x: torch.Tensor,
        group_mask: torch.Tensor | None = None,
        variable_slots: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if group_x.ndim != 3:
            raise ValueError(
                f"group_x must have shape [B,T,F_g], got {tuple(group_x.shape)}"
            )
        if group_mask is not None and tuple(group_mask.shape) != tuple(group_x.shape):
            raise ValueError(
                f"group_mask must match group_x shape {tuple(group_x.shape)}, got {tuple(group_mask.shape)}"
            )
        batch_size, time_steps, variable_count = map(int, group_x.shape)
        patch_count = max(1, int(ceil(max(1, time_steps) / self.patch_length)))
        if variable_count == 0 or time_steps == 0:
            tokens = group_x.new_zeros(
                batch_size, variable_count, patch_count, self.model_dim
            )
            valid = torch.zeros(
                batch_size,
                variable_count,
                patch_count,
                dtype=torch.bool,
                device=group_x.device,
            )
            return (tokens, valid)
        finite = torch.isfinite(group_x)
        declared_valid = (
            torch.ones_like(group_x, dtype=torch.bool)
            if group_mask is None
            else group_mask.to(device=group_x.device, dtype=torch.bool)
        )
        element_valid = finite & declared_valid
        clean = torch.nan_to_num(group_x, nan=0.0, posinf=0.0, neginf=0.0)
        clean = torch.where(element_valid, clean, torch.zeros_like(clean))
        values_by_variable = clean.transpose(1, 2)
        valid_by_variable = element_valid.transpose(1, 2)
        pad_steps = patch_count * self.patch_length - time_steps
        if pad_steps > 0:
            values_by_variable = Fnn.pad(values_by_variable, (0, pad_steps), value=0.0)
            valid_by_variable = Fnn.pad(valid_by_variable, (0, pad_steps), value=False)
        patches = values_by_variable.reshape(
            batch_size, variable_count, patch_count, self.patch_length
        )
        patch_element_valid = valid_by_variable.reshape(
            batch_size, variable_count, patch_count, self.patch_length
        )
        patch_valid = patch_element_valid.any(dim=-1)
        valid_ratio = patch_element_valid.to(dtype=patches.dtype).mean(
            dim=-1, keepdim=True
        )
        if variable_slots is None:
            slots = torch.arange(
                variable_count, device=group_x.device, dtype=torch.long
            )
        else:
            slots = variable_slots.to(device=group_x.device, dtype=torch.long).reshape(
                -1
            )
        if int(slots.numel()) != variable_count:
            raise ValueError(
                f"variable_slots must contain F_g={variable_count} entries, got {int(slots.numel())}"
            )
        if bool((slots < 0).any()) or bool((slots >= self.num_variable_slots).any()):
            raise ValueError(
                f"variable_slots must be in [0,{self.num_variable_slots}), got {slots.detach().cpu().tolist()}"
            )
        value_token = self.patch_projection(patches)
        variable_token = self.variable_embedding(slots).reshape(
            1, variable_count, 1, self.model_dim
        )
        position_token = self._patch_position_encoding(
            patch_count, self.model_dim, group_x.device, group_x.dtype
        ).reshape(1, 1, patch_count, self.model_dim)
        validity_token = self.valid_ratio_projection(valid_ratio)
        tokens = self.token_norm(
            value_token + variable_token + position_token + validity_token
        )
        tokens = self.token_dropout(tokens)
        tokens = torch.where(
            patch_valid.unsqueeze(-1), tokens, torch.zeros_like(tokens)
        )
        return (tokens, patch_valid)


class PhysicsGroupedVariableTemporalEncoder(nn.Module):
    def __init__(
        self, model_dim: int, nhead: int, num_layers: int = 1, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.model_dim = int(model_dim)
        layer_count = max(1, int(num_layers))
        self.temporal_attention_layers = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    model_dim, nhead, dropout=dropout, batch_first=True
                )
                for _ in range(layer_count)
            ]
        )
        self.temporal_attention_norms = nn.ModuleList(
            [nn.LayerNorm(model_dim) for _ in range(layer_count)]
        )
        self.temporal_ffn_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(model_dim, model_dim * 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(model_dim * 2, model_dim),
                )
                for _ in range(layer_count)
            ]
        )
        self.temporal_ffn_norms = nn.ModuleList(
            [nn.LayerNorm(model_dim) for _ in range(layer_count)]
        )
        self.variable_score = nn.Sequential(
            nn.LayerNorm(model_dim), nn.Linear(model_dim, 1, bias=False)
        )
        self.group_norm = nn.LayerNorm(model_dim)

    def forward(
        self, variable_patch_tokens: torch.Tensor, variable_patch_valid: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if variable_patch_tokens.ndim != 4:
            raise ValueError(
                f"variable_patch_tokens must have shape [B,F_g,L,D], got {tuple(variable_patch_tokens.shape)}"
            )
        if tuple(variable_patch_valid.shape) != tuple(variable_patch_tokens.shape[:3]):
            raise ValueError(
                f"variable_patch_valid must match [B,F_g,L], got {tuple(variable_patch_valid.shape)} for tokens {tuple(variable_patch_tokens.shape)}"
            )
        batch_size, variable_count, patch_count, model_dim = map(
            int, variable_patch_tokens.shape
        )
        if model_dim != self.model_dim:
            raise ValueError(
                f"expected token dim D={self.model_dim}, got D={model_dim}"
            )
        if variable_count == 0 or patch_count == 0:
            token = variable_patch_tokens.new_zeros(batch_size, self.model_dim)
            group_valid = torch.zeros(
                batch_size, dtype=torch.bool, device=variable_patch_tokens.device
            )
            variable_weights = variable_patch_tokens.new_zeros(
                batch_size, variable_count
            )
            variable_valid = torch.zeros(
                batch_size,
                variable_count,
                dtype=torch.bool,
                device=variable_patch_tokens.device,
            )
            return (token, group_valid, variable_weights, variable_valid)
        original_patch_valid = variable_patch_valid.to(dtype=torch.bool)
        flat_valid = original_patch_valid.reshape(
            batch_size * variable_count, patch_count
        )
        variable_valid_flat = flat_valid.any(dim=-1)
        safe_valid = flat_valid.clone()
        safe_valid[~variable_valid_flat, 0] = True
        h = variable_patch_tokens.reshape(
            batch_size * variable_count, patch_count, self.model_dim
        )
        h = torch.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)
        h = torch.where(flat_valid.unsqueeze(-1), h, torch.zeros_like(h))
        for attention, attention_norm, ffn, ffn_norm in zip(
            self.temporal_attention_layers,
            self.temporal_attention_norms,
            self.temporal_ffn_layers,
            self.temporal_ffn_norms,
        ):
            attended, _ = multihead_attention_bool_mask_forward(
                attention, h, h, h, key_padding_mask=~safe_valid, need_weights=False
            )
            h = attention_norm(h + attended)
            h = ffn_norm(h + ffn(h))
            h = torch.where(flat_valid.unsqueeze(-1), h, torch.zeros_like(h))
        patch_weights = flat_valid.to(dtype=h.dtype).unsqueeze(-1)
        variable_tokens = (h * patch_weights).sum(dim=1) / patch_weights.sum(
            dim=1
        ).clamp_min(1.0)
        variable_tokens = variable_tokens.reshape(
            batch_size, variable_count, self.model_dim
        )
        variable_valid = variable_valid_flat.reshape(batch_size, variable_count)
        variable_tokens = torch.where(
            variable_valid.unsqueeze(-1),
            variable_tokens,
            torch.zeros_like(variable_tokens),
        )
        group_valid = variable_valid.any(dim=-1)
        safe_variable_valid = variable_valid.clone()
        safe_variable_valid[~group_valid, 0] = True
        variable_scores = self.variable_score(variable_tokens).squeeze(-1)
        variable_scores = variable_scores.masked_fill(
            ~safe_variable_valid, torch.finfo(variable_scores.dtype).min
        )
        variable_weights = torch.softmax(variable_scores, dim=-1)
        variable_weights = torch.where(
            variable_valid, variable_weights, torch.zeros_like(variable_weights)
        )
        variable_weights = variable_weights / variable_weights.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-12)
        variable_weights = torch.where(
            group_valid.unsqueeze(-1),
            variable_weights,
            torch.zeros_like(variable_weights),
        )
        pooled = torch.sum(variable_tokens * variable_weights.unsqueeze(-1), dim=1)
        group_token = self.group_norm(pooled)
        group_token = torch.where(
            group_valid.unsqueeze(-1), group_token, torch.zeros_like(group_token)
        )
        return (group_token, group_valid, variable_weights, variable_valid)


class SWECentricHydroProcessAttention(nn.Module):
    def __init__(self, model_dim: int, nhead: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.model_dim = int(model_dim)
        self.query_projection = nn.Sequential(
            nn.Linear(model_dim + 1, model_dim), nn.GELU(), nn.LayerNorm(model_dim)
        )
        self.attention = nn.MultiheadAttention(
            model_dim, nhead, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(model_dim)
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, model_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim * 2, model_dim),
        )
        self.ffn_norm = nn.LayerNorm(model_dim)

    def forward(
        self,
        query_source: torch.Tensor,
        current_swe: torch.Tensor | None,
        evidence_tokens: torch.Tensor,
        token_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, token_count = (
            int(evidence_tokens.shape[0]),
            int(evidence_tokens.shape[1]),
        )
        if current_swe is None:
            current = query_source.new_zeros(batch_size, 1)
        else:
            current = torch.nan_to_num(
                current_swe.reshape(batch_size, 1).to(
                    device=query_source.device, dtype=query_source.dtype
                ),
                nan=0.0,
            )
        query = self.query_projection(
            torch.cat([query_source, current], dim=-1)
        ).unsqueeze(1)
        if token_count == 0:
            empty_weights = query_source.new_zeros(batch_size, 0)
            return (query.squeeze(1), empty_weights, token_valid)
        original_valid = token_valid.to(dtype=torch.bool)
        row_valid = original_valid.any(dim=-1)
        safe_valid = original_valid.clone()
        safe_valid[~row_valid, 0] = True
        safe_tokens = torch.nan_to_num(evidence_tokens, nan=0.0, posinf=0.0, neginf=0.0)
        attended, weights = multihead_attention_bool_mask_forward(
            self.attention,
            query,
            safe_tokens,
            safe_tokens,
            key_padding_mask=~safe_valid,
            need_weights=True,
            average_attn_weights=True,
        )
        context = self.norm(query.squeeze(1) + attended.squeeze(1))
        context = self.ffn_norm(context + self.ffn(context))
        context = torch.where(row_valid.unsqueeze(-1), context, query.squeeze(1))
        weights = weights.squeeze(1)
        weights = torch.where(original_valid, weights, torch.zeros_like(weights))
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        weights = torch.where(
            row_valid.unsqueeze(-1), weights, torch.zeros_like(weights)
        )
        return (context, weights, original_valid)


class PhysicsGuidedHydroProcessEncoder(nn.Module):
    def __init__(
        self,
        seq_dim: int,
        static_dim: int,
        prior_dim: int,
        model_dim: int = 128,
        nhead: int = 4,
        num_layers: int = 1,
        dropout: float = 0.1,
        patch_length_days: int = 5,
        long_patch_length_steps: int = 4,
        analog_patch_length_steps: int = 5,
        swe_feature_index: int = 0,
        include_prior_token: bool = True,
    ) -> None:
        super().__init__()
        self.seq_dim = int(seq_dim)
        self.model_dim = int(model_dim)
        self.include_prior_token = bool(include_prior_token)
        self.swe_feature_index = int(swe_feature_index)
        self.grouper = PhysicsProcessFeatureGrouper()
        self.group_names = PHYSICS_PROCESS_GROUP_NAMES
        self.scale_names = PHYSICS_SCALE_NAMES
        patch_lengths = {
            "short": max(1, int(patch_length_days)),
            "long": max(1, int(long_patch_length_steps)),
            "analog": max(1, int(analog_patch_length_steps)),
        }
        self.patch_embeddings = nn.ModuleDict()
        self.group_temporal_encoders = nn.ModuleDict()
        variable_slot_counts = {
            group_name: len(self.grouper.group_rules[group_name])
            if group_name in self.grouper.group_rules
            else self.seq_dim
            for group_name in self.group_names
        }
        for scale_name in self.scale_names:
            for group_name in self.group_names:
                key = self._module_key(scale_name, group_name)
                self.patch_embeddings[key] = PhysicsGroupedVariableDSWEmbedding(
                    patch_lengths[scale_name],
                    model_dim,
                    num_variable_slots=max(1, int(variable_slot_counts[group_name])),
                    dropout=dropout,
                )
                self.group_temporal_encoders[key] = (
                    PhysicsGroupedVariableTemporalEncoder(
                        model_dim, nhead, num_layers=num_layers, dropout=dropout
                    )
                )
        self.static_projection = nn.Sequential(
            nn.Linear(static_dim, model_dim), nn.GELU(), nn.LayerNorm(model_dim)
        )
        self.prior_projection = (
            nn.Sequential(
                nn.Linear(prior_dim, model_dim), nn.GELU(), nn.LayerNorm(model_dim)
            )
            if self.include_prior_token
            else None
        )
        self.swe_centric_attention = SWECentricHydroProcessAttention(
            model_dim, nhead, dropout=dropout
        )

    @staticmethod
    def _module_key(scale_name: str, group_name: str) -> str:
        return f"{scale_name}__{group_name}"

    def _swe_feature_index(
        self, feature_names: list[str] | tuple[str, ...] | None, feature_dim: int
    ) -> int:
        if feature_names is not None:
            lowered = [str(name).lower() for name in feature_names]
            for candidate in ("swe_prev", "target_history_swe", "swe"):
                if candidate in lowered:
                    return int(lowered.index(candidate))
        return min(max(self.swe_feature_index, 0), max(0, int(feature_dim) - 1))

    @staticmethod
    def _summarize_process_groups(
        group_tokens: torch.Tensor, group_valid: torch.Tensor
    ) -> torch.Tensor:
        weights = group_valid.to(dtype=group_tokens.dtype).unsqueeze(-1)
        scale_valid = group_valid.any(dim=-1)
        scale_token = (group_tokens * weights).sum(dim=1) / weights.sum(
            dim=1
        ).clamp_min(1.0)
        return torch.where(
            scale_valid.unsqueeze(-1), scale_token, torch.zeros_like(scale_token)
        )

    def _encode_process_group(
        self,
        scale_name: str,
        group_name: str,
        scale_x: torch.Tensor,
        scale_mask: torch.Tensor | None,
        grouping: PhysicsProcessGrouping,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        indices = grouping.group_indices.get(group_name, ())
        slots = grouping.group_feature_slots.get(group_name, ())
        index_tensor = torch.tensor(indices, dtype=torch.long, device=scale_x.device)
        slot_tensor = torch.tensor(slots, dtype=torch.long, device=scale_x.device)
        if indices:
            group_x = scale_x.index_select(dim=-1, index=index_tensor)
            group_mask = (
                None
                if scale_mask is None
                else scale_mask.index_select(dim=-1, index=index_tensor)
            )
        else:
            group_x = scale_x.new_zeros(scale_x.shape[0], scale_x.shape[1], 0)
            group_mask = (
                None
                if scale_mask is None
                else scale_mask.new_zeros(scale_mask.shape[0], scale_mask.shape[1], 0)
            )
        key = self._module_key(scale_name, group_name)
        patch_tokens, patch_valid = self.patch_embeddings[key](
            group_x, group_mask, slot_tensor
        )
        group_token, group_valid, variable_weights, variable_valid = (
            self.group_temporal_encoders[key](patch_tokens, patch_valid)
        )
        return (
            group_token,
            group_valid,
            variable_weights,
            variable_valid,
            index_tensor,
        )

    def _encode_scale(
        self,
        scale_name: str,
        scale_x: torch.Tensor | None,
        scale_mask: torch.Tensor | None,
        grouping: PhysicsProcessGrouping,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if scale_x is None:
            group_tokens = torch.zeros(
                batch_size,
                len(self.group_names),
                self.model_dim,
                device=device,
                dtype=dtype,
            )
            group_valid = torch.zeros(
                batch_size, len(self.group_names), dtype=torch.bool, device=device
            )
            scale_token = torch.zeros(
                batch_size, self.model_dim, device=device, dtype=dtype
            )
            variable_attention = torch.zeros(
                batch_size,
                len(self.group_names),
                self.seq_dim,
                device=device,
                dtype=dtype,
            )
            variable_attention_valid = torch.zeros(
                batch_size,
                len(self.group_names),
                self.seq_dim,
                device=device,
                dtype=torch.bool,
            )
            return (
                group_tokens,
                group_valid,
                scale_token,
                variable_attention,
                variable_attention_valid,
            )
        tokens: list[torch.Tensor] = []
        valids: list[torch.Tensor] = []
        variable_attention = torch.zeros(
            batch_size, len(self.group_names), self.seq_dim, device=device, dtype=dtype
        )
        variable_attention_valid = torch.zeros(
            batch_size,
            len(self.group_names),
            self.seq_dim,
            device=device,
            dtype=torch.bool,
        )
        for group_index, group_name in enumerate(self.group_names):
            (
                group_token,
                group_valid,
                local_variable_weights,
                local_variable_valid,
                index_tensor,
            ) = self._encode_process_group(
                scale_name=scale_name,
                group_name=group_name,
                scale_x=scale_x,
                scale_mask=scale_mask,
                grouping=grouping,
            )
            tokens.append(group_token)
            valids.append(group_valid)
            if int(index_tensor.numel()) > 0:
                variable_attention[:, group_index, index_tensor] = (
                    local_variable_weights
                )
                variable_attention_valid[:, group_index, index_tensor] = (
                    local_variable_valid
                )
        group_tokens = torch.stack(tokens, dim=1)
        group_valid = torch.stack(valids, dim=1)
        scale_token = self._summarize_process_groups(group_tokens, group_valid)
        return (
            group_tokens,
            group_valid,
            scale_token,
            variable_attention,
            variable_attention_valid,
        )

    def _project_static_token(
        self, x_static: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        clean_static = torch.nan_to_num(x_static, nan=0.0, posinf=0.0, neginf=0.0)
        token = self.static_projection(clean_static)
        valid = torch.isfinite(x_static).any(dim=-1)
        return (torch.where(valid.unsqueeze(-1), token, torch.zeros_like(token)), valid)

    def _project_prior_token(
        self, x_prior: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.prior_projection is None:
            raise RuntimeError("prior evidence token is disabled for this encoder")
        clean_prior = torch.nan_to_num(x_prior, nan=0.0, posinf=0.0, neginf=0.0)
        token = self.prior_projection(clean_prior)
        valid = torch.isfinite(x_prior).any(dim=-1)
        return (torch.where(valid.unsqueeze(-1), token, torch.zeros_like(token)), valid)

    def _swe_diagnostics(
        self,
        x_seq: torch.Tensor,
        x_mask: torch.Tensor | None,
        current_swe: torch.Tensor | None,
        feature_names: list[str] | tuple[str, ...] | None,
    ) -> dict[str, torch.Tensor]:
        swe_idx = self._swe_feature_index(feature_names, int(x_seq.shape[-1]))
        swe = x_seq[:, :, swe_idx]
        if x_mask is None:
            swe_valid = torch.ones_like(swe, dtype=torch.bool)
        else:
            swe_valid = x_mask[:, :, swe_idx].to(dtype=torch.bool)
        swe_valid = swe_valid & torch.isfinite(swe)
        clean_swe = torch.where(
            swe_valid,
            torch.nan_to_num(swe, nan=0.0, posinf=0.0, neginf=0.0),
            torch.zeros_like(swe),
        )
        time_steps = int(swe.shape[1])
        time_index = torch.arange(
            time_steps, device=swe.device, dtype=swe.dtype
        ).reshape(1, -1)
        weights = swe_valid.to(dtype=swe.dtype)
        count = weights.sum(dim=-1).clamp_min(1.0)
        time_mean = (time_index * weights).sum(dim=-1) / count
        swe_mean = clean_swe.sum(dim=-1) / count
        centered_time = time_index - time_mean.unsqueeze(-1)
        centered_swe = clean_swe - swe_mean.unsqueeze(-1)
        variance = (centered_time.square() * weights).sum(dim=-1)
        covariance = (centered_time * centered_swe * weights).sum(dim=-1)
        trend_slope = torch.where(
            variance > 0,
            covariance / variance.clamp_min(1e-06),
            torch.zeros_like(variance),
        )
        trend_slope = torch.where(
            weights.sum(dim=-1) > 1, trend_slope, torch.zeros_like(trend_slope)
        )
        trend = swe_mean.unsqueeze(-1) + trend_slope.unsqueeze(-1) * centered_time
        trend = torch.where(swe_valid, trend, torch.zeros_like(trend))
        residual = torch.where(
            swe_valid, clean_swe - trend, torch.zeros_like(clean_swe)
        )
        residual_volatility = torch.sqrt(
            (residual.square().sum(dim=-1) / count).clamp_min(0.0)
        )
        valid_observation_ratio = weights.mean(dim=-1)
        if current_swe is not None:
            current = torch.nan_to_num(
                current_swe.reshape(-1).to(device=x_seq.device, dtype=x_seq.dtype),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
        else:
            positions = torch.arange(
                time_steps, device=x_seq.device, dtype=torch.long
            ).reshape(1, -1)
            last_positions = (
                torch.where(swe_valid, positions, torch.full_like(positions, -1))
                .max(dim=-1)
                .values
            )
            gather_index = last_positions.clamp_min(0).reshape(-1, 1)
            current = clean_swe.gather(dim=1, index=gather_index).reshape(-1)
            current = torch.where(
                last_positions >= 0, current, torch.zeros_like(current)
            )
        return {
            "trend": trend,
            "residual": residual,
            "trend_slope": trend_slope,
            "residual_volatility": residual_volatility,
            "current_swe": current,
            "valid_observation_ratio": valid_observation_ratio,
        }

    def forward(
        self,
        x_seq: torch.Tensor,
        x_mask: torch.Tensor | None,
        x_static: torch.Tensor,
        x_prior: torch.Tensor,
        current_swe: torch.Tensor | None = None,
        x_long: torch.Tensor | None = None,
        x_long_mask: torch.Tensor | None = None,
        x_analog: torch.Tensor | None = None,
        x_analog_mask: torch.Tensor | None = None,
        feature_names: list[str] | tuple[str, ...] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        grouping = self.grouper.group(feature_names, seq_dim=int(x_seq.shape[-1]))
        batch_size = int(x_seq.shape[0])
        device, dtype = (x_seq.device, x_seq.dtype)
        short_groups, short_valid, _, short_variable_attention, short_variable_valid = (
            self._encode_scale(
                "short", x_seq, x_mask, grouping, batch_size, device, dtype
            )
        )
        short_token = self._summarize_process_groups(short_groups, short_valid)
        (
            long_groups,
            long_valid,
            long_token,
            long_variable_attention,
            long_variable_valid,
        ) = self._encode_scale(
            "long", x_long, x_long_mask, grouping, batch_size, device, dtype
        )
        (
            analog_groups,
            analog_valid,
            analog_token,
            analog_variable_attention,
            analog_variable_valid,
        ) = self._encode_scale(
            "analog", x_analog, x_analog_mask, grouping, batch_size, device, dtype
        )
        group_tokens = torch.stack([short_groups, long_groups, analog_groups], dim=1)
        group_valid = torch.stack([short_valid, long_valid, analog_valid], dim=1)
        variable_attention = torch.stack(
            [
                short_variable_attention,
                long_variable_attention,
                analog_variable_attention,
            ],
            dim=1,
        )
        variable_attention_valid = torch.stack(
            [short_variable_valid, long_variable_valid, analog_variable_valid], dim=1
        )
        scale_tokens = torch.stack([short_token, long_token, analog_token], dim=1)
        scale_valid = group_valid.any(dim=-1)
        swe_diag = self._swe_diagnostics(
            x_seq=x_seq,
            x_mask=x_mask,
            current_swe=current_swe,
            feature_names=feature_names,
        )
        static_token, static_valid = self._project_static_token(x_static)
        prior_token: torch.Tensor | None = None
        prior_valid: torch.Tensor | None = None
        if self.include_prior_token:
            prior_token, prior_valid = self._project_prior_token(x_prior)
        token_parts = [
            short_groups,
            long_groups,
            analog_groups,
            static_token.unsqueeze(1),
        ]
        valid_parts = [short_valid, long_valid, analog_valid, static_valid.unsqueeze(1)]
        if prior_token is not None and prior_valid is not None:
            token_parts.append(prior_token.unsqueeze(1))
            valid_parts.append(prior_valid.unsqueeze(1))
        evidence_tokens = torch.cat(token_parts, dim=1)
        evidence_valid = torch.cat(valid_parts, dim=1)
        snow_group_index = self.group_names.index("snow_state")
        snow_query = short_groups[:, snow_group_index, :]
        short_query = scale_tokens[:, SCALE_SHORT_INDEX, :]
        query_source = torch.where(
            short_valid[:, snow_group_index].unsqueeze(-1), snow_query, short_query
        )
        temporal_context, attention_weights, attention_token_valid = (
            self.swe_centric_attention(
                query_source=query_source,
                current_swe=swe_diag["current_swe"],
                evidence_tokens=evidence_tokens,
                token_valid=evidence_valid,
            )
        )
        analog_consistency = scale_valid[:, SCALE_ANALOG_INDEX].to(dtype=dtype)
        dynamic_feature_names = (
            []
            if feature_names is None
            else [str(name) for name in feature_names[: self.seq_dim]]
        )
        dynamic_feature_names.extend(
            (
                f"feature_{idx}"
                for idx in range(len(dynamic_feature_names), self.seq_dim)
            )
        )
        diagnostics: dict[str, torch.Tensor] = {
            "variable_patch_process_encoder_used": torch.tensor(
                True, dtype=torch.bool, device=device
            ),
            "physics_grouped_variable_dsw_used": torch.tensor(
                True, dtype=torch.bool, device=device
            ),
            "encoder_context_includes_static": torch.tensor(
                True, dtype=torch.bool, device=device
            ),
            "physics_process_group_tokens": group_tokens,
            "physics_process_group_valid": group_valid,
            "physics_process_variable_attention_weights": variable_attention,
            "physics_process_variable_attention_valid": variable_attention_valid,
            "physics_process_attention_weights": attention_weights,
            "physics_process_attention_token_valid": attention_token_valid,
            "scale_tokens": scale_tokens,
            "scale_valid": scale_valid,
            "analog_context": scale_tokens[:, SCALE_ANALOG_INDEX, :],
            "analog_top_indices": torch.zeros(
                batch_size, 0, dtype=torch.long, device=device
            ),
            "analog_top_weights": torch.zeros(
                batch_size, 0, dtype=dtype, device=device
            ),
            "analog_consistency": analog_consistency,
            **swe_diag,
        }
        diagnostics["physics_process_group_names"] = list(self.group_names)
        diagnostics["physics_process_scale_names"] = list(self.scale_names)
        diagnostics["physics_process_dynamic_feature_names"] = dynamic_feature_names
        attention_token_names = [
            *(f"short:{name}" for name in self.group_names),
            *(f"long:{name}" for name in self.group_names),
            *(f"analog:{name}" for name in self.group_names),
            "static_topography",
        ]
        if self.include_prior_token:
            attention_token_names.append("prior_mechanism")
        diagnostics["physics_process_attention_token_names"] = attention_token_names
        return (temporal_context, diagnostics)
