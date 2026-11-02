"""Extract valid SWE observations without treating missing values as snow-free."""

from __future__ import annotations

import torch

SWE_FEATURE_NAMES = {"swe_prev", "swe_prev_observed", "target_history_swe", "swe"}
SWE_VALUE_FEATURE_NAMES = {"swe_prev", "target_history_swe", "swe"}


def _feature_names_lower(
    feature_names: list[str] | tuple[str, ...] | None,
) -> list[str]:
    return [str(name).lower() for name in feature_names or []]


def _indices_for_names(
    feature_names: list[str] | tuple[str, ...] | None,
    candidates: set[str] | tuple[str, ...],
    seq_dim: int,
) -> list[int]:
    if not feature_names:
        return []
    lowered = _feature_names_lower(feature_names)
    candidate_set = {str(name).lower() for name in candidates}
    return [
        idx for idx, name in enumerate(lowered[: int(seq_dim)]) if name in candidate_set
    ]


def get_last_valid_swe(
    x_seq: torch.Tensor,
    x_mask: torch.Tensor | None,
    feature_names: list[str] | tuple[str, ...] | None = None,
) -> dict[str, torch.Tensor]:
    batch_size, time_steps, seq_dim = (
        int(x_seq.shape[0]),
        int(x_seq.shape[1]),
        int(x_seq.shape[2]),
    )
    swe_indices = _indices_for_names(feature_names, SWE_VALUE_FEATURE_NAMES, seq_dim)
    swe_idx = swe_indices[0] if swe_indices else 0
    swe = x_seq[:, :, swe_idx]
    if x_mask is None:
        swe_valid = torch.isfinite(swe)
    else:
        swe_valid = x_mask[:, :, swe_idx].to(dtype=torch.bool) & torch.isfinite(swe)
    clean_swe = torch.where(
        swe_valid,
        torch.nan_to_num(swe, nan=0.0, posinf=0.0, neginf=0.0),
        torch.zeros_like(swe),
    )
    positions = (
        torch.arange(time_steps, dtype=torch.long, device=x_seq.device)
        .reshape(1, -1)
        .expand(batch_size, -1)
    )
    last_positions = (
        torch.where(swe_valid, positions, torch.full_like(positions, -1))
        .max(dim=-1)
        .values
    )
    gather_index = last_positions.clamp_min(0).reshape(-1, 1)
    safe_current = clean_swe.gather(dim=1, index=gather_index).reshape(-1)
    current_valid = last_positions >= 0
    safe_current = torch.where(
        current_valid, safe_current, torch.zeros_like(safe_current)
    )
    swe_ratio = swe_valid.to(dtype=x_seq.dtype).mean(dim=-1)
    return {
        "safe_current_swe": safe_current,
        "current_swe_valid": current_valid,
        "swe_history_valid_ratio": swe_ratio,
        "swe_history_last_valid_index": last_positions,
    }
