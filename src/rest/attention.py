"""Boolean-mask attention, including the PyTorch 2.0 compatibility path."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


def _torch_20_mask_fastpath_bug() -> bool:
    version = torch.__version__.split("+", 1)[0].split(".")
    try:
        return (int(version[0]), int(version[1])) == (2, 0)
    except (IndexError, ValueError):
        return False


def _normalize_bool_mask(mask: torch.Tensor | None) -> torch.Tensor | None:
    if mask is None:
        return None
    bool_mask = mask.to(dtype=torch.bool)
    if not bool(bool_mask.any()):
        return None
    return bool_mask


def multihead_attention_bool_mask_forward(
    attention: nn.MultiheadAttention,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    key_padding_mask: torch.Tensor | None = None,
    attn_mask: torch.Tensor | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run MHA with bool masks while avoiding PyTorch 2.0's masked eval fast path."""
    key_padding_mask = _normalize_bool_mask(key_padding_mask)
    attn_mask = _normalize_bool_mask(attn_mask)
    has_mask = key_padding_mask is not None or attn_mask is not None
    affected_eval = (
        _torch_20_mask_fastpath_bug()
        and (not attention.training)
        and (not torch.is_grad_enabled())
    )
    if has_mask and affected_eval and (query is key) and (key is value):
        value = value.clone()
    call_kwargs = dict(kwargs)
    if key_padding_mask is not None:
        call_kwargs["key_padding_mask"] = key_padding_mask
    if attn_mask is not None:
        call_kwargs["attn_mask"] = attn_mask
    return attention(query, key, value, **call_kwargs)
