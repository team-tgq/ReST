"""Paper defaults and explicit input contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import torch

DEFAULT_FEATURE_NAMES = (
    "swe_prev",
    "swe_prev_observed",
    "day_sin",
    "day_cos",
    "tmax",
    "tmin",
    "precipitation",
    "shortwave_radiation",
    "wind_velocity",
    "relative_humidity_max",
    "relative_humidity_min",
    "specific_humidity",
    "tb_19ve",
    "tb_37ve",
    "tb_diff",
)


@dataclass(frozen=True)
class ObservedHistory:
    """One historical scale with inseparable network, physical and validity views."""

    normalized: torch.Tensor
    raw: torch.Tensor
    observed: torch.Tensor

    def validate(self, name: str, feature_dim: int) -> tuple[int, torch.device]:
        tensors = (self.normalized, self.raw, self.observed)
        if not all((torch.is_tensor(value) for value in tensors)):
            raise TypeError(f"{name} normalized/raw/observed must all be tensors")
        if self.normalized.ndim != 3:
            raise ValueError(
                f"{name} must have shape [B,T,F], got {tuple(self.normalized.shape)}"
            )
        if tuple(self.raw.shape) != tuple(self.normalized.shape):
            raise ValueError(f"{name}.raw must match {name}.normalized")
        if tuple(self.observed.shape) != tuple(self.normalized.shape):
            raise ValueError(f"{name}.observed must match {name}.normalized")
        if int(self.normalized.shape[-1]) != int(feature_dim):
            raise ValueError(
                f"{name} feature dimension must be {feature_dim}, got {self.normalized.shape[-1]}"
            )
        if self.normalized.dtype != torch.float32 or self.raw.dtype != torch.float32:
            raise TypeError(f"{name} normalized/raw must be strict FP32 tensors")
        if self.observed.dtype not in {torch.bool, torch.float32}:
            raise TypeError(f"{name}.observed must be bool or FP32")
        if (
            self.normalized.device != self.raw.device
            or self.normalized.device != self.observed.device
        ):
            raise ValueError(f"{name} normalized/raw/observed must share one device")
        return (int(self.normalized.shape[0]), self.normalized.device)


@dataclass(frozen=True)
class ReSTInput:
    """Strict input contract for the public model."""

    short: ObservedHistory
    long: ObservedHistory
    analog: ObservedHistory
    static_topography: torch.Tensor
    mechanism_prior: torch.Tensor

    def validate(self, *, feature_dim: int, static_dim: int, prior_dim: int) -> None:
        short_batch, device = self.short.validate("short", feature_dim)
        long_batch, long_device = self.long.validate("long", feature_dim)
        analog_batch, analog_device = self.analog.validate("analog", feature_dim)
        if short_batch != long_batch or short_batch != analog_batch:
            raise ValueError("short, long and analog must share the same batch size")
        if device != long_device or device != analog_device:
            raise ValueError("short, long and analog must share one device")
        if tuple(self.static_topography.shape) != (short_batch, int(static_dim)):
            raise ValueError(
                f"static_topography must have shape [{short_batch},{static_dim}], got {tuple(self.static_topography.shape)}"
            )
        if tuple(self.mechanism_prior.shape) != (short_batch, int(prior_dim)):
            raise ValueError(
                f"mechanism_prior must have shape [{short_batch},{prior_dim}], got {tuple(self.mechanism_prior.shape)}"
            )
        for name, tensor in (
            ("static_topography", self.static_topography),
            ("mechanism_prior", self.mechanism_prior),
        ):
            if tensor.dtype != torch.float32:
                raise TypeError(f"{name} must be a strict FP32 tensor")
            if tensor.device != device:
                raise ValueError(f"{name} must share the history device")


@dataclass(frozen=True)
class StructuredReplayConfig:
    swe_tail_dropout_prob: float = 0.3
    feature_block_dropout_prob: float = 0.15
    max_tail_days: int = 7
    max_feature_block_days: int = 5

    def validate(self) -> None:
        for name in ("swe_tail_dropout_prob", "feature_block_dropout_prob"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0,1], got {value}")
        if self.max_tail_days < 1 or self.max_feature_block_days < 1:
            raise ValueError("structured replay block lengths must be positive")


@dataclass(frozen=True)
class AMRMConfig:
    state_hidden_dim: int = 64
    gap_hidden_dim: int = 32
    max_gap_correction_mm: float = 100.0
    swe_free_anchor_hidden_dim: int = 64
    swe_free_anchor_aux_loss_weight: float = 0.1
    swe_free_anchor_smooth_l1_beta_mm: float = 50.0

    def validate(self) -> None:
        if self.swe_free_anchor_aux_loss_weight < 0.0:
            raise ValueError(
                "SWE-Free Anchor auxiliary loss weight must be non-negative"
            )


ARCHITECTURE_REVISION = "rest-rse-v1"


@dataclass(frozen=True)
class ReSTConfig:
    feature_names: tuple[str, ...] = DEFAULT_FEATURE_NAMES
    prior_dim: int = 5
    horizon: int = 10
    model_dim: int = 128
    nhead: int = 4
    encoder_num_layers: int = 1
    hidden_dim: int = 192
    num_experts: int = 5
    router_top_k: int = 2
    dropout: float = 0.1
    router_prior_strength: float = 0.25
    patch_length_days: int = 5
    long_patch_length_steps: int = 4
    analog_patch_length_steps: int = 5
    window_scale_eps: float = 1e-06
    architecture_revision: str = ARCHITECTURE_REVISION
    amrm: AMRMConfig = field(default_factory=AMRMConfig)
    structured_replay: StructuredReplayConfig = field(
        default_factory=StructuredReplayConfig
    )

    def __post_init__(self):
        if tuple(self.feature_names) != DEFAULT_FEATURE_NAMES:
            raise ValueError("ReST requires the manuscript's ordered 15-channel schema")
        if self.model_dim <= 0 or self.nhead <= 0 or self.model_dim % self.nhead:
            raise ValueError("model_dim must be positive and divisible by nhead")
        if (
            self.horizon != 10
            or self.prior_dim != 5
            or self.num_experts != 5
            or (self.router_top_k != 2)
        ):
            raise ValueError(
                "The paper contract requires 10 days, 5 priors, 5 experts and Top-2 routing"
            )
        self.amrm.validate()
        self.structured_replay.validate()

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, values):
        values = dict(values)
        for key, kind in (
            ("amrm", AMRMConfig),
            ("structured_replay", StructuredReplayConfig),
        ):
            if key in values:
                values[key] = kind(**values[key])
        if "feature_names" in values:
            values["feature_names"] = tuple(values["feature_names"])
        return cls(**values)
