"""Causal three-scale windows, seven study splits, and train-only normalization."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .config import DEFAULT_FEATURE_NAMES, ObservedHistory, ReSTInput

SOURCES = ("snotel", "canswe", "cdec_sw1_bulk", "norswe_region_europe_nonoverlap")
STATIC_NAMES = ("elevation", "latitude", "longitude", "landcover_code", "southness")
PRIOR_NAMES = (
    "prior_accumulation",
    "prior_stable_persistence",
    "prior_melt",
    "prior_high_elevation_cold_snow",
    "prior_sparse_source",
)
PROTOCOLS = (
    "temporal",
    "cs_snotel",
    "cs_canswe",
    "cs_cdec",
    "cs_norswe",
    "ce_high",
    "ce_lower",
)
MINMAX_DYNAMIC = np.array([0, *range(4, 15)])


def history_indices(dates, issue_days):
    """Return Short, Long and Analogue indices; -1 marks absent past dates."""
    dates = np.asarray(dates, dtype="datetime64[D]")
    issue_days = np.asarray(issue_days, dtype=int)
    short = issue_days[:, None] + np.arange(-29, 1)
    long = issue_days[:, None] + np.arange(-210, 0, 7)
    issue_dates = dates[issue_days]
    years = issue_dates.astype("datetime64[Y]").astype(int) + 1970
    october = np.array([f"{year}-10-01" for year in years], dtype="datetime64[D]")
    start_years = years - (issue_dates < october)
    starts = np.array([f"{year}-10-01" for year in start_years], dtype="datetime64[D]")
    days_in_year = issue_dates - starts
    centers = np.stack(
        [
            np.array(
                [f"{year - back}-10-01" for year in start_years], dtype="datetime64[D]"
            )
            + days_in_year
            for back in range(5, 0, -1)
        ],
        axis=1,
    )
    candidates = (
        centers[:, :, None] + np.arange(-7, 8).astype("timedelta64[D]")
    ).reshape(-1, 75)
    analog = np.searchsorted(dates, candidates)
    clipped = np.clip(analog, 0, len(dates) - 1)
    valid = (
        (analog < len(dates))
        & (dates[clipped] == candidates)
        & (candidates <= issue_dates[:, None])
    )
    analog = np.where(valid, analog, -1)
    short = np.where(short >= 0, short, -1)
    long = np.where(long >= 0, long, -1)
    return short, long, analog


class TaskSource:
    """Read a seven-file task dataset without copying its large arrays."""

    def __init__(self, path, priors):
        path = Path(path)
        self.name = path.name
        meta = json.loads((path / "meta.json").read_text(encoding="utf-8-sig"))
        names = meta["feature_names"]
        self.feature_indices = [names.index(n) for n in DEFAULT_FEATURE_NAMES]
        self.sites = pd.read_csv(path / "sites.csv", dtype={"site_id": str})
        self.dates = np.load(path / "dates.npy", allow_pickle=False).astype(
            "datetime64[D]"
        )
        if len(self.dates) < 1 or not np.all(
            np.diff(self.dates) == np.timedelta64(1, "D")
        ):
            raise ValueError(f"{self.name}: dates must be unique, ascending and daily")
        for name in ("inputs", "feature_mask", "targets", "target_mask"):
            setattr(
                self,
                name,
                np.load(path / f"{name}.npy", mmap_mode="r", allow_pickle=False),
            )
        if self.inputs.shape != self.feature_mask.shape or self.inputs.shape != (
            len(self.dates),
            len(self.sites),
            len(names),
        ):
            raise ValueError(
                f"{self.name}: input shape does not match dates/sites/schema"
            )
        if self.targets.shape != self.target_mask.shape or self.targets.shape != (
            *self.inputs.shape[:2],
            1,
        ):
            raise ValueError(f"{self.name}: targets must have shape [time, site, 1]")
        if meta.get("target_unit") != "mm":
            raise ValueError(f"{self.name}: target_unit must be mm")
        aliases = {"latitude": "lat", "longitude": "lon"}
        self.static = np.stack(
            [
                self.sites[n if n in self.sites else aliases.get(n, n)].to_numpy(
                    np.float64
                )
                for n in STATIC_NAMES
            ],
            axis=1,
        )
        if not np.isfinite(self.static).all():
            raise ValueError(f"{self.name}: static attributes must be finite")
        if self.sites.site_id.duplicated().any():
            raise ValueError(f"{self.name}: duplicate site_id")
        keys = pd.MultiIndex.from_arrays(
            [[self.name] * len(self.sites), self.sites.site_id]
        )
        self.prior = priors.reindex(keys)[list(PRIOR_NAMES)].to_numpy(np.float32)
        if (
            not np.isfinite(self.prior).all()
            or (self.prior < 0).any()
            or (self.prior.sum(1) <= 0).any()
        ):
            raise ValueError(
                f"{self.name}: missing or invalid offline prior; supply one row per site"
            )
        self.prior /= self.prior.sum(1, keepdims=True)


class WindowDataset(Dataset):
    """Forecast samples with complete 10-day targets and possibly incomplete inputs."""

    def __init__(
        self,
        root,
        prior_table,
        *,
        sources=SOURCES,
        start="2010-01-01",
        end="2020-12-31",
    ):
        priors = pd.read_csv(
            prior_table, dtype={"source": str, "site_id": str}
        ).set_index(["source", "site_id"])
        if priors.index.duplicated().any():
            raise ValueError("Duplicate source/site_id in prior table")
        self.sources = [TaskSource(Path(root) / s, priors) for s in sources]
        self.windows = []
        self.cuts = []
        refs = []
        for source_id, source in enumerate(self.sources):
            start_day = np.searchsorted(source.dates, np.datetime64(start))
            end_day = np.searchsorted(source.dates, np.datetime64(end))
            if (
                start_day >= len(source.dates)
                or end_day >= len(source.dates)
                or source.dates[start_day] != np.datetime64(start)
                or source.dates[end_day] != np.datetime64(end)
            ):
                raise ValueError(f"{source.name}: requested period is not covered")
            days = np.arange(start_day + 210, end_day - 10 + 1)
            calendar = pd.DatetimeIndex(source.dates[days])
            keep = (
                (calendar.month == 12)
                | (calendar.month < 5)
                | ((calendar.month == 5) & (calendar.day <= 30))
            )
            days = days[keep]
            # Store by absolute issue index to keep __getitem__ small and deterministic.
            indices = history_indices(source.dates, days)
            self.windows.append(
                {
                    int(day): tuple(scale[i] for scale in indices)
                    for i, day in enumerate(days)
                }
            )
            self.cuts.append(
                start_day + int(np.floor((end_day - start_day + 1) * 0.75))
            )
            for day in days:
                targets = source.targets[day + 1 : day + 11, :, 0]
                valid = (
                    source.target_mask[day + 1 : day + 11, :, 0] > 0
                ) & np.isfinite(targets)
                refs.extend(
                    (source_id, int(day), int(site))
                    for site in np.flatnonzero(valid.all(0))
                )
        self.refs = np.asarray(refs, dtype=np.int64).reshape(-1, 3)
        if not len(self.refs):
            raise ValueError("No complete forecast windows in the requested period")

    def __len__(self):
        return len(self.refs)

    def __getitem__(self, index):
        source_id, day, site = self.refs[index]
        source = self.sources[source_id]
        item = {
            "static": source.static[site].copy(),
            "prior": source.prior[site].copy(),
            "target": np.array(
                source.targets[day + 1 : day + 11, site, 0], dtype=np.float32
            ),
            "target_mask": np.ones(10, dtype=np.float32),
            "ref": self.refs[index],
        }
        for name, days in zip(
            ("short", "long", "analog"), self.windows[source_id][int(day)]
        ):
            values = np.asarray(
                source.inputs[np.maximum(days, 0), site][:, source.feature_indices],
                dtype=np.float32,
            )
            mask = (
                source.feature_mask[np.maximum(days, 0), site][
                    :, source.feature_indices
                ]
                > 0
            ) & (days[:, None] >= 0)
            if (mask & ~np.isfinite(values)).any():
                raise ValueError(f"{source.name}: non-finite value marked observed")
            item[f"{name}_raw"] = np.where(mask, values, 0).astype(np.float32)
            item[f"{name}_mask"] = mask.astype(np.float32)
        return item

    def split(self, protocol):
        if protocol not in PROTOCOLS:
            raise ValueError(f"Unknown protocol: {protocol}")
        source_ids, days, sites = self.refs.T
        if protocol == "temporal":
            validation = days >= np.asarray(self.cuts)[source_ids]
        elif protocol.startswith("cs_"):
            heldout = dict(zip(PROTOCOLS[1:5], SOURCES))[protocol]
            validation = np.array([self.sources[i].name == heldout for i in source_ids])
        else:
            high = np.array(
                [
                    self.sources[i].static[s, 0] >= 2400
                    for i, s in zip(source_ids, sites)
                ]
            )
            validation = high if protocol == "ce_high" else ~high
        train, val = np.flatnonzero(~validation), np.flatnonzero(validation)
        if not len(train) or not len(val):
            raise ValueError(f"{protocol}: empty training or validation split")
        return train, val


class Normalizer:
    """Masked min-max statistics fitted only on coordinates visible to training windows."""

    def __init__(self, state):
        self.state = state

    @classmethod
    def fit(cls, dataset, train_indices):
        refs = dataset.refs[np.asarray(train_indices, dtype=int)]
        if not len(refs):
            raise ValueError("Cannot fit normalization without training windows")
        low = np.full(20, np.inf)
        high = np.full(20, -np.inf)
        for i, source in enumerate(dataset.sources):
            selected = refs[refs[:, 0] == i]
            if not len(selected):
                continue
            used = np.zeros(source.inputs.shape[:2], dtype=bool)
            for _, day, site in selected:
                for indices in dataset.windows[i][int(day)]:
                    used[indices[indices >= 0], site] = True
            for j, column in enumerate(source.feature_indices):
                values = source.inputs[:, :, column]
                valid = used & (source.feature_mask[:, :, column] > 0)
                if (valid & ~np.isfinite(values)).any():
                    raise ValueError("Training data has non-finite observed values")
                if valid.any():
                    low[j] = min(low[j], float(values[valid].min()))
                    high[j] = max(high[j], float(values[valid].max()))
            static = source.static[np.unique(selected[:, 2])]
            low[15:] = np.minimum(low[15:], static.min(0))
            high[15:] = np.maximum(high[15:], static.max(0))
        required = np.r_[MINMAX_DYNAMIC, 15:18]
        if (
            not np.isfinite(low[required]).all()
            or not np.isfinite(high[required]).all()
        ):
            raise ValueError("A min-max feature has no observed training values")
        low[~np.isfinite(low)] = 0
        high[~np.isfinite(high)] = 0
        return cls(
            {
                "minimum": low.tolist(),
                "maximum": high.tolist(),
                "fit_samples": int(len(refs)),
            }
        )

    def inputs(self, batch, device="cpu"):
        low = torch.tensor(self.state["minimum"], dtype=torch.float32, device=device)
        # Compute each denominator before the FP32 cast, as in the study pipeline.
        ranges = np.asarray(self.state["maximum"]) - np.asarray(self.state["minimum"])
        denominator = torch.tensor(ranges + 1e-8, dtype=torch.float32, device=device)
        usable = torch.tensor(ranges > 1e-8, device=device)

        def normalize(raw, positions, selected):
            result = raw.clone()
            values = torch.where(
                usable[positions],
                (raw - low[positions]) / denominator[positions],
                torch.zeros_like(raw),
            )
            result[..., selected] = values[..., selected]
            return result

        histories = []
        for name in ("short", "long", "analog"):
            raw = batch[f"{name}_raw"].to(device, dtype=torch.float32)
            mask = batch[f"{name}_mask"].to(device, dtype=torch.float32)
            norm = normalize(raw, slice(0, 15), MINMAX_DYNAMIC.tolist())
            norm = torch.where(mask > 0, norm, torch.zeros_like(norm))
            histories.append(ObservedHistory(norm, raw, mask))
        static = normalize(
            batch["static"].to(device, dtype=torch.float32), slice(15, 20), [0, 1, 2]
        )
        return ReSTInput(
            *histories, static, batch["prior"].to(device, dtype=torch.float32)
        )
