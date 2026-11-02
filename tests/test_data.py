import json

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader, Subset

from rest.config import DEFAULT_FEATURE_NAMES
from rest.data import PRIOR_NAMES, Normalizer, WindowDataset, history_indices


@pytest.fixture
def dataset_files(tmp_path):
    source = tmp_path / "snotel"
    source.mkdir()
    dates = np.arange("2010-01-01", "2013-01-01", dtype="datetime64[D]")
    values = (
        np.random.default_rng(52).uniform(0, 10, (len(dates), 2, 15)).astype(np.float32)
    )
    targets = values[..., :1].copy()
    for name, data in {
        "dates": dates,
        "inputs": values,
        "feature_mask": np.ones_like(values),
        "targets": targets,
        "target_mask": np.ones_like(targets),
    }.items():
        np.save(source / f"{name}.npy", data)
    pd.DataFrame(
        {
            "site_id": ["a", "b"],
            "elevation": [1000, 3000],
            "lat": [40, 41],
            "lon": [-110, -112],
            "landcover_code": [10, 20],
            "southness": [0.1, -0.2],
        }
    ).to_csv(source / "sites.csv", index=False)
    (source / "meta.json").write_text(
        json.dumps({"feature_names": list(DEFAULT_FEATURE_NAMES), "target_unit": "mm"}),
        encoding="utf-8",
    )
    priors = pd.DataFrame(
        {
            "source": ["snotel", "snotel"],
            "site_id": ["a", "b"],
            **{n: [0.2, 0.2] for n in PRIOR_NAMES},
        }
    )
    priors.to_csv(tmp_path / "priors.csv", index=False)
    return tmp_path


def test_history_is_causal_and_uses_water_year_day():
    dates = np.arange("2018-01-01", "2022-01-01", dtype="datetime64[D]")
    issue = int(np.searchsorted(dates, np.datetime64("2020-03-01")))
    short, long, analog = history_indices(dates, [issue])
    assert short.shape == long.shape == (1, 30) and analog.shape == (1, 75)
    assert short[0, -1] == issue and long[0, -1] == issue - 7
    assert long[0, 0] == issue - 210
    assert (analog[analog >= 0] < issue).all()
    # Leap year alignment is by hydrological day; last year's center is March 2.
    assert dates[analog[0, -8]] == np.datetime64("2019-03-02")
    assert (analog[0, :45] == -1).all()


def test_splits_normalization_and_complete_targets(dataset_files):
    dataset = WindowDataset(
        dataset_files,
        dataset_files / "priors.csv",
        sources=["snotel"],
        end="2012-12-31",
    )
    train, val = dataset.split("ce_high")
    assert (dataset.refs[train, 2] == 0).all() and (dataset.refs[val, 2] == 1).all()
    state = Normalizer.fit(dataset, train).state
    assert state["minimum"][15] == state["maximum"][15] == 1000
    batch = next(iter(DataLoader(Subset(dataset, val[:2]), batch_size=2)))
    inputs = Normalizer(state).inputs(batch)
    assert inputs.short.raw.shape == (2, 30, 15)
    assert torch.isfinite(inputs.short.normalized).all()
    old_count = len(dataset)
    mask = np.load(dataset_files / "snotel/target_mask.npy")
    _, day, site = dataset.refs[0]
    mask[day + 1, site, 0] = 0
    # Release the Windows memory mapping before replacing this fixture file.
    dataset.sources[0].target_mask._mmap.close()
    np.save(dataset_files / "snotel/target_mask.npy", mask)
    changed = WindowDataset(
        dataset_files,
        dataset_files / "priors.csv",
        sources=["snotel"],
        end="2012-12-31",
    )
    assert len(changed) < old_count


def test_invalid_prior_and_nondaily_dates_rejected(dataset_files):
    priors = pd.read_csv(dataset_files / "priors.csv")
    priors.loc[0, PRIOR_NAMES[0]] = -1
    priors.to_csv(dataset_files / "bad.csv", index=False)
    with pytest.raises(ValueError, match="prior"):
        WindowDataset(
            dataset_files,
            dataset_files / "bad.csv",
            sources=["snotel"],
            end="2012-12-31",
        )
    dates = np.load(dataset_files / "snotel/dates.npy")
    dates[1] = dates[0]
    np.save(dataset_files / "snotel/dates.npy", dates)
    with pytest.raises(ValueError, match="daily"):
        WindowDataset(
            dataset_files,
            dataset_files / "priors.csv",
            sources=["snotel"],
            end="2012-12-31",
        )
