"""Evaluate a saved ReST checkpoint without fitting preprocessing or model parameters."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from . import ReST, ReSTConfig
from .data import PROTOCOLS, SOURCES, Normalizer, WindowDataset
from .metrics import metric_dict


def load_checkpoint(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    config = ReSTConfig.from_dict(checkpoint["config"])
    model = ReST(15, 5, config).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model, Normalizer(checkpoint["normalization"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--priors", type=Path, required=True)
    parser.add_argument("--protocol", choices=(*PROTOCOLS, "all"), default="temporal")
    parser.add_argument("--sources", nargs="+", default=SOURCES)
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--end", default="2020-12-31")
    parser.add_argument(
        "--issue-start",
        help="First evaluated initialization date; earlier records remain available as history",
    )
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    model, normalizer = load_checkpoint(args.checkpoint, args.device)
    dataset = WindowDataset(
        args.data, args.priors, sources=args.sources, start=args.start, end=args.end
    )
    indices = (
        np.arange(len(dataset))
        if args.protocol == "all"
        else dataset.split(args.protocol)[1]
    )
    if args.issue_start:
        first = np.datetime64(args.issue_start)
        indices = np.array(
            [
                i
                for i in indices
                if dataset.sources[dataset.refs[i, 0]].dates[dataset.refs[i, 1]]
                >= first
            ],
            dtype=int,
        )
    if not len(indices):
        parser.error("No forecast windows remain after initialization-date filtering")
    prediction, target = [], []
    with torch.inference_mode():
        for batch in DataLoader(Subset(dataset, indices), batch_size=args.batch_size):
            prediction.append(
                model(normalizer.inputs(batch, args.device)).prediction.cpu().numpy()
            )
            target.append(batch["target"].numpy())
    prediction, target = np.concatenate(prediction), np.concatenate(target)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output, prediction=prediction, target=target, refs=dataset.refs[indices]
    )
    print(json.dumps(metric_dict(target, prediction), indent=2))


if __name__ == "__main__":
    main()
