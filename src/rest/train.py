"""Train ReST on one manuscript protocol."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from . import ReST, ReSTConfig
from .data import PROTOCOLS, Normalizer, WindowDataset
from .metrics import metric_dict


def masked_mae(prediction, target, mask):
    valid = (mask > 0) & torch.isfinite(target)
    if not valid.any():
        raise ValueError("Batch contains no valid targets")
    if not torch.isfinite(prediction[valid]).all():
        raise FloatingPointError("Non-finite prediction")
    return (prediction[valid] - target[valid]).abs().mean()


def run_epoch(model, loader, normalizer, device, optimizer=None):
    model.train(optimizer is not None)
    losses, count, targets, predictions = (0.0, 0, [], [])
    with torch.set_grad_enabled(optimizer is not None):
        for batch in loader:
            target = batch["target"].to(device)
            mask = batch["target_mask"].to(device)
            output = model(normalizer.inputs(batch, device))
            # Preserve gradients below zero; evaluation uses projected SWE.
            loss = (
                masked_mae(output.raw_prediction, target, mask)
                + 0.01 * output.router_load_balance_loss
            )
            if optimizer is not None:
                loss = loss + output.auxiliary_loss
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite training loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0, error_if_nonfinite=True
                )
                optimizer.step()
            losses += loss.detach().item() * len(target)
            count += len(target)
            if optimizer is None:
                targets.append(target.cpu().numpy())
                predictions.append(output.prediction.cpu().numpy())
    metrics = (
        metric_dict(np.concatenate(targets), np.concatenate(predictions))
        if targets
        else {}
    )
    return {"loss": losses / count, **metrics}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--priors", type=Path, required=True)
    parser.add_argument("--protocol", choices=PROTOCOLS, default="temporal")
    parser.add_argument("--seed", type=int, default=52)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch-size must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dataset = WindowDataset(args.data, args.priors)
    train, val = dataset.split(args.protocol)
    normalizer = Normalizer.fit(dataset, train)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        Subset(dataset, train),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )
    val_loader = DataLoader(Subset(dataset, val), batch_size=args.batch_size)
    config = ReSTConfig()
    model = ReST(15, 5, config).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-06
    )
    best_nse, best_loss, stale = (-float("inf"), float("inf"), 0)
    history = []
    for epoch in range(1, args.epochs + 1):
        training = run_epoch(model, train_loader, normalizer, args.device, optimizer)
        validation = run_epoch(model, val_loader, normalizer, args.device)
        if not np.isfinite(validation["loss"]) or not np.isfinite(validation["nse"]):
            raise ValueError(
                "Validation loss/NSE is undefined; inspect the validation targets"
            )
        history.append({"epoch": epoch, "train": training, "validation": validation})
        print(json.dumps(history[-1]), flush=True)
        (args.output / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )
        scheduler.step(validation["loss"])
        if validation["nse"] > best_nse:
            best_nse = validation["nse"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": config.to_dict(),
                    "normalization": normalizer.state,
                    "epoch": epoch,
                    "protocol": args.protocol,
                    "seed": args.seed,
                    "metrics": validation,
                    "train_samples": len(train),
                    "validation_samples": len(val),
                },
                args.output / "best.pt",
            )
        if validation["loss"] < best_loss:
            best_loss, stale = (validation["loss"], 0)
        else:
            stale += 1
        if stale >= 10:
            break


if __name__ == "__main__":
    main()
