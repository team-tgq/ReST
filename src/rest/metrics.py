from __future__ import annotations

import numpy as np


def metric_dict(
    y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray | None = None
) -> dict[str, float]:
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    if mask is None:
        valid = np.isfinite(yt) & np.isfinite(yp)
    else:
        valid = (np.asarray(mask) > 0) & np.isfinite(yt) & np.isfinite(yp)
    if not valid.any():
        return {k: float("nan") for k in ("mae", "rmse", "nse", "r2", "kge", "bias")}

    yt = yt[valid]
    yp = yp[valid]
    err = yp - yt
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    bias = float(np.mean(err))
    denom = float(np.sum((yt - np.mean(yt)) ** 2))
    nse = float(1.0 - np.sum(err**2) / denom) if denom > 0 else float("nan")
    yt_std = float(np.std(yt))
    yp_std = float(np.std(yp))
    if yt.size > 1 and yt_std > 0 and yp_std > 0:
        centered_true = yt - float(np.mean(yt))
        centered_pred = yp - float(np.mean(yp))
        corr = float(np.mean(centered_true * centered_pred) / (yt_std * yp_std))
    else:
        corr = float("nan")
    r2 = corr * corr if np.isfinite(corr) else float("nan")
    alpha = float(yp_std / yt_std) if yt_std > 0 else float("nan")
    beta = (
        float(np.mean(yp) / np.mean(yt)) if abs(np.mean(yt)) > 1e-12 else float("nan")
    )
    kge = (
        float(1.0 - np.sqrt((corr - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2))
        if np.isfinite(corr + alpha + beta)
        else float("nan")
    )
    return {"mae": mae, "rmse": rmse, "nse": nse, "r2": r2, "kge": kge, "bias": bias}
