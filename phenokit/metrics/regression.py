from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Union, Dict, Any

import numpy as np

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from scipy.stats import pearsonr, spearmanr


ArrayLike = Union[Sequence[float], np.ndarray]


def _to_1d_array(x: ArrayLike, name: str) -> np.ndarray:
    """Convert input to 1D numpy array (float)."""
    arr = np.asarray(x, dtype=float)

    # squeeze and flatten
    arr = np.squeeze(arr)

    if arr.ndim != 1:
        # If user passes (n,1) or (1,n), squeeze should handle. Others are not supported here.
        raise ValueError(f"{name} must be 1D array-like. Got shape={arr.shape}")

    return arr


def _validate_pair(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError(f"y_true and y_pred must have same length. "
                         f"Got {y_true.shape[0]} vs {y_pred.shape[0]}")

    if y_true.shape[0] == 0:
        raise ValueError("y_true and y_pred must be non-empty.")

    # Check NaN/Inf
    if not np.isfinite(y_true).all():
        raise ValueError("y_true contains NaN or Inf.")
    if not np.isfinite(y_pred).all():
        raise ValueError("y_pred contains NaN or Inf.")


def regression_metrics(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    metrics: Optional[Iterable[str]] = None,
    sample_weight: Optional[ArrayLike] = None,
    verbose: bool = False,
) -> Dict[str, float]:
    """
    Compute regression evaluation metrics.

    Parameters
    ----------
    y_true : array-like
        Ground truth values (1D).
    y_pred : array-like
        Predicted values (1D).
    metrics : iterable of str, optional
        Which metrics to compute. Supported:
        {"mse","rmse","mae","r2","pearson_r","pearson_p","spearman_r","spearman_p"}.
        If None, compute all.
    sample_weight : array-like, optional
        Sample weights used for mse/mae/r2 (sklearn metrics). Correlation metrics ignore weights.
    verbose : bool
        If True, print metrics nicely.

    Returns
    -------
    dict
        Metric name -> value
    """
    y_t = _to_1d_array(y_true, "y_true")
    y_p = _to_1d_array(y_pred, "y_pred")
    _validate_pair(y_t, y_p)

    if sample_weight is not None:
        w = _to_1d_array(sample_weight, "sample_weight")
        if w.shape[0] != y_t.shape[0]:
            raise ValueError(f"sample_weight length must match y_true. "
                             f"Got {w.shape[0]} vs {y_t.shape[0]}")
        if not np.isfinite(w).all():
            raise ValueError("sample_weight contains NaN or Inf.")
    else:
        w = None

    supported = {
        "mse", "rmse", "mae", "r2",
        "pearson_r", "pearson_p",
        "spearman_r", "spearman_p",
    }

    if metrics is None:
        metrics = supported
    else:
        metrics = set(metrics)
        unknown = metrics - supported
        if unknown:
            raise ValueError(f"Unknown metrics: {sorted(unknown)}. Supported: {sorted(supported)}")

    out: Dict[str, float] = {}

    # sklearn metrics (support sample_weight)
    if "mse" in metrics or "rmse" in metrics:
        mse = mean_squared_error(y_t, y_p, sample_weight=w)
        if "mse" in metrics:
            out["mse"] = float(mse)
        if "rmse" in metrics:
            out["rmse"] = float(np.sqrt(mse))

    if "mae" in metrics:
        out["mae"] = float(mean_absolute_error(y_t, y_p, sample_weight=w))

    if "r2" in metrics:
        out["r2"] = float(r2_score(y_t, y_p, sample_weight=w))

    # correlations (ignore sample_weight)
    need_pearson = ("pearson_r" in metrics) or ("pearson_p" in metrics)
    if need_pearson:
        r, p = pearsonr(y_t, y_p)
        if "pearson_r" in metrics:
            out["pearson_r"] = float(r)
        if "pearson_p" in metrics:
            out["pearson_p"] = float(p)

    need_spearman = ("spearman_r" in metrics) or ("spearman_p" in metrics)
    if need_spearman:
        rho, p2 = spearmanr(y_t, y_p)
        if "spearman_r" in metrics:
            out["spearman_r"] = float(rho)
        if "spearman_p" in metrics:
            out["spearman_p"] = float(p2)

    if verbose:
        # 固定顺序打印
        order = ["mse", "rmse", "mae", "r2", "pearson_r", "pearson_p", "spearman_r", "spearman_p"]
        for k in order:
            if k in out:
                print(f"{k:>12}: {out[k]}")

    return out
