"""Numeric evaluation metrics. numpy only — no torch, no I/O."""
import numpy as np


def mse_curve(real, pred):
    """Per-timestep pixel MSE between two uint8 clips, on [0,1]-normalized pixels.

    Args:
        real: uint8 [T, H, W] ground-truth frames.
        pred: uint8 [T, H, W] predicted frames, same shape as `real`.

    Returns:
        np.float32 array [T] where
        out[t] = mean over H, W of ((real[t]/255 - pred[t]/255) ** 2).
    """
    real = np.asarray(real)
    pred = np.asarray(pred)
    assert real.dtype == np.uint8, f"real must be uint8, got {real.dtype}"
    assert pred.dtype == np.uint8, f"pred must be uint8, got {pred.dtype}"
    assert real.ndim == 3, f"real must be [T,H,W], got shape {real.shape}"
    assert real.shape == pred.shape, (
        f"shape mismatch: real {real.shape} vs pred {pred.shape}"
    )
    diff = real.astype(np.float32) / 255.0 - pred.astype(np.float32) / 255.0
    return (diff * diff).mean(axis=(1, 2)).astype(np.float32)


def summarize_episodes(points, hits, lengths):
    """Aggregate per-episode outcomes into a JSON-serializable summary dict.

    Args:
        points: int array [N], per-episode point from the agent's perspective:
            +1 = agent won the point, -1 = agent conceded, 0 = truncated.
        hits: number array [N], agent paddle hits per episode.
        lengths: number array [N], steps per episode.

    Returns:
        dict with plain python values (json-serializable):
            mean_point (float), win_rate (float, fraction points == +1),
            loss_rate (float, fraction points == -1),
            trunc_rate (float, fraction points == 0),
            mean_hits (float), mean_len (float), n (int).
    """
    points = np.asarray(points)
    hits = np.asarray(hits)
    lengths = np.asarray(lengths)
    assert points.ndim == 1 and points.size > 0, (
        f"points must be a non-empty 1-D array, got shape {points.shape}"
    )
    assert hits.shape == points.shape, (
        f"hits shape {hits.shape} != points shape {points.shape}"
    )
    assert lengths.shape == points.shape, (
        f"lengths shape {lengths.shape} != points shape {points.shape}"
    )
    return {
        "mean_point": float(points.astype(np.float64).mean()),
        "win_rate": float((points == 1).mean()),
        "loss_rate": float((points == -1).mean()),
        "trunc_rate": float((points == 0).mean()),
        "mean_hits": float(hits.astype(np.float64).mean()),
        "mean_len": float(lengths.astype(np.float64).mean()),
        "n": int(points.size),
    }
