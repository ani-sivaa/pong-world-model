"""Headless line-plot helpers.

Forces the Agg backend BEFORE pyplot is imported, saves PNGs at dpi=150,
creates parent directories, closes every figure after saving (safe in loops),
and never calls plt.show().
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # must precede the pyplot import — keeps everything headless
import matplotlib.pyplot as plt  # noqa: E402  (deliberately after use("Agg"))


def line_plot(xs, ys, title, xlabel, ylabel, out_path, log_y=False):
    """Plot one or more series against a shared x-axis and save a single PNG.

    Args:
        xs: sequence of length N — shared x values for every series.
        ys: non-empty dict[str, sequence] mapping legend label -> y values
            (each of length N, matching xs).
        title: figure title (str).
        xlabel: x-axis label (str).
        ylabel: y-axis label (str).
        out_path: str or Path for the output PNG; parent dirs are created.
        log_y: if True, use a logarithmic y-axis.

    Returns:
        Path(out_path).
    """
    assert isinstance(ys, dict) and len(ys) > 0, (
        f"ys must be a non-empty dict of label -> sequence, got {type(ys).__name__}"
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    xs = list(xs)
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, y in ys.items():
        y = list(y)
        assert len(y) == len(xs), (
            f"series {label!r} has length {len(y)} but xs has length {len(xs)}"
        )
        ax.plot(xs, y, label=str(label))
    if log_y:
        ax.set_yscale("log")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
