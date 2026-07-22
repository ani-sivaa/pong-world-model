"""evalutils — plotting, GIF, and metrics helpers for the Pong world-model project.

Usage: `from evalutils import plots, gifs, metrics`.

Importing this package forces the matplotlib Agg backend (via evalutils.plots),
so it is safe on headless machines and never opens a window.
"""
from . import plots, gifs, metrics

__all__ = ["plots", "gifs", "metrics"]
