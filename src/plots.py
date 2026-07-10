"""Shared matplotlib styling for the UMAP figures."""

from __future__ import annotations

import matplotlib as mpl

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"


def _style() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "figure.dpi": 120,
            "pdf.fonttype": 42,       # keep text editable (TrueType) in vector PDFs
            "ps.fonttype": 42,
            "text.color": INK,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.edgecolor": BASELINE,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.9,
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "Times", "STIXGeneral"],
            "mathtext.fontset": "dejavuserif",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "legend.frameon": False,
            "lines.linewidth": 2.0,
        }
    )