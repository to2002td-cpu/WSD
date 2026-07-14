"""Shared figure style for the paper (NeurIPS/ICML two-column): the palette,
purity colormap, and global matplotlib rcParams, applied on import."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
from cycler import cycler
from matplotlib.colors import LinearSegmentedColormap

WIDE_WIDTH = 6.75      # inches: a figure* spanning both columns

PAPER = "#FFFFFF"
INK = "#1A1A1A"
INK_SOFT = "#444444"
MUTED = "#6B7280"
GRID = "#E5E7EB"
EDGE = "#9CA3AF"

# Diverging red -> neutral -> blue, keyed to purity 0 -> chance -> 1.
PURITY_CMAP = LinearSegmentedColormap.from_list("purity_div", [
    (0.00, "#B23A48"), (0.25, "#E7A8A8"), (0.50, "#F7F7F5"),
    (0.75, "#9DBCE5"), (1.00, "#1F4E8C"),
])

# Per-sense hues (CVD-safe), assigned to senses 1..N in this fixed order.
SENSE_COLORS = [
    "#2563EB", "#D97706", "#059669", "#C026D3",
    "#DC2626", "#0891B2", "#7C3AED", "#65A30D",
]


def fmt_step(s: int) -> str:
    """Compact training-step label: 143000 -> '143k', 512 -> '512'."""
    return f"{s // 1000}k" if s >= 1000 and s % 1000 == 0 else str(s)


def _style() -> None:
    """Minimal axes, serif type, print-safe colors, readable at column size."""
    mpl.rcParams.update({
        "figure.facecolor": PAPER, "axes.facecolor": PAPER, "savefig.facecolor": PAPER,
        "figure.dpi": 150, "savefig.dpi": 400,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42, "ps.fonttype": 42,                 # editable vector text
        "text.usetex": False,
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8, "text.color": INK,
        "axes.titlesize": 8.5, "axes.titleweight": "normal", "axes.titlecolor": INK,
        "axes.labelsize": 8, "axes.labelcolor": INK_SOFT,
        "axes.edgecolor": EDGE, "axes.linewidth": 0.7,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.axisbelow": True, "axes.grid": False,
        "grid.color": GRID, "grid.linewidth": 0.5, "grid.alpha": 0.6,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.top": False, "ytick.right": False,
        "xtick.minor.visible": False, "ytick.minor.visible": False,
        "xtick.major.size": 3, "ytick.major.size": 3,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "lines.linewidth": 1.6, "lines.markersize": 4.5,
        "legend.frameon": False, "legend.fontsize": 7,
        "legend.handlelength": 1.8, "legend.handletextpad": 0.5, "legend.borderaxespad": 0.3,
        "axes.prop_cycle": cycler(color=SENSE_COLORS),
    })


def save(fig, path, dpi: int = 400) -> None:
    """Write the figure as both PDF (vector, for the paper) and PNG (preview)."""
    import matplotlib.pyplot as plt

    path = Path(path)
    for ext in (".pdf", ".png"):
        fig.savefig(path.with_suffix(ext), dpi=dpi, bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)


_style()