from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
from cycler import cycler


# ============================================================================
# Page geometry (NeurIPS / ICML / ICLR)
# ============================================================================

WIDE_WIDTH = 6.75      # a figure* spanning both columns


def fmt_step(s: int) -> str:
    """Compact training-step label: 143000 -> '143k', 512 -> '512'."""
    return f"{s // 1000}k" if s >= 1000 and s % 1000 == 0 else str(s)


# ============================================================================
# Paper identity
# ============================================================================

PAPER = "#FFFFFF"

INK = "#1A1A1A"
INK_SOFT = "#444444"
MUTED = "#6B7280"

GRID = "#E5E7EB"
EDGE = "#9CA3AF"


# ============================================================================
# Colormaps
# ============================================================================

PURITY_CMAP = LinearSegmentedColormap.from_list(
    "purity_div",
    [
        (0.00, "#B23A48"),
        (0.25, "#E7A8A8"),
        (0.50, "#F7F7F5"),
        (0.75, "#9DBCE5"),
        (1.00, "#1F4E8C"),
    ],
)


# ============================================================================
# Categorical colors (per-sense)
#
# Eight CVD-validated hues; assigned to senses 1..N in this fixed order.
# ============================================================================

SENSE_COLORS = [
    "#2563EB",  # blue (sense 1)
    "#D97706",  # orange
    "#059669",  # green
    "#C026D3",  # fuchsia
    "#DC2626",  # red
    "#0891B2",  # cyan
    "#7C3AED",  # purple
    "#65A30D",  # olive
]


# ============================================================================
# Global style
# ============================================================================

def _style() -> None:
    """
    Apply NeurIPS / ICML style globally.

    Design principles:
    - minimal axes
    - strong hierarchy
    - print-safe colors
    - readable at two-column size
    """

    mpl.rcParams.update({

        # ------------------------------------------------------------------
        # Figure
        # ------------------------------------------------------------------

        "figure.facecolor": PAPER,
        "axes.facecolor": PAPER,
        "savefig.facecolor": PAPER,

        "figure.dpi": 150,
        "savefig.dpi": 400,

        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,

        "pdf.fonttype": 42,
        "ps.fonttype": 42,


        # ------------------------------------------------------------------
        # Typography
        # ------------------------------------------------------------------

        "text.usetex": False,

        "font.family": "serif",
        "font.serif": [
            "Times New Roman",
            "Times",
            "DejaVu Serif",
        ],

        "mathtext.fontset": "stix",

        "font.size": 8,

        "text.color": INK,

        "axes.titlesize": 8.5,
        "axes.titleweight": "normal",
        "axes.titlecolor": INK,

        "axes.labelsize": 8,
        "axes.labelcolor": INK_SOFT,


        # ------------------------------------------------------------------
        # Axes
        # ------------------------------------------------------------------

        "axes.edgecolor": EDGE,
        "axes.linewidth": 0.7,

        # NeurIPS style:
        # no top/right box unless explicitly requested
        "axes.spines.top": False,
        "axes.spines.right": False,

        "axes.axisbelow": True,


        # ------------------------------------------------------------------
        # Grid
        # ------------------------------------------------------------------

        "axes.grid": False,

        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.6,


        # ------------------------------------------------------------------
        # Ticks
        # ------------------------------------------------------------------

        "xtick.color": MUTED,
        "ytick.color": MUTED,

        "xtick.labelsize": 7,
        "ytick.labelsize": 7,

        "xtick.direction": "out",
        "ytick.direction": "out",

        "xtick.top": False,
        "ytick.right": False,

        # avoid clutter in small conference figures
        "xtick.minor.visible": False,
        "ytick.minor.visible": False,

        "xtick.major.size": 3,
        "ytick.major.size": 3,

        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,


        # ------------------------------------------------------------------
        # Lines
        # ------------------------------------------------------------------

        "lines.linewidth": 1.6,
        "lines.markersize": 4.5,


        # ------------------------------------------------------------------
        # Legend
        # ------------------------------------------------------------------

        "legend.frameon": False,

        "legend.fontsize": 7,

        "legend.handlelength": 1.8,
        "legend.handletextpad": 0.5,
        "legend.borderaxespad": 0.3,


        # ------------------------------------------------------------------
        # Colors
        # ------------------------------------------------------------------

        "axes.prop_cycle": cycler(
            color=SENSE_COLORS
        ),
    })


# ============================================================================
# Saving
# ============================================================================

def save(fig, path, dpi: int = 400) -> None:
    """
    Save figure as PDF + PNG.

    PDF:
        - vector text
        - LaTeX compatible
        - editable

    PNG:
        - preview/debug
    """

    import matplotlib.pyplot as plt

    path = Path(path)

    for ext in (".pdf", ".png"):

        fig.savefig(
            path.with_suffix(ext),
            dpi=dpi,
            bbox_inches="tight",
            facecolor=PAPER,
        )

    plt.close(fig)


# Apply automatically
_style()