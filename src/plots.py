


from __future__ import annotations

import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
from cycler import cycler


# ============================================================================
# Paper identity
# ============================================================================

# Neutral white background
PAPER = "#FFFFFF"

# Typography
INK = "#1A1A1A"
INK_SOFT = "#3F4752"
MUTED = "#737A84"

# Structure
GRID = "#ECEEF2"
EDGE = "#B8BEC8"

# Signature accent
#
# Deep desaturated blue that gives every figure a recognizable identity.
ACCENT = "#2D5DA8"


# ============================================================================
# Sequential colormap
# ============================================================================

# Custom perceptually-smooth map
#
# midnight
#      ↓
# royal blue
#      ↓
# teal
#      ↓
# sage
#      ↓
# warm sand
#
# Distinctive enough to become recognizable across the paper while remaining
# suitable for scientific visualization.

SEQ = LinearSegmentedColormap.from_list(
    "paper_seq",
    [
        (0.00, "#1F2940"),
        (0.20, "#2D5DA8"),
        (0.45, "#3C8DAD"),
        (0.72, "#73B89E"),
        (1.00, "#F3E9C9"),
    ],
)


# ============================================================================
# Categorical palette
# ============================================================================

# Based on Okabe-Ito, lightly tuned so the overall paper keeps a coherent
# blue-centered identity.

SENSE_COLORS = [
    "#2D5DA8",  # signature blue
    "#E69F00",  # orange
    "#009E73",  # green
    "#CC79A7",  # purple
    "#D55E00",  # vermillion
    "#56B4E9",  # sky
    "#7A9E3A",  # olive
    "#7A6FBE",  # indigo
    "#8C564B",  # brown
    "#666666",  # grey
]


# ============================================================================
# Global style
# ============================================================================

def _style() -> None:
    """Apply the global paper style."""

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

        "pdf.fonttype": 42,
        "ps.fonttype": 42,

        # ------------------------------------------------------------------
        # Typography
        # ------------------------------------------------------------------

        "font.family": "sans-serif",
        "font.sans-serif": [
            "Arial",
            "Helvetica",
            "DejaVu Sans",
        ],

        "mathtext.fontset": "dejavusans",

        "font.size": 12,

        "text.color": INK,

        "axes.titlesize": 13,
        "axes.titleweight": "semibold",
        "axes.titlecolor": INK,

        "axes.labelsize": 12,
        "axes.labelcolor": INK_SOFT,

        # ------------------------------------------------------------------
        # Axes
        # ------------------------------------------------------------------

        "axes.edgecolor": EDGE,
        "axes.linewidth": 1.0,

        "axes.spines.top": False,
        "axes.spines.right": False,

        "axes.axisbelow": True,

        # ------------------------------------------------------------------
        # Grid
        # ------------------------------------------------------------------

        "axes.grid": True,

        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.alpha": 0.65,

        # ------------------------------------------------------------------
        # Ticks
        # ------------------------------------------------------------------

        "xtick.color": MUTED,
        "ytick.color": MUTED,

        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,

        "xtick.direction": "out",
        "ytick.direction": "out",

        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,

        # ------------------------------------------------------------------
        # Lines
        # ------------------------------------------------------------------

        "lines.linewidth": 2.2,
        "lines.markersize": 6,

        # ------------------------------------------------------------------
        # Legends
        # ------------------------------------------------------------------

        "legend.frameon": False,
        "legend.fontsize": 10,

        "legend.handlelength": 1.8,
        "legend.handletextpad": 0.5,
        "legend.borderaxespad": 0.4,

        # ------------------------------------------------------------------
        # Default color cycle
        # ------------------------------------------------------------------

        "axes.prop_cycle": cycler(color=SENSE_COLORS),
    })


# ============================================================================
# 3D styling
# ============================================================================

def style_axes3d(ax) -> None:
    """
    Make 3D plots visually consistent with the paper style.
    """

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):

        axis.pane.set_facecolor(PAPER)
        axis.pane.set_edgecolor(GRID)
        axis.pane.set_alpha(0.06)

        axis._axinfo["grid"].update(
            color=GRID,
            linewidth=0.5,
        )

    ax.tick_params(colors=MUTED)


# ============================================================================
# Saving
# ============================================================================

def save(fig, path, dpi: int = 400) -> None:
    """
    Save a publication-quality figure.
    """

    fig.savefig(
        path,
        dpi=dpi,
        bbox_inches="tight",
        facecolor=PAPER,
    )

    import matplotlib.pyplot as plt

    plt.close(fig)


# Apply the style on import.
_style()