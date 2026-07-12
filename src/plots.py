"""Shared figure identity. Edit the palette here to restyle every figure at once.

Every plot goes through ``_style()`` (global rcParams), draws sequential data with
``SEQ`` and categorical senses with ``SENSE_COLORS``, and is written with ``save``.
"""

from __future__ import annotations

import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

# --- Paper visual identity -------------------------------------------------- #
PAPER = "#fcfcfa"      # background (warm near-white)
INK = "#181817"        # titles, primary lines
INK_SOFT = "#54534f"   # axis labels
MUTED = "#8c8a83"      # tick labels
GRID = "#e7e6df"       # gridlines
EDGE = "#cdccc3"       # spines / axis frame
ACCENT = "#2f6f6a"     # signature hue (deep teal)

# Signature sequential colormap (low -> high = dark navy -> teal -> warm gold).
SEQ = LinearSegmentedColormap.from_list("wsd", [
    (0.00, "#12263a"), (0.28, "#1f5673"), (0.54, "#2f8f88"),
    (0.78, "#83bf82"), (1.00, "#f0d68c")])

# Categorical palette for senses (muted, distinct, print-legible).
SENSE_COLORS = ["#3d5a80", "#e07a5f", "#5b8c5a", "#9b5c8f", "#c9a227",
                "#3a9188", "#8a6d3b", "#c98a9a", "#6b7089", "#98a869"]


def _style() -> None:
    mpl.rcParams.update({
        "figure.facecolor": PAPER,
        "axes.facecolor": PAPER,
        "savefig.facecolor": PAPER,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "figure.dpi": 120,
        "pdf.fonttype": 42,           # editable text in vector PDFs
        "ps.fonttype": 42,
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Times", "STIXGeneral"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 11,
        "text.color": INK,
        "axes.titlesize": 12.5,
        "axes.titlecolor": INK,
        "axes.labelsize": 11,
        "axes.labelcolor": INK_SOFT,
        "axes.edgecolor": EDGE,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.axisbelow": True,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.alpha": 0.8,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.fontsize": 9,
        "legend.frameon": False,
        "lines.linewidth": 1.8,
    })


def style_axes3d(ax) -> None:
    """Elegant panes/grid for a 3-D axes: faint paper walls, hairline grid."""
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor(PAPER)
        axis.pane.set_edgecolor(GRID)
        axis.pane.set_alpha(1.0)
        axis._axinfo["grid"].update(color=GRID, linewidth=0.5)
    ax.tick_params(colors=MUTED)


def save(fig, path, dpi: int = 300) -> None:
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=PAPER)
    import matplotlib.pyplot as plt
    plt.close(fig)
