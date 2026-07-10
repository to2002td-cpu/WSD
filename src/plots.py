"""Shared matplotlib styling for the UMAP figures."""

from __future__ import annotations

import matplotlib as mpl
from cycler import cycler

COLORS = {
    "blue_face":   "#b7d4ea",
    "blue_edge":   "#0b3c6d",
    "pink_face":   "#f7c6d9",
    "pink_edge":   "#d81b60",
    "green_face":  "#bfe6dc",
    "green_edge":  "#00695c",
    "orange_face": "#fde4c8",
    "orange_edge": "#b45309",
    "purple_face": "#ddd6fe",
    "purple_edge": "#5b21b6",
    "gray_edge":   "#4d4d4d",
}

PALETTE = [
    COLORS["blue_edge"],
    COLORS["pink_edge"],
    COLORS["green_edge"],
    COLORS["orange_edge"],
    COLORS["purple_edge"],
]

_RC = {
    "figure.dpi": 150,
    "figure.facecolor": "white",
    "figure.constrained_layout.use": True,
    "savefig.facecolor": "white",
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,

    "font.family": "serif",
    "font.serif": [
        "Times New Roman",
        "STIXGeneral",
        "DejaVu Serif",
    ],
    "mathtext.fontset": "stix",
    "font.size": 10,
    "text.color": "#1a1a1a",
    "axes.labelcolor": "#1a1a1a",
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.titlepad": 8,

    "axes.facecolor": "white",
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.axisbelow": True,
    "axes.prop_cycle": cycler(color=PALETTE),

    "axes.grid": True,
    "grid.color": "#e3e3e3",
    "grid.linewidth": 0.6,
    "grid.alpha": 1.0,

    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,

    "lines.linewidth": 2.0,
    "lines.solid_capstyle": "round",
    "lines.solid_joinstyle": "round",
    "scatter.marker": "o",

    "legend.frameon": False,
    "legend.fontsize": 9,
    "legend.handlelength": 1.6,

    "hatch.linewidth": 0.6,
    "image.interpolation": "nearest",
}


def style() -> None:
    mpl.rcParams.update(_RC)