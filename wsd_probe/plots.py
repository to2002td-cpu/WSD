"""Static matplotlib figures for the sense-separation analysis.

Design rules applied throughout (from the dataviz method):
  * layer index is an *ordered* quantity -> single-hue blue ramp, light->dark
    (never a cycled categorical palette, never a rainbow);
  * one y-axis per figure; recessive hairline grid; no top/right spines;
  * a legend whenever more than one series is shown;
  * text in ink colors, never in series colors.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

log = logging.getLogger(__name__)

# Reference palette (light mode) — swap here to re-brand every figure.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
# Blue sequential ramp; ordinal usage starts at step 250 so the lightest
# mark still clears 2:1 contrast on the light surface.
BLUE_SEQ = ["#cde2fb", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
            "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#0d366b"]
BLUE_ORDINAL = BLUE_SEQ[2:]

CMAP_SEQ = LinearSegmentedColormap.from_list("blue_seq", BLUE_SEQ)


def _style() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "text.color": INK,
            "axes.labelcolor": INK_2,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.edgecolor": BASELINE,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "font.family": "sans-serif",
            "font.size": 10,
            "axes.titlesize": 11,
            "legend.frameon": False,
            "lines.linewidth": 2.0,
        }
    )


def _layer_colors(layers: "list[int]") -> "dict[int, str]":
    """Map ordered layer indices onto the ordinal blue ramp (light->dark)."""
    ramp = LinearSegmentedColormap.from_list("blue_ord", BLUE_ORDINAL)
    if len(layers) == 1:
        return {layers[0]: BLUE_ORDINAL[-3]}
    return {
        l: mpl.colors.to_hex(ramp(i / (len(layers) - 1)))
        for i, l in enumerate(sorted(layers))
    }


def _step_axis(ax: plt.Axes) -> None:
    """Checkpoint steps are log-spaced and include 0 -> symlog x-axis."""
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlabel("Pre-training step (symlog)")


def _pick_layers(all_layers: "list[int]", k: int = 6) -> "list[int]":
    """Evenly spaced subset of layers so line charts stay readable."""
    if len(all_layers) <= k:
        return sorted(all_layers)
    idx = np.linspace(0, len(all_layers) - 1, k).round().astype(int)
    return [sorted(all_layers)[i] for i in idx]


def plot_metric_vs_step(
    summary: pd.DataFrame, metric: str, ylabel: str, out: Path
) -> None:
    """Aggregate separation metric across checkpoints, one line per layer."""
    layers = _pick_layers(sorted(summary["layer"].unique()))
    colors = _layer_colors(layers)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for layer in layers:
        d = summary[summary["layer"] == layer].sort_values("step")
        ax.plot(d["step"], d[metric], color=colors[layer],
                label=f"layer {layer}")
    _step_axis(ax)
    ax.set_ylabel(ylabel)
    ax.set_title(f"Sense separation across pre-training — {ylabel}",
                 loc="left", color=INK)
    ax.legend(title="", ncols=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    log.info("Wrote %s", out)


def plot_layer_step_heatmap(
    summary: pd.DataFrame, metric: str, label: str, out: Path
) -> None:
    """Layer x checkpoint heatmap of the aggregate separation metric."""
    pivot = summary.pivot(index="layer", columns="step", values=metric)
    fig, ax = plt.subplots(
        figsize=(0.45 * len(pivot.columns) + 2.5, 0.22 * len(pivot) + 2)
    )
    ax.grid(False)
    im = ax.imshow(pivot.values, aspect="auto", cmap=CMAP_SEQ,
                   origin="lower", interpolation="nearest")
    ax.set_xticks(range(len(pivot.columns)),
                  [str(s) for s in pivot.columns], rotation=45,
                  ha="right", fontsize=8)
    ax.set_yticks(range(len(pivot.index)), [str(l) for l in pivot.index],
                  fontsize=8)
    ax.set_xlabel("Pre-training step")
    ax.set_ylabel("Layer")
    ax.set_title(f"{label} by layer and checkpoint", loc="left", color=INK)
    fig.colorbar(im, ax=ax, label=label, shrink=0.85)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    log.info("Wrote %s", out)


def plot_per_word(
    df: pd.DataFrame, out: Path, metric: str = "silhouette", top_k: int = 12
) -> None:
    """Small multiples: best-layer separation trajectory for the words whose
    final-checkpoint separation is highest (chosen from the data, not a
    priori)."""
    final_step = df["step"].max()
    best = (
        df[df["step"] == final_step]
        .groupby("word")[metric]
        .max()
        .sort_values(ascending=False)
        .head(top_k)
    )
    words = list(best.index)
    ncols = 4
    nrows = int(np.ceil(len(words) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.1 * ncols, 2.3 * nrows),
        sharex=True, sharey=True, squeeze=False,
    )
    for ax in axes.flat:
        ax.set_visible(False)
    for i, word in enumerate(words):
        ax = axes[i // ncols][i % ncols]
        ax.set_visible(True)
        d = df[df["word"] == word]
        # Per word, plot the layer that ends up most separating that word:
        # the layer is data-selected, per word.
        best_layer = d[d["step"] == final_step].set_index("layer")[metric].idxmax()
        traj = d[d["layer"] == best_layer].sort_values("step")
        ax.plot(traj["step"], traj[metric], color=BLUE_ORDINAL[-4])
        _step_axis(ax)
        ax.set_title(f"{word} (L{best_layer})", loc="left", fontsize=9,
                     color=INK)
        if i % ncols == 0:
            ax.set_ylabel(metric)
        if i // ncols < nrows - 1:
            ax.set_xlabel("")
    fig.suptitle(
        f"Per-word sense separation ({metric}, best layer per word)",
        x=0.01, ha="left", color=INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=200)
    plt.close(fig)
    log.info("Wrote %s", out)


def make_all_plots(df: pd.DataFrame, summary: pd.DataFrame, out_dir: Path) -> None:
    _style()
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_metric_vs_step(
        summary, "silhouette_mean", "Mean silhouette (cosine)",
        out_dir / "silhouette_vs_step.png",
    )
    plot_metric_vs_step(
        summary, "ratio_mean", "Mean inter/intra distance ratio",
        out_dir / "ratio_vs_step.png",
    )
    plot_metric_vs_step(
        summary, "frac_significant", "Fraction of words with p < 0.05",
        out_dir / "significance_vs_step.png",
    )
    plot_layer_step_heatmap(
        summary, "silhouette_mean", "Mean silhouette",
        out_dir / "heatmap_silhouette.png",
    )
    plot_layer_step_heatmap(
        summary, "frac_significant", "Fraction significant (p < 0.05)",
        out_dir / "heatmap_significance.png",
    )
    plot_per_word(df, out_dir / "per_word_trajectories.png")
