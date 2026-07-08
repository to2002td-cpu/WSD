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

# Categorical palette for coloring senses in the cloud-split scatter (distinct
# hues, colorblind-mindful): blue, orange, green, red, purple, teal.
CATEGORICAL = ["#2a78d6", "#e0872e", "#3a923a", "#c0504d", "#7b5cc4", "#4aa3a2"]

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


def _pca_2d(x: np.ndarray) -> np.ndarray:
    """Project rows onto their top 2 principal directions (unit-normalized
    first, matching the cosine geometry of the analysis)."""
    x = x.astype(np.float32)
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)
    x = x - x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    return x @ vt[:2].T


def plot_cloud_split(
    df: pd.DataFrame,
    records: "list[dict]",
    emb_dir: Path,
    out: Path,
    n_words: int = 3,
    n_steps: int = 5,
) -> None:
    """The headline figure: does one cloud split into two?

    A grid of 2-D PCA scatters — rows are the words whose senses separate most
    by the final checkpoint (data-selected), columns are checkpoints. Each
    point is one occurrence, colored by its annotated sense. You should see a
    single blob at early steps pull apart into distinct clouds as training
    proceeds.
    """
    meta = pd.DataFrame(records)
    meta["row"] = np.arange(len(meta))

    final = int(df["step"].max())
    fin = df[df["step"] == final]
    # Words that end up most separated, and each word's best-separating layer.
    words = list(
        fin.groupby("word")["silhouette"].max()
        .sort_values(ascending=False).head(n_words).index
    )
    best_layer = {
        w: int(fin[fin["word"] == w].set_index("layer")["silhouette"].idxmax())
        for w in words
    }

    steps = sorted(df["step"].unique())
    if len(steps) > n_steps:
        idx = np.linspace(0, len(steps) - 1, n_steps).round().astype(int)
        steps = [steps[i] for i in idx]

    fig, axes = plt.subplots(
        len(words), len(steps),
        figsize=(2.5 * len(steps), 2.5 * len(words)),
        squeeze=False,
    )
    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)

    word_meta = {w: meta[meta["word"] == w] for w in words}
    for j, step in enumerate(steps):
        npz = emb_dir / f"step{step}.npz"
        if not npz.exists():
            continue
        vectors = np.load(npz)["vectors"]
        valid = np.linalg.norm(vectors[:, -1, :].astype(np.float32), axis=1) > 0
        for i, word in enumerate(words):
            ax = axes[i][j]
            g = word_meta[word]
            g = g[valid[g["row"].to_numpy()]]
            counts = g["sense"].value_counts()
            g = g[g["sense"].isin(counts[counts >= 2].index)]
            if g["sense"].nunique() < 2:
                continue
            xy = _pca_2d(vectors[g["row"].to_numpy(), best_layer[word], :])
            senses = sorted(g["sense"].unique())
            sense_arr = g["sense"].to_numpy()
            for k, s in enumerate(senses):
                m = sense_arr == s
                ax.scatter(xy[m, 0], xy[m, 1], s=16, alpha=0.75,
                           color=CATEGORICAL[k % len(CATEGORICAL)],
                           label=s.split(".", 1)[-1], edgecolors="none")
            if i == 0:
                ax.set_title(f"step {step}", fontsize=10, color=INK)
            if j == 0:
                ax.set_ylabel(f"{word}\n(L{best_layer[word]})", fontsize=9,
                              color=INK, rotation=0, ha="right", va="center",
                              labelpad=18)
            if j == len(steps) - 1:
                ax.legend(fontsize=7, loc="center left",
                          bbox_to_anchor=(1.0, 0.5), handletextpad=0.2)

    fig.suptitle(
        "Do the sense clouds separate over pre-training? (2-D PCA per word)",
        x=0.01, ha="left", color=INK,
    )
    fig.tight_layout(rect=(0.02, 0, 1, 0.96))
    fig.savefig(out, dpi=200)
    plt.close(fig)
    log.info("Wrote %s", out)


def make_all_plots(
    df: pd.DataFrame,
    summary: pd.DataFrame,
    out_dir: Path,
    records: "list[dict]",
    emb_dir: Path,
) -> None:
    _style()
    out_dir.mkdir(parents=True, exist_ok=True)
    # The direct visual answer: one blob or two clouds, over training.
    plot_cloud_split(df, records, emb_dir, out_dir / "cloud_split.png")
    # Aggregate separation as a single number over training.
    plot_metric_vs_step(
        summary, "silhouette_mean",
        "Mean silhouette  (0 = one cloud, 1 = two clouds)",
        out_dir / "separation_vs_step.png",
    )
    plot_metric_vs_step(
        summary, "frac_separated", "Fraction of words with separated senses",
        out_dir / "frac_separated_vs_step.png",
    )
    plot_layer_step_heatmap(
        summary, "silhouette_mean", "Mean silhouette",
        out_dir / "heatmap_silhouette.png",
    )
