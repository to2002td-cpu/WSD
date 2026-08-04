import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from geometry import load_records, available_steps, layer_list, cached_vectors, standardize_dimensions


def gloss(sense, limit=52):
    from nltk.corpus import wordnet as wn
    try:
        text = wn.synset(sense).definition()
    except Exception:
        return ""
    return text if len(text) <= limit else text[:limit - 3] + "..."


def step_label(step):
    if step >= 1000:
        return str(step // 1000) + "k"
    return str(step)


def lemma_rows(meta, word, pos):
    group = meta[(meta["word"] == word) & (meta["pos"] == pos)]
    if group.empty:
        raise SystemExit("no rows for " + word + " (" + pos + ")")
    return group.reset_index(drop=True)


def load_lemma_vectors(emb_dir, step, rows, all_layers, layers, cache_dir=None):
    vectors = cached_vectors(emb_dir, step, cache_dir)
    indices = rows["row"].to_numpy()
    block = np.asarray(vectors[indices]).astype(np.float32)
    return {layer: block[:, all_layers.index(layer), :] for layer in layers}


def prepare(matrix, center=True, standardize=False):
    keep = np.linalg.norm(matrix, axis=1) > 0
    x = matrix[keep]
    if center:
        x = x - x.mean(axis=0, keepdims=True)
    if standardize:
        x = standardize_dimensions(x)
    return x, keep


def pca_grid(emb_dir, dataset_path, word, pos, out_path, steps=None, layers=None, point_size=4, cache_dir=None, standardize=True):
    meta = load_records(dataset_path)
    meta["row"] = np.arange(len(meta))
    rows = lemma_rows(meta, word, pos)

    steps = steps or available_steps(emb_dir)
    all_layers = layer_list(emb_dir, steps[0])
    layers = layers or all_layers

    senses = sorted(rows["sense"].unique())
    palette = plt.get_cmap("tab10")
    colours = {s: palette(i % 10) for i, s in enumerate(senses)}

    fig, axes = plt.subplots(len(layers), len(steps),
                             figsize=(1.75 * len(steps), 1.75 * len(layers)),
                             squeeze=False)

    for col, step in enumerate(steps):
        print("pca, step", step, flush=True)
        picked = load_lemma_vectors(emb_dir, step, rows, all_layers, layers, cache_dir)
        for line, layer in enumerate(layers):
            ax = axes[line][col]
            x, keep = prepare(picked[layer], standardize=standardize)
            labels = rows["sense"].to_numpy()[keep]
            x = x - x.mean(axis=0, keepdims=True)
            u, s, _ = np.linalg.svd(x, full_matrices=False)
            coords = u[:, :2] * s[:2]
            explained = (s[:2] ** 2).sum() / (s ** 2).sum()
            for sense in senses:
                mask = labels == sense
                ax.scatter(coords[mask, 0], coords[mask, 1], s=point_size,
                           color=colours[sense], linewidths=0, alpha=0.6)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.text(0.03, 0.95, str(int(round(100 * explained))) + "%", transform=ax.transAxes,
                    fontsize=6, va="top")
            if col == 0:
                ax.set_ylabel(str(layer), rotation=0, labelpad=12, fontsize=9)
            if line == 0:
                ax.set_title(step_label(step), fontsize=9)
        del picked

    handles = [plt.Line2D([], [], marker="o", linestyle="none", color=colours[s],
                          label=s + "  " + gloss(s)) for s in senses]
    ncol = 2 if len(senses) > 4 else 1
    fig.legend(handles=handles, loc="lower center", ncol=ncol, fontsize=7, frameon=False,
               bbox_to_anchor=(0.5, -0.02 - 0.02 * len(senses) / ncol))
    fig.suptitle(word + " (" + pos + ")   training step", fontsize=11)
    fig.text(0.02, 0.5, "layer", rotation=90, va="center", fontsize=10)
    fig.tight_layout(rect=[0.03, 0.06 + 0.015 * len(senses) / ncol, 1, 0.96])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def purity_matrix(x, labels, k_values):
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)
    similarity = x @ x.T
    np.fill_diagonal(similarity, -np.inf)
    order = np.argsort(-similarity, axis=1)
    same = labels[order] == labels[:, None]
    return [float(same[:, :k].mean()) for k in k_values]


def purity_grid(emb_dir, dataset_path, word, pos, out_path, steps=None, layers=None,
                k_values=(5, 20, 50, 100, 200, 500), cache_dir=None, standardize=True):
    meta = load_records(dataset_path)
    meta["row"] = np.arange(len(meta))
    rows = lemma_rows(meta, word, pos)

    steps = steps or available_steps(emb_dir)
    all_layers = layer_list(emb_dir, steps[0])
    layers = layers or all_layers
    k_values = [k for k in k_values if k < len(rows)]

    grids = {}
    for step in steps:
        print("purity, step", step, flush=True)
        picked = load_lemma_vectors(emb_dir, step, rows, all_layers, layers, cache_dir)
        grid = np.zeros((len(layers), len(k_values)))
        for line, layer in enumerate(layers):
            x, keep = prepare(picked[layer], standardize=standardize)
            labels = rows["sense"].to_numpy()[keep]
            grid[line] = purity_matrix(x, labels, k_values)
        grids[step] = grid
        del picked

    columns = int(np.ceil(len(steps) / 2))
    fig, axes = plt.subplots(2, columns, figsize=(2.1 * columns, 6.0), squeeze=False)
    fig.subplots_adjust(hspace=0.55)
    for i, step in enumerate(steps):
        ax = axes[i // columns][i % columns]
        image = ax.pcolormesh(grids[step], cmap="coolwarm_r", vmin=0.0, vmax=1.0)
        ax.set_title("step " + step_label(step), fontsize=9)
        ax.set_xticks(np.arange(len(k_values)) + 0.5)
        ax.set_xticklabels([str(k) for k in k_values], fontsize=7, rotation=45)
        ax.set_yticks(np.arange(len(layers)) + 0.5)
        ax.set_yticklabels([str(l) for l in layers], fontsize=7)
        if i % columns != 0:
            ax.set_yticklabels([])
    for j in range(len(steps), 2 * columns):
        axes[j // columns][j % columns].axis("off")

    fig.supxlabel("k", fontsize=10)
    fig.supylabel("layer", fontsize=10)
    fig.suptitle(word + " (" + pos + ")   k-NN sense purity", fontsize=11)
    fig.colorbar(image, ax=axes, shrink=0.7, label="k-NN sense purity")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    return out_path


def emergence_curves(table, out_path, metric="excess_k10", layer=16, top=12):
    data = table[table["layer"] == layer]
    order = data[data["step"] == data["step"].max()].sort_values(metric, ascending=False)
    words = order.head(top)[["word", "pos"]].apply(tuple, axis=1).tolist()

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for word, pos in words:
        series = data[(data["word"] == word) & (data["pos"] == pos)].sort_values("step")
        steps = series["step"].to_numpy()
        steps = np.where(steps == 0, 1, steps)
        ax.plot(steps, series[metric].to_numpy(), marker="o", markersize=3, label=word + "." + pos)
    ax.set_xscale("log")
    ax.set_xlabel("training step")
    ax.set_ylabel(metric)
    ax.set_title("sense separation over pretraining, layer " + str(layer))
    ax.legend(fontsize=7, ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
