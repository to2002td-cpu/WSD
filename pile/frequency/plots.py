import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def share_distribution(table, out_path, tail_marks=(0.05, 0.02)):
    shares = np.sort(table["share"].values)[::-1]
    ranks = np.arange(1, len(shares) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].loglog(ranks, shares, marker=".", linestyle="none", markersize=3)
    axes[0].set_xlabel("synset rank")
    axes[0].set_ylabel("share within lemma")
    axes[0].set_title("sense shares, ranked")
    for mark in tail_marks:
        axes[0].axhline(mark, linewidth=0.8, linestyle="--", color="grey")

    axes[1].hist(np.log10(shares), bins=40)
    axes[1].set_xlabel("log10 share within lemma")
    axes[1].set_ylabel("number of synsets")
    axes[1].set_title("distribution of sense shares")
    for mark in tail_marks:
        axes[1].axvline(np.log10(mark), linewidth=0.8, linestyle="--", color="grey")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def lemma_composition(table, lemma, pos, out_path, top_n=10):
    rows = table[(table["lemma"] == lemma) & (table["pos"] == pos)]
    rows = rows.sort_values("share", ascending=False).head(top_n)
    labels = rows["synset"].values[::-1]
    values = rows["share"].values[::-1]

    fig, ax = plt.subplots(figsize=(8, 0.45 * len(labels) + 1.5))
    ax.barh(np.arange(len(values)), values)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("share within lemma")
    ax.set_title(lemma + " (" + pos + ")")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
