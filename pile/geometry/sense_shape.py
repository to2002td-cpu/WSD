import numpy as np
import pandas as pd

from geometry import (load_records, available_steps, layer_list, load_step,
                      standardize_dimensions)


def shape_of_lemma(vectors, labels, standardize=True):
    """Position and spread of each sense inside its lemma, on a common scale.

    peripherality: distance of the sense centroid from the lemma centre, over the
    typical distance of a point from that centre. Near 0 means the sense sits where
    the lemma sits on average, above 1 means it occupies its own region.

    dispersion: typical distance of the sense's points from their own centroid, on
    the same scale. High means the sense is internally heterogeneous.
    """
    x = vectors.astype(np.float32)
    keep = np.linalg.norm(x, axis=1) > 0
    x = x[keep]
    labels = np.asarray(labels)[keep]
    if len(x) < 10:
        return []

    x = x - x.mean(axis=0, keepdims=True)
    if standardize:
        x = standardize_dimensions(x)

    scale = np.sqrt((x ** 2).sum(axis=1).mean())
    rows = []
    for sense in pd.unique(labels):
        mask = labels == sense
        centroid = x[mask].mean(axis=0)
        offsets = x[mask] - centroid
        rows.append({
            "sense": sense,
            "n_sense": int(mask.sum()),
            "peripherality": float(np.linalg.norm(centroid) / scale),
            "dispersion": float(np.sqrt((offsets ** 2).sum(axis=1).mean()) / scale),
        })
    return rows


def run_shape(emb_dir, dataset_path, steps=None, layers=None, cache_dir=None, standardize=True):
    meta = load_records(dataset_path)
    meta["row"] = np.arange(len(meta))

    steps = steps or available_steps(emb_dir)
    all_layers = layer_list(emb_dir, steps[0])
    layers = layers or all_layers

    rows = []
    for step in steps:
        for layer in layers:
            index = all_layers.index(layer)
            vectors = load_step(emb_dir, step, index, cache_dir)
            for (word, pos), group in meta.groupby(["word", "pos"]):
                sub = vectors[group["row"].to_numpy()]
                for row in shape_of_lemma(sub, group["sense"].to_numpy(), standardize):
                    row.update({"step": step, "layer": layer, "word": word, "pos": pos})
                    rows.append(row)
            del vectors
            print("done step", step, "layer", layer, flush=True)
    return pd.DataFrame(rows)


def shape_correlations(merged, columns=("peripherality", "dispersion"), min_senses=3):
    """Rank correlation between share and each shape measure, computed inside each
    lemma and then aggregated, the same way as for purity."""
    rows = []
    for (step, layer), block in merged.groupby(["step", "layer"]):
        entry = {"step": step, "layer": layer}
        for column in columns:
            values = []
            for (word, pos), group in block.groupby(["word", "pos"]):
                if len(group) < min_senses or group["share"].nunique() < 2:
                    continue
                rho = group["share"].corr(group[column], method="spearman")
                if rho == rho:
                    values.append(rho)
            if values:
                values = np.array(values)
                entry["n_lemmas"] = len(values)
                entry["mean_rho_" + column] = values.mean()
                entry["positive_" + column] = float((values > 0).mean())
        rows.append(entry)
    return pd.DataFrame(rows).sort_values(["layer", "step"])
