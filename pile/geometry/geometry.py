import json
import os

import numpy as np
import pandas as pd


def load_records(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return pd.DataFrame(records)


def available_steps(emb_dir):
    steps = []
    for name in os.listdir(emb_dir):
        if name.startswith("step") and name.endswith(".npz"):
            steps.append(int(name[4:-4]))
    return sorted(steps)


def cached_vectors(emb_dir, step, cache_dir=None):
    """Memory-map the checkpoint vectors. The npz is fully decompressed on every
    read, so keep an uncompressed npy twin and map that instead: slicing a lemma
    then touches only its rows."""
    npz_path = os.path.join(emb_dir, "step" + str(step) + ".npz")
    if cache_dir is None:
        return np.load(npz_path)["vectors"]
    os.makedirs(cache_dir, exist_ok=True)
    npy_path = os.path.join(cache_dir, "step" + str(step) + ".npy")
    if not os.path.exists(npy_path):
        print("caching step", step, "as npy", flush=True)
        temporary = npy_path + ".tmp"
        with open(temporary, "wb") as handle:
            np.save(handle, np.load(npz_path)["vectors"])
        os.rename(temporary, npy_path)
    return np.load(npy_path, mmap_mode="r")


def load_step(emb_dir, step, layer_index, cache_dir=None):
    vectors = cached_vectors(emb_dir, step, cache_dir)
    return np.asarray(vectors[:, layer_index, :]).astype(np.float32)


def layer_list(emb_dir, step):
    data = np.load(os.path.join(emb_dir, "step" + str(step) + ".npz"))
    return [int(x) for x in data["layers"]]


def standardize_dimensions(x):
    """Divide each dimension by its standard deviation. Transformer hidden states
    have a few dimensions of very large magnitude that dominate cosine similarity
    and PCA (rogue dimensions); scaling puts every dimension on equal footing."""
    scale = x.std(axis=0, keepdims=True)
    return x / np.maximum(scale, 1e-6)


def knn_purity(vectors, labels, k_values, center=True, standardize=False):
    """Fraction of the k nearest neighbours sharing the sense, averaged over points.
    Reported next to the chance level, which is the probability that a random other
    point shares the sense."""
    x = vectors.astype(np.float32)
    keep = np.linalg.norm(x, axis=1) > 0
    x = x[keep]
    labels = np.asarray(labels)[keep]
    if len(x) < max(k_values) + 1:
        return None

    if center:
        x = x - x.mean(axis=0, keepdims=True)
    if standardize:
        x = standardize_dimensions(x)
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)

    similarity = x @ x.T
    np.fill_diagonal(similarity, -np.inf)
    order = np.argsort(-similarity, axis=1)

    counts = pd.Series(labels).value_counts()
    total = len(labels)
    chance = sum(c * (c - 1) for c in counts) / (total * (total - 1))

    out = {"n": len(labels), "n_senses": len(counts), "chance": chance}
    same = labels[order] == labels[:, None]
    for k in k_values:
        purity = same[:, :k].mean()
        out["purity_k" + str(k)] = purity
        out["excess_k" + str(k)] = (purity - chance) / (1.0 - chance)
    return out


def mean_pairwise_similarity(vectors, labels, center=True, standardize=False):
    """Mean cosine similarity within a sense minus mean across senses, after
    removing the corpus-wide direction."""
    x = vectors.astype(np.float32)
    keep = np.linalg.norm(x, axis=1) > 0
    x = x[keep]
    labels = np.asarray(labels)[keep]
    if center:
        x = x - x.mean(axis=0, keepdims=True)
    if standardize:
        x = standardize_dimensions(x)
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)

    similarity = x @ x.T
    same = labels[:, None] == labels[None, :]
    np.fill_diagonal(same, False)
    off = ~np.eye(len(x), dtype=bool)
    within = similarity[same].mean()
    between = similarity[off & ~same].mean()
    return {"within": float(within), "between": float(between), "gap": float(within - between)}


def run(emb_dir, dataset_path, k_values=(1, 5, 10, 20, 50), steps=None, layers=None, cache_dir=None, standardize=False):
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
                result = knn_purity(sub, group["sense"].to_numpy(), k_values, standardize=standardize)
                if result is None:
                    continue
                similarity = mean_pairwise_similarity(sub, group["sense"].to_numpy(), standardize=standardize)
                result.update({"step": step, "layer": layer, "word": word, "pos": pos})
                result.update(similarity)
                rows.append(result)
            del vectors
            print("done step", step, "layer", layer, flush=True)
    return pd.DataFrame(rows)


def per_sense_purity(vectors, labels, k_values, center=True, standardize=False):
    """Purity computed separately for each sense: of the k nearest neighbours of a
    point of sense s, how many share s, averaged over the points of s only. The
    chance level differs per sense, so the excess is what compares across senses."""
    x = vectors.astype(np.float32)
    keep = np.linalg.norm(x, axis=1) > 0
    x = x[keep]
    labels = np.asarray(labels)[keep]
    total = len(labels)
    if total < max(k_values) + 1:
        return []

    if center:
        x = x - x.mean(axis=0, keepdims=True)
    if standardize:
        x = standardize_dimensions(x)
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)

    similarity = x @ x.T
    np.fill_diagonal(similarity, -np.inf)
    order = np.argsort(-similarity, axis=1)
    same = labels[order] == labels[:, None]

    counts = pd.Series(labels).value_counts()
    rows = []
    for sense, count in counts.items():
        mask = labels == sense
        chance = (count - 1) / (total - 1)
        row = {"sense": sense, "n_sense": int(count), "n_lemma": total, "chance": chance}
        for k in k_values:
            purity = float(same[mask, :k].mean())
            row["purity_k" + str(k)] = purity
            row["excess_k" + str(k)] = (purity - chance) / (1.0 - chance) if chance < 1 else np.nan
        rows.append(row)
    return rows


def run_per_sense(emb_dir, dataset_path, k_values=(5, 10, 20, 50), steps=None, layers=None,
                  cache_dir=None, standardize=True):
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
                for row in per_sense_purity(sub, group["sense"].to_numpy(), k_values,
                                            standardize=standardize):
                    row.update({"step": step, "layer": layer, "word": word, "pos": pos})
                    rows.append(row)
            del vectors
            print("done step", step, "layer", layer, flush=True)
    return pd.DataFrame(rows)
