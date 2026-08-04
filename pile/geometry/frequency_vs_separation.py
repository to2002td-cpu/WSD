import numpy as np
import pandas as pd


def merge_frequency(separation, frequency_path, metric="excess_k10"):
    """Attach the corpus share of each sense to its geometric separation. The
    separation is measured on a fixed number of occurrences per sense, so sample
    size cannot drive the relation."""
    frequency = pd.read_csv(frequency_path)
    frequency = frequency.rename(columns={"lemma": "word", "synset": "sense"})
    keep = ["word", "pos", "sense", "n", "n_conf", "share", "share_conf"]
    keep = [c for c in keep if c in frequency.columns]
    merged = separation.merge(frequency[keep], on=["word", "pos", "sense"], how="inner")
    merged["log_share"] = np.log10(merged["share"])
    return merged


def baseline_corrected(merged, metric="excess_k10", baseline_step=0):
    """Subtract each sense's separation at the untrained checkpoint, which captures
    what lexical overlap of the context gives for free."""
    base = merged[merged["step"] == baseline_step]
    base = base[["word", "pos", "sense", "layer", metric]].rename(columns={metric: "baseline"})
    out = merged.merge(base, on=["word", "pos", "sense", "layer"], how="left")
    out[metric + "_corrected"] = out[metric] - out["baseline"]
    return out


def correlations(merged, metric="excess_k10"):
    rows = []
    for (step, layer), group in merged.groupby(["step", "layer"]):
        group = group[group["share"] > 0]
        if len(group) < 10:
            continue
        rows.append({
            "step": step,
            "layer": layer,
            "n_senses": len(group),
            "spearman": group["share"].corr(group[metric], method="spearman"),
            "pearson_log": group["log_share"].corr(group[metric]),
            "mean_metric": group[metric].mean(),
        })
    return pd.DataFrame(rows).sort_values(["layer", "step"])


def emergence_step(merged, metric="excess_k10", threshold=0.5, layer=16):
    """The first checkpoint at which a sense passes the separation threshold, which
    is the quantity the frequency hypothesis predicts should scale with rarity."""
    data = merged[merged["layer"] == layer].sort_values("step")
    rows = []
    for (word, pos, sense), group in data.groupby(["word", "pos", "sense"]):
        passed = group[group[metric] >= threshold]
        rows.append({
            "word": word,
            "pos": pos,
            "sense": sense,
            "share": group["share"].iloc[0],
            "log_share": group["log_share"].iloc[0],
            "emergence_step": int(passed["step"].iloc[0]) if len(passed) else np.nan,
            "final_metric": group[metric].iloc[-1],
        })
    out = pd.DataFrame(rows)
    reached = out[out["emergence_step"].notna()].copy()
    if len(reached) > 5:
        reached["log_step"] = np.log10(reached["emergence_step"].replace(0, 1))
        print("senses reaching the threshold:", len(reached), "of", len(out))
        print("spearman between share and emergence step:",
              round(reached["share"].corr(reached["emergence_step"], method="spearman"), 4))
        print("pearson between log share and log emergence step:",
              round(reached["log_share"].corr(reached["log_step"]), 4))
    return out


def within_lemma_correlations(merged, metric="excess_k10", min_senses=3):
    """Correlate share with separation inside each lemma, then aggregate. Lemmas
    differ widely in baseline separability, so pooling senses across lemmas buries
    the within-lemma effect the hypothesis is about."""
    rows = []
    for (step, layer), block in merged.groupby(["step", "layer"]):
        per_lemma = []
        for (word, pos), group in block.groupby(["word", "pos"]):
            if len(group) < min_senses:
                continue
            if group["share"].nunique() < 2 or group[metric].nunique() < 2:
                continue
            per_lemma.append(group["share"].corr(group[metric], method="spearman"))
        per_lemma = [value for value in per_lemma if value == value]
        if not per_lemma:
            continue
        values = np.array(per_lemma)
        rows.append({
            "step": step,
            "layer": layer,
            "n_lemmas": len(values),
            "mean_rho": values.mean(),
            "median_rho": np.median(values),
            "share_positive": float((values > 0).mean()),
        })
    return pd.DataFrame(rows).sort_values(["layer", "step"])


def rank_within_lemma(merged, metric="excess_k10"):
    """Rank senses by share and by separation inside each lemma and layer, so the
    two can be compared on a common scale across lemmas of different difficulty."""
    out = merged.copy()
    keys = ["word", "pos", "step", "layer"]
    out["share_rank"] = out.groupby(keys)["share"].rank(pct=True)
    out["metric_rank"] = out.groupby(keys)[metric].rank(pct=True)
    return out
