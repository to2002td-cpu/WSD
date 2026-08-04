import pandas as pd


def load_table(path):
    return pd.read_csv(path)


def sense_frequency(df, tau=0.8, floor=30, drop_lemmas=None, drop_synsets=None, drop_aux=True, drop_monosemous=False):
    drop_lemmas = set(drop_lemmas or [])
    drop_synsets = set(drop_synsets or [])

    work = df
    if drop_aux:
        work = work[~work["is_aux_or_modal"]]
    if drop_lemmas:
        work = work[~work["lemma"].isin(drop_lemmas)]
    if drop_synsets:
        work = work[~work["synset"].isin(drop_synsets)]

    keys = ["lemma", "pos", "synset", "sense_key"]
    table = work.groupby(keys).size().rename("n").reset_index()

    conf = work[work["top_prob"] >= tau]
    conf_table = conf.groupby(keys).size().rename("n_conf").reset_index()

    table = table.merge(conf_table, on=keys, how="left")
    table["n_conf"] = table["n_conf"].fillna(0).astype(int)

    totals = table.groupby(["lemma", "pos"])["n"].transform("sum")
    table["share"] = table["n"] / totals

    conf_totals = table.groupby(["lemma", "pos"])["n_conf"].transform("sum")
    table["share_conf"] = (table["n_conf"] / conf_totals).fillna(0.0)

    if drop_monosemous:
        senses_in_inventory = table.groupby(["lemma", "pos"])["synset"].transform("size")
        table = table[senses_in_inventory > 1]

    table = table[table["n_conf"] >= floor]
    return table.sort_values(["lemma", "pos", "n"], ascending=[True, True, False])


def coverage(table):
    return pd.Series({
        "synsets": len(table),
        "lemma_pos": table[["lemma", "pos"]].drop_duplicates().shape[0],
        "tail_below_5pct": int((table["share"] < 0.05).sum()),
        "tail_below_2pct": int((table["share"] < 0.02).sum()),
        "tail_below_5pct_conf": int((table["share_conf"] < 0.05).sum()),
        "tail_below_2pct_conf": int((table["share_conf"] < 0.02).sum()),
    })
