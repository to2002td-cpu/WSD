import pandas as pd

from analysis import load_table, sense_frequency


def dominance(table):
    rows = []
    for (lemma, pos), group in table.groupby(["lemma", "pos"]):
        group = group.sort_values("share", ascending=False)
        rows.append({
            "lemma": lemma,
            "pos": pos,
            "n_senses": len(group),
            "top_share": round(group["share"].iloc[0], 3),
            "second_share": round(group["share"].iloc[1], 3) if len(group) > 1 else 0.0,
            "occurrences": int(group["n"].sum()),
        })
    return pd.DataFrame(rows)


def distinguishability(table):
    work = table.copy()
    work["conf_ratio"] = work["n_conf"] / work["n"]
    return work


def main():
    df = load_table("big_annotation/annotations_table.csv")
    table = sense_frequency(df, tau=0.8, floor=30, drop_monosemous=True)

    dom = dominance(table)

    print("lemmi con piu sensi sopra floor")
    print(dom.sort_values("n_senses", ascending=False).head(15).to_string(index=False))
    print()

    print("lemmi piu bilanciati, senso dominante piu basso")
    print(dom[dom["n_senses"] >= 3].sort_values("top_share").head(15).to_string(index=False))
    print()

    print("lemmi piu sbilanciati, senso dominante piu alto")
    print(dom[dom["n_senses"] >= 3].sort_values("top_share", ascending=False).head(10).to_string(index=False))
    print()

    print("distribuzione della quota del senso dominante")
    print(dom["top_share"].describe().round(3))
    print()

    dist = distinguishability(table)
    print("sensi meno distinguibili, quota di occorrenze confidenti piu bassa")
    print(dist.sort_values("conf_ratio").head(15)[["lemma", "pos", "synset", "n", "n_conf", "share", "conf_ratio"]].round(3).to_string(index=False))
    print()

    print("token da codice, come sono ripartiti")
    for lemma, pos in [("class", "n"), ("value", "n"), ("return", "n"), ("code", "n"), ("function", "n")]:
        rows = table[(table["lemma"] == lemma) & (table["pos"] == pos)]
        if len(rows) == 0:
            continue
        print(rows.sort_values("share", ascending=False)[["lemma", "synset", "n", "share"]].head(4).round(3).to_string(index=False))
        print()


if __name__ == "__main__":
    main()
