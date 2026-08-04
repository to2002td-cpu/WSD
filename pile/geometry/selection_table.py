import argparse

import pandas as pd

from analysis import load_table, sense_frequency


def build(table, min_occurrences):
    rows = []
    for (lemma, pos), group in table.groupby(["lemma", "pos"]):
        group = group.sort_values("share", ascending=False)
        usable = group[group["n_conf"] >= min_occurrences]
        if len(usable) < 2:
            continue
        shares = usable["share"].values
        rows.append({
            "lemma": lemma,
            "pos": pos,
            "usable_senses": len(usable),
            "top_share": round(shares[0], 3),
            "second_share": round(shares[1], 3),
            "min_share": round(shares[-1], 4),
            "share_span": int(round(shares[0] / shares[-1])),
            "min_n_conf": int(usable["n_conf"].min()),
            "total_n": int(group["n"].sum()),
        })
    return pd.DataFrame(rows)


def main():
    args = parse_args()

    df = load_table(args.table)
    table = sense_frequency(df, tau=args.tau, floor=args.floor, drop_monosemous=True)

    selection = build(table, args.min_occurrences)
    selection = selection.sort_values(["usable_senses", "share_span"], ascending=[False, False])
    selection.to_csv(args.out_lemmas, index=False)

    print("lemma-pos pairs with at least two senses having", args.min_occurrences, "confident occurrences:", len(selection))
    print()
    print(selection.head(30).to_string(index=False))
    print()

    keep = set(zip(selection["lemma"], selection["pos"]))
    senses = table[[t in keep for t in zip(table["lemma"], table["pos"])]]
    senses = senses[senses["n_conf"] >= args.min_occurrences]
    senses = senses.sort_values(["lemma", "pos", "share"], ascending=[True, True, False])
    senses.to_csv(args.out_senses, index=False)

    print("senses available for embedding extraction:", len(senses))
    print("wrote", args.out_lemmas, "and", args.out_senses)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", default="big_annotation/annotations_table.csv")
    parser.add_argument("--tau", type=float, default=0.8)
    parser.add_argument("--floor", type=int, default=30)
    parser.add_argument("--min-occurrences", type=int, default=500)
    parser.add_argument("--out-lemmas", default="selection_lemmas.csv")
    parser.add_argument("--out-senses", default="selection_senses.csv")
    return parser.parse_args()


if __name__ == "__main__":
    main()
