import argparse

import numpy as np
import pandas as pd


def main():
    args = parse_args()

    pythia = pd.read_csv(args.pythia)
    pythia = pythia[["lemma", "count"]].rename(columns={"count": "pythia_count"})

    pile = pd.read_csv(args.pile, header=None, names=["lemma", "pile_count"])

    merged = pythia.merge(pile, on="lemma", how="inner")

    if args.drop_lemmas:
        dropped = set(args.drop_lemmas.split(","))
        merged = merged[~merged["lemma"].isin(dropped)]

    before = len(merged)
    merged = merged[merged["pile_count"] >= args.min_pile_count]
    print("lemmas removed by the min pile count filter:", before - len(merged))

    merged["pythia_share"] = merged["pythia_count"] / merged["pythia_count"].sum()
    merged["pile_share"] = merged["pile_count"] / merged["pile_count"].sum()
    merged["ratio"] = merged["pythia_share"] / merged["pile_share"]
    merged["log_ratio"] = np.log10(merged["ratio"])

    print("lemmas compared:", len(merged))
    print("lemmas only in pythia table:", len(pythia) - len(merged))
    print("lemmas only in pile table:", len(pile) - len(merged))
    print()

    spearman = merged["pythia_share"].corr(merged["pile_share"], method="spearman")
    pearson_log = np.log10(merged["pythia_share"]).corr(np.log10(merged["pile_share"]))
    print("spearman rank correlation:", round(spearman, 4))
    print("pearson correlation on log shares:", round(pearson_log, 4))
    print()

    print("ratio of shares, distribution")
    print(merged["ratio"].describe().round(3))
    print()

    merged = merged.sort_values("log_ratio")
    print("most under-represented in pythia relative to the public pile")
    print(merged.head(10)[["lemma", "pythia_share", "pile_share", "ratio"]].round(6).to_string(index=False))
    print()
    print("most over-represented in pythia relative to the public pile")
    print(merged.tail(10)[["lemma", "pythia_share", "pile_share", "ratio"]].round(6).to_string(index=False))

    merged.sort_values("pythia_count", ascending=False).to_csv(args.out, index=False)
    print()
    print("wrote", args.out)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pythia", default="pythia_lemma_counts.csv")
    parser.add_argument("--pile", default="../pile_lemma_seen.csv")
    parser.add_argument("--out", default="count_comparison.csv")
    parser.add_argument("--min-pile-count", type=int, default=0,
                        help="drop lemmas the extraction pipeline does not really target")
    parser.add_argument("--drop-lemmas", default="",
                        help="comma separated lemmas to exclude, e.g. inflected forms of other entries")
    return parser.parse_args()


if __name__ == "__main__":
    main()
