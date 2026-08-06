import argparse
import os

import pandas as pd

from geometry_plots import pca_grid, purity_grid, emergence_curves, sense_curves


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    for target in args.lemmas.split(","):
        word, pos = target.split(".")
        stem = os.path.join(args.out_dir, word + "_" + pos)
        print(pca_grid(args.emb_dir, args.dataset, word, pos, stem + "_pca.png",
                       cache_dir=args.cache_dir))
        print(purity_grid(args.emb_dir, args.dataset, word, pos, stem + "_purity.png",
                          cache_dir=args.cache_dir))

    if args.table and os.path.exists(args.table):
        table = pd.read_csv(args.table)
        for layer in [int(x) for x in args.layers.split(",")]:
            path = os.path.join(args.out_dir, "curves_layer" + str(layer) + ".png")
            print(emergence_curves(table, path, metric=args.metric, layer=layer, top=args.top))

    if args.per_sense and os.path.exists(args.per_sense):
        per_sense = pd.read_csv(args.per_sense)
        for target in args.lemmas.split(","):
            word, pos = target.split(".")
            path = os.path.join(args.out_dir, word + "_" + pos + "_sense_curves.png")
            print(sense_curves(per_sense, word, pos, path, metric=args.metric, layer=16))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb-dir", required=True)
    parser.add_argument("--dataset", default="pile_dataset.jsonl")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--out-dir", default="figures")
    parser.add_argument("--lemmas", default="cell.n,order.n,case.n,find.v,work.v,court.n")
    parser.add_argument("--table", default="geometry_full_std.csv")
    parser.add_argument("--per-sense", default="separation_vs_frequency.csv")
    parser.add_argument("--layers", default="8,16,24")
    parser.add_argument("--metric", default="excess_k10")
    parser.add_argument("--top", type=int, default=12)
    return parser.parse_args()


if __name__ == "__main__":
    main()
