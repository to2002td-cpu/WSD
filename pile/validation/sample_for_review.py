import argparse
import json

import pandas as pd
from nltk.corpus import wordnet as wn


def gloss_for(synset_name):
    try:
        return wn.synset(synset_name).definition()
    except Exception:
        return ""


def mark_target(tokens, target_index):
    marked = list(tokens)
    marked[target_index] = "<<" + marked[target_index] + ">>"
    return " ".join(marked)


def main():
    args = parse_args()

    table = pd.read_csv(args.table)
    table = table[~table["is_aux_or_modal"]]

    freq = pd.read_csv(args.frequency)
    shares = freq[["lemma", "pos", "synset", "share"]]

    merged = table.merge(shares, left_on=["lemma", "pos", "synset"], right_on=["lemma", "pos", "synset"], how="inner")

    rare = merged[merged["share"] <= args.max_share]
    frequent = merged[merged["share"] > args.max_share]

    n_rare = min(args.n_per_stratum, len(rare))
    n_frequent = min(args.n_per_stratum, len(frequent))
    rare = rare.sample(n=n_rare, random_state=args.seed).assign(stratum="rare")
    frequent = frequent.sample(n=n_frequent, random_state=args.seed).assign(stratum="frequent")

    sample = pd.concat([rare, frequent]).sample(frac=1.0, random_state=args.seed)
    wanted = set(sample["occ_id"])

    sentences = {}
    with open(args.occurrences) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r["occ_id"] in wanted:
                sentences[r["occ_id"]] = mark_target(r["tokens"], r["target_index"])
                if len(sentences) == len(wanted):
                    break

    sample["sentence"] = sample["occ_id"].map(sentences)
    sample = sample[sample["sentence"].notna()]
    sample = sample[sample["sentence"].str.len() <= args.max_chars]

    sample["gloss"] = sample["synset"].map(gloss_for)
    sample = sample.rename(columns={"synset": "pred_synset"})

    sample[["occ_id", "stratum"]].to_csv(args.out.replace(".csv", "_key.csv"), index=False)

    review = sample[["occ_id", "lemma", "pos", "pred_synset", "gloss", "top_prob", "sentence"]].copy()
    review.to_csv(args.out, index=False)

    print("rare sampled:", (sample["stratum"] == "rare").sum())
    print("frequent sampled:", (sample["stratum"] == "frequent").sum())
    print("wrote", args.out)
    print("wrote", args.out.replace(".csv", "_key.csv"))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", required=True, help="annotations_table.csv from build_table.py")
    parser.add_argument("--occurrences", required=True, help="occurrences jsonl, read once for the sentences")
    parser.add_argument("--frequency", required=True, help="sense_frequency csv, used to split rare from frequent")
    parser.add_argument("--out", default="review_sample.csv")
    parser.add_argument("--n-per-stratum", type=int, default=50)
    parser.add_argument("--max-share", type=float, default=0.05)
    parser.add_argument("--max-chars", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    main()
