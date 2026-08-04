import argparse
import json

import pandas as pd


def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    args = parse_args()

    annotations = load_jsonl(args.annotations)
    occ = {r["occ_id"]: r for r in load_jsonl(args.occurrences)}

    records = []
    for a in annotations:
        o = occ.get(a["occ_id"])
        if o is None:
            continue
        records.append({
            "lemma": o["lemma"],
            "pos": o["pos"],
            "synset": a["pred_synset"],
            "sense_key": a["pred_sense_key"],
            "top_prob": a["top_prob"],
            "is_aux_or_modal": o.get("is_aux_or_modal", False),
        })

    df = pd.DataFrame(records)
    df = df[~df["is_aux_or_modal"]]

    keys = ["lemma", "pos", "synset", "sense_key"]
    table = df.groupby(keys).size().rename("n").reset_index()

    conf = df[df["top_prob"] >= args.tau]
    conf_table = conf.groupby(keys).size().rename("n_conf").reset_index()

    table = table.merge(conf_table, on=keys, how="left")
    table["n_conf"] = table["n_conf"].fillna(0).astype(int)

    lemma_totals = table.groupby(["lemma", "pos"])["n"].transform("sum")
    table["share"] = table["n"] / lemma_totals

    table = table.sort_values(["lemma", "pos", "n"], ascending=[True, True, False])
    table.to_csv(args.out, index=False)

    kept = table[table["n_conf"] >= args.floor]
    kept_lemmas = kept[["lemma", "pos"]].drop_duplicates()
    kept_lemmas.to_csv(args.out.replace(".csv", "_kept_lemmas.csv"), index=False)

    print("wrote", args.out)
    print("synsets above floor:", len(kept))
    print("lemmas with at least one synset above floor:", len(kept_lemmas))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", required=True, help="Annotations jsonl from annotate.py")
    parser.add_argument("--occurrences", required=True, help="Occurrences jsonl from extract.py")
    parser.add_argument("--out", default="sense_frequency.csv")
    parser.add_argument("--tau", type=float, default=0.5, help="Confidence threshold for the filtered count")
    parser.add_argument("--floor", type=int, default=50, help="Minimum confident occurrences to keep a synset")
    return parser.parse_args()


if __name__ == "__main__":
    main()
