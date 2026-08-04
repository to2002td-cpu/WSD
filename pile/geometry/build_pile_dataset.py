import argparse
import json
import random

import pandas as pd


def sentence_with_offsets(tokens, target_index):
    parts = []
    start = None
    end = None
    position = 0
    for i, token in enumerate(tokens):
        if i > 0:
            position += 1
        if i == target_index:
            start = position
            end = position + len(token)
        parts.append(token)
        position += len(token)
    return " ".join(parts), start, end


def main():
    args = parse_args()

    senses = pd.read_csv(args.senses)
    wanted = {}
    for row in senses.itertuples():
        wanted[(row.lemma, row.pos, row.synset)] = row.sense_key

    table = pd.read_csv(args.table, usecols=["occ_id", "lemma", "pos", "synset", "top_prob", "is_aux_or_modal"])
    table = table[~table["is_aux_or_modal"]]
    table = table[table["top_prob"] >= args.tau]
    keys = list(zip(table["lemma"], table["pos"], table["synset"]))
    table = table[[k in wanted for k in keys]]
    print("confident occurrences in the selected senses:", len(table))

    random.seed(args.seed)
    chosen = {}
    for (lemma, pos, synset), group in table.groupby(["lemma", "pos", "synset"]):
        ids = group["occ_id"].tolist()
        if len(ids) < args.per_sense:
            continue
        picked = random.sample(ids, args.per_sense)
        for occ_id in picked:
            chosen[occ_id] = (lemma, pos, synset)
    print("senses kept at", args.per_sense, "occurrences each:", len(set(chosen.values())))
    print("rows to write:", len(chosen))

    records = []
    with open(args.occurrences) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            occ = json.loads(line)
            target = chosen.get(occ["occ_id"])
            if target is None:
                continue
            lemma, pos, synset = target
            sentence, start, end = sentence_with_offsets(occ["tokens"], occ["target_index"])
            if start is None:
                continue
            records.append({
                "word": lemma,
                "pos": pos,
                "sense": synset,
                "sentence": sentence,
                "target_start": start,
                "target_end": end,
                "surface": occ["surface"],
                "occ_id": occ["occ_id"],
                "source": "pile",
            })

    records.sort(key=lambda r: (r["word"], r["pos"], r["sense"], r["occ_id"]))
    for i, rec in enumerate(records):
        rec["sent_id"] = i

    with open(args.out, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    frame = pd.DataFrame(records)
    print()
    print("rows written:", len(frame))
    print("distinct senses:", frame["sense"].nunique())
    print("distinct lemma-pos:", frame[["word", "pos"]].drop_duplicates().shape[0])
    print("mean sentence length in characters:", int(frame["sentence"].str.len().mean()))
    print("wrote", args.out)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", default="big_annotation/annotations_table.csv")
    parser.add_argument("--occurrences", default="big_annotation/occurrences_full.jsonl")
    parser.add_argument("--senses", default="selection_senses.csv")
    parser.add_argument("--out", default="pile_dataset.jsonl")
    parser.add_argument("--per-sense", type=int, default=200)
    parser.add_argument("--tau", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    main()
