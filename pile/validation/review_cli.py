import argparse
import json
import os
import textwrap

import pandas as pd


def collapse(text):
    return " ".join(str(text).split())


def verdicts_path(annotator):
    return "verdicts_" + annotator + ".jsonl"


def load_verdicts(path):
    verdicts = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    row = json.loads(line)
                    verdicts[row["occ_id"]] = row["verdict"]
    return verdicts


def show_all_senses(all_senses, predicted):
    print("all WordNet senses of this lemma:")
    for entry in str(all_senses).split(" | "):
        name = entry.split(":")[0]
        marker = "->" if name == predicted else "  "
        for line in textwrap.wrap(marker + " " + entry, width=100, subsequent_indent="     "):
            print(line)
    print("(-> marks the sense the classifier predicted)")
    print()


def review(args):
    path = verdicts_path(args.annotator)
    sample = pd.read_csv(args.sample)
    verdicts = load_verdicts(path)
    pending = sample[~sample["occ_id"].isin(verdicts.keys())]

    print("annotator:", args.annotator)
    print("already judged:", len(verdicts))
    print("remaining:", len(pending))
    print()
    print("For each item, decide whether the predicted sense matches how the")
    print("marked word <<like this>> is used in the sentence.")
    print()
    print("  y  the predicted gloss is correct for this occurrence")
    print("  n  a different WordNet sense of this lemma would fit better")
    print("  o  no WordNet sense fits this use at all")
    print("  m  show all WordNet senses of this lemma, then decide")
    print("  s  skip this item (too little context to judge)")
    print("  q  save and quit; you can resume later")
    print()

    out = open(path, "a")
    for row in pending.itertuples():
        print(row.lemma + " (" + row.pos + ")   classifier confidence " + str(row.top_prob))
        print("predicted sense: " + str(row.pred_synset))
        print("gloss: " + collapse(row.gloss))
        print()
        for line in textwrap.wrap(collapse(row.sentence), width=100):
            print("  " + line)
        print()
        answer = None
        while answer is None:
            key = input("verdict [y/n/o/m/s/q]: ").strip().lower()
            if key == "m":
                print()
                show_all_senses(row.all_senses, row.pred_synset)
                continue
            if key in ("y", "n", "o", "s", "q"):
                answer = key
            else:
                print("please type y, n, o, m, s or q")
        print()
        if answer == "q":
            break
        if answer == "s":
            continue
        out.write(json.dumps({"occ_id": row.occ_id, "verdict": answer}) + "\n")
        out.flush()
    out.close()
    print("verdicts saved to", path)


def score(args):
    path = verdicts_path(args.annotator)
    sample = pd.read_csv(args.sample)
    verdicts = load_verdicts(path)

    df = sample.copy()
    df["verdict"] = df["occ_id"].map(verdicts)
    df = df[df["verdict"].notna()]

    print("annotator:", args.annotator)
    print("judged:", len(df), "of", len(sample))
    print()

    if len(df) == 0:
        print("no verdicts yet")
        return

    print("verdict counts")
    print(df["verdict"].value_counts())
    print()
    print("share judged correct:", round((df["verdict"] == "y").mean(), 3))
    print()

    bins = [0.0, 0.5, 0.7, 0.9, 1.01]
    df["prob_bin"] = pd.cut(df["top_prob"], bins=bins, right=False)
    by_prob = df.groupby("prob_bin").apply(lambda g: pd.Series({
        "n": len(g),
        "share_correct": (g["verdict"] == "y").mean(),
    }))
    print("by classifier confidence")
    print(by_prob)

    key_path = args.sample.replace(".csv", "_key.csv")
    if not os.path.exists(key_path):
        return

    key = pd.read_csv(key_path)
    df = df.merge(key, on="occ_id")
    print()
    print("verdicts by stratum")
    print(df.groupby(["stratum", "verdict"]).size().unstack(fill_value=0))
    print()
    correct = df[df["verdict"] == "y"].groupby("stratum").size()
    total = df.groupby("stratum").size()
    accuracy = (correct / total).rename("share_correct")
    print(pd.concat([total.rename("n"), accuracy], axis=1))


def main():
    args = parse_args()
    if args.score:
        score(args)
    else:
        review(args)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", default="review_sample.csv")
    parser.add_argument("--annotator", required=True)
    parser.add_argument("--score", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
