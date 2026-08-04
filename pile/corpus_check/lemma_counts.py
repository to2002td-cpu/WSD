import argparse
import json

import numpy as np
import pandas as pd


def surface_forms(lemma):
    forms = {lemma, lemma.capitalize()}
    if lemma.endswith("e"):
        forms.update({lemma + "s", lemma + "d", lemma[:-1] + "ing"})
    elif lemma.endswith("y") and len(lemma) > 2 and lemma[-2] not in "aeiou":
        forms.update({lemma[:-1] + "ies", lemma[:-1] + "ied", lemma + "ing"})
    else:
        forms.update({lemma + "s", lemma + "es", lemma + "ed", lemma + "ing"})
    expanded = set()
    for form in forms:
        expanded.add(form)
        expanded.add(form.capitalize())
    return sorted(expanded)


def load_vocab(tokenizer_path):
    with open(tokenizer_path) as f:
        tok = json.load(f)
    return tok["model"]["vocab"]


def main():
    args = parse_args()

    vocab = load_vocab(args.tokenizer)
    counts = np.load(args.counts)

    lemmas = []
    with open(args.lemmas) as f:
        for line in f:
            line = line.strip()
            if line:
                lemmas.append(line.split()[0].lower())

    rows = []
    missing = []
    for lemma in lemmas:
        for form in surface_forms(lemma):
            for prefix, label in [("\u0120", "with_space"), ("", "no_space")]:
                key = prefix + form
                token_id = vocab.get(key)
                if token_id is None:
                    missing.append((lemma, key))
                    continue
                rows.append({
                    "lemma": lemma,
                    "form": form,
                    "position": label,
                    "token_id": token_id,
                    "count": int(counts[token_id]) if token_id < len(counts) else 0,
                })

    forms_table = pd.DataFrame(rows)
    forms_table = forms_table[forms_table["count"] > 0]
    forms_table = forms_table.sort_values(["lemma", "count"], ascending=[True, False])
    forms_table.to_csv(args.out_forms, index=False)

    lemma_table = forms_table.groupby("lemma")["count"].sum().rename("count").reset_index()
    total = counts.sum()
    lemma_table["per_million"] = (1e6 * lemma_table["count"] / total).round(2)
    lemma_table = lemma_table.sort_values("count", ascending=False)
    lemma_table.to_csv(args.out_lemmas, index=False)

    print("total tokens in shard:", int(total))
    print("lemmas with at least one form found:", len(lemma_table))
    print("forms that are not a single token:", len(missing))
    print()
    print(lemma_table.head(25).to_string(index=False))
    print()
    print("wrote", args.out_forms, "and", args.out_lemmas)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--counts", default="token_counts.npy")
    parser.add_argument("--lemmas", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--out-forms", default="pythia_form_counts.csv")
    parser.add_argument("--out-lemmas", default="pythia_lemma_counts.csv")
    return parser.parse_args()


if __name__ == "__main__":
    main()
