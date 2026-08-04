import argparse
import csv
import json


def load_occ_meta(path):
    pairs = {}
    meta = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            key = (r["lemma"], r["pos"], bool(r.get("is_aux_or_modal", False)))
            code = pairs.get(key)
            if code is None:
                code = len(pairs)
                pairs[key] = code
            meta[r["occ_id"]] = code
    codes = [None] * len(pairs)
    for key, code in pairs.items():
        codes[code] = key
    return meta, codes


def main():
    args = parse_args()

    meta, codes = load_occ_meta(args.occurrences)
    print("occurrences indexed:", len(meta))
    print("distinct lemma pos aux combinations:", len(codes))

    written = 0
    missing = 0
    out = open(args.out, "w", newline="")
    writer = csv.writer(out)
    writer.writerow(["occ_id", "lemma", "pos", "synset", "sense_key", "top_prob", "is_aux_or_modal"])

    with open(args.annotations) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            a = json.loads(line)
            code = meta.get(a["occ_id"])
            if code is None:
                missing += 1
                continue
            lemma, pos, is_aux = codes[code]
            writer.writerow([
                a["occ_id"],
                lemma,
                pos,
                a["pred_synset"],
                a["pred_sense_key"],
                a["top_prob"],
                is_aux,
            ])
            written += 1
            if written % 1000000 == 0:
                print("written", written, flush=True)

    out.close()
    print("rows:", written)
    print("annotations without a matching occurrence:", missing)
    print("wrote", args.out)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--occurrences", required=True)
    parser.add_argument("--out", default="annotations_table.csv")
    return parser.parse_args()


if __name__ == "__main__":
    main()
