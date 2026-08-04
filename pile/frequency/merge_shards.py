import argparse
import glob
import json


def main():
    args = parse_args()

    paths = sorted(glob.glob(args.prefix + "*.jsonl"))
    if not paths:
        print("no shard files matching", args.prefix + "*.jsonl")
        return

    kept = {}
    idx = 0
    out = open(args.out, "w")
    for path in paths:
        n = 0
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                key = (r["lemma"], r["pos"])
                if kept.get(key, 0) >= args.cap:
                    continue
                r["occ_id"] = "occ" + str(idx)
                idx += 1
                out.write(json.dumps(r) + "\n")
                kept[key] = kept.get(key, 0) + 1
                n += 1
        print(path, n)
    out.close()

    print("written", idx)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default="big_annotation/occ_shard_")
    parser.add_argument("--out", default="big_annotation/occurrences_full.jsonl")
    parser.add_argument("--cap", type=int, default=100000)
    return parser.parse_args()


if __name__ == "__main__":
    main()
