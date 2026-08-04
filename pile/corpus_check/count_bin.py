import argparse
import os

import numpy as np


VOCAB_SIZE = 50432
SEQ_LEN = 2049
SEQS_PER_STEP = 1024
TOKENS_PER_STEP = SEQ_LEN * SEQS_PER_STEP


def main():
    args = parse_args()

    size_bytes = os.path.getsize(args.bin_path)
    n_tokens = size_bytes // 2
    n_steps = n_tokens / TOKENS_PER_STEP
    print("file:", args.bin_path)
    print("tokens in file:", n_tokens)
    print("training steps covered by this shard:", round(n_steps, 1))
    if args.shard_index is not None:
        first = args.shard_index * n_steps
        print("global step range, assuming equally sized shards:",
              round(first, 1), "-", round(first + n_steps, 1))
    print()

    data = np.memmap(args.bin_path, dtype=np.uint16, mode="r")

    start = args.start_token
    end = args.end_token if args.end_token is not None else n_tokens
    print("counting tokens from", start, "to", end)

    counts = np.zeros(VOCAB_SIZE, dtype=np.int64)
    chunk = args.chunk_tokens
    position = start
    while position < end:
        upper = min(position + chunk, end)
        block = np.asarray(data[position:upper])
        counts += np.bincount(block, minlength=VOCAB_SIZE)
        position = upper
        if (position - start) % (chunk * 20) == 0:
            done = (position - start) / (end - start)
            print("progress", round(100 * done, 1), "percent", flush=True)

    np.save(args.out, counts)
    print()
    print("tokens counted:", int(counts.sum()))
    print("distinct token ids seen:", int((counts > 0).sum()))
    print("wrote", args.out)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bin-path", required=True)
    parser.add_argument("--out", default="token_counts.npy")
    parser.add_argument("--shard-index", type=int, default=None)
    parser.add_argument("--start-token", type=int, default=0)
    parser.add_argument("--end-token", type=int, default=None)
    parser.add_argument("--chunk-tokens", type=int, default=50_000_000)
    return parser.parse_args()


if __name__ == "__main__":
    main()
