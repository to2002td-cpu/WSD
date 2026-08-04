import argparse
import glob
import io
import json
import multiprocessing
import os

import spacy

from targets import read_lemmas, build_target_lemmapos
from wnutils import coarse_from_spacy


def iter_local(pile_dir):
    import zstandard

    if os.path.isfile(pile_dir):
        paths = [pile_dir]
    else:
        paths = sorted(glob.glob(os.path.join(pile_dir, "*.jsonl.zst")))
    dctx = zstandard.ZstdDecompressor()
    for path in paths:
        with open(path, "rb") as fh:
            reader = dctx.stream_reader(fh)
            text_stream = io.TextIOWrapper(reader, encoding="utf-8")
            for line in text_stream:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                yield obj.get("text", "")


def iter_pythia_bin(bin_path, tokenizer_path, start_token, end_token):
    import numpy as np
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(tokenizer_path)
    data = np.memmap(bin_path, dtype=np.uint16, mode="r")
    end = end_token if end_token is not None else len(data)

    eod = 0
    pending = []
    position = start_token
    block_size = 20_000_000
    while position < end:
        upper = min(position + block_size, end)
        block = np.asarray(data[position:upper])
        boundaries = np.flatnonzero(block == eod)
        previous = 0
        for boundary in boundaries:
            pending.append(block[previous:boundary])
            document = np.concatenate(pending) if len(pending) > 1 else pending[0]
            pending = []
            previous = boundary + 1
            if len(document) > 0:
                yield tokenizer.decode(document.tolist())
        if previous < len(block):
            pending.append(block[previous:])
        position = upper


def iter_hf():
    from datasets import load_dataset

    ds = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True)
    for obj in ds:
        yield obj["text"]


def iter_pile(source, pile_dir, max_docs, args=None):
    if source == "local":
        gen = iter_local(pile_dir)
    elif source == "pythia":
        gen = iter_pythia_bin(args.bin_path, args.tokenizer, args.start_token, args.end_token)
    else:
        gen = iter_hf()
    for i, text in enumerate(gen):
        if max_docs is not None and i >= max_docs:
            break
        yield text


def clean_texts(source, pile_dir, max_docs, max_chars, max_token_len, args=None):
    for text in iter_pile(source, pile_dir, max_docs, args):
        if not text:
            continue
        text = text[:max_chars]
        if any(len(tok) > max_token_len for tok in text.split()):
            continue
        yield text


def is_aux_or_modal(token):
    return token.pos_ == "AUX" or token.tag_ == "MD"


def main():
    args = parse_args()

    if args.n_process > 1:
        try:
            multiprocessing.set_start_method("spawn")
        except RuntimeError:
            pass

    lemmas = read_lemmas(args.targets)
    targets = build_target_lemmapos(lemmas)

    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    nlp.add_pipe("sentencizer")
    nlp.max_length = 2_000_000

    out = open(args.out, "w")
    seen = {}
    kept = {}
    cap = args.cap
    occ_index = 0

    texts = clean_texts(args.source, args.pile_dir, args.max_docs, args.max_chars, args.max_token_len, args)
    processed = 0
    for doc in nlp.pipe(texts, batch_size=args.batch_size, n_process=args.n_process):
        processed += 1
        if processed % 2000 == 0:
            print("processed docs", processed, "written", occ_index, flush=True)
            out.flush()
        for sent in doc.sents:
            sent_tokens = [t.text for t in sent]
            for local_idx, token in enumerate(sent):
                pos = coarse_from_spacy(token.pos_)
                if pos is None:
                    continue
                lemma = token.lemma_.lower()
                key = (lemma, pos)
                if key not in targets:
                    continue
                seen[key] = seen.get(key, 0) + 1
                if kept.get(key, 0) >= cap:
                    continue
                record = {
                    "occ_id": "occ" + str(occ_index),
                    "lemma": lemma,
                    "pos": pos,
                    "tokens": sent_tokens,
                    "target_index": local_idx,
                    "surface": token.text,
                    "is_aux_or_modal": is_aux_or_modal(token),
                }
                out.write(json.dumps(record) + "\n")
                kept[key] = kept.get(key, 0) + 1
                occ_index += 1

    out.close()

    for key in sorted(seen):
        print(key[0], key[1], "seen", seen[key], "kept", kept.get(key, 0))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", required=True, help="Path to the top100 lemma list")
    parser.add_argument("--out", required=True, help="Output occurrences jsonl")
    parser.add_argument("--source", choices=["local", "hf", "pythia"], default="hf")
    parser.add_argument("--bin-path", default=None, help="Pythia preshuffled .bin when source is pythia")
    parser.add_argument("--tokenizer", default=None, help="path to 20B_tokenizer.json")
    parser.add_argument("--start-token", type=int, default=0)
    parser.add_argument("--end-token", type=int, default=None)
    parser.add_argument("--pile-dir", default=None, help="Directory of jsonl.zst shards when source is local")
    parser.add_argument("--cap", type=int, default=3000, help="Reservoir size per (lemma, pos)")
    parser.add_argument("--max-docs", type=int, default=None, help="Document budget over the stream")
    parser.add_argument("--n-process", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--max-chars", type=int, default=100000)
    parser.add_argument("--max-token-len", type=int, default=100)
    return parser.parse_args()


if __name__ == "__main__":
    main()
