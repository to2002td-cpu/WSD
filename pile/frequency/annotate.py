import argparse
import json
import os
import sys

import hydra
import torch

sys.path.insert(0, os.environ.get("CONSEC_REPO", "consec"))

from src.pl_modules import ConsecPLModule
from src.consec_dataset import ConsecSample, ConsecDefinition
from src.disambiguation_corpora import DisambiguationInstance
from src.scripts.model.predict import predict

from wnutils import synsets_for, sense_key_for


def build_sample(record):
    lemma = record["lemma"]
    pos = record["pos"]
    synsets = synsets_for(lemma, pos)
    if len(synsets) == 0:
        return None, None

    candidate_definitions = []
    def_to_sense = {}
    for synset in synsets:
        definition = synset.definition()
        candidate_definitions.append(ConsecDefinition(definition, lemma))
        def_to_sense[definition] = (sense_key_for(synset, lemma), synset.name())

    tokens = record["tokens"]
    context = [
        DisambiguationInstance("d0", "s0", "i" + str(i), tok, None, None, None)
        for i, tok in enumerate(tokens)
    ]
    sample = ConsecSample(
        sample_id=record["occ_id"],
        position=record["target_index"],
        disambiguation_context=context,
        candidate_definitions=candidate_definitions,
        context_definitions=[],
        in_context_sample_id2position={record["occ_id"]: record["target_index"]},
        disambiguation_instance=None,
        gold_definitions=None,
        kwargs={},
    )
    return sample, def_to_sense


def load_records(path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def chunked(iterable, size):
    chunk = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) >= size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def main():
    args = parse_args()

    device = torch.device(args.device if args.device != -1 else "cpu")
    module = ConsecPLModule.load_from_checkpoint(args.checkpoint)
    module.to(device)
    module.freeze()
    module.sense_extractor.evaluation_mode = True
    tokenizer = hydra.utils.instantiate(module.hparams.tokenizer.consec_tokenizer)

    done = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            for line in f:
                line = line.strip()
                if line:
                    done.add(json.loads(line)["occ_id"])
    print("already annotated:", len(done))

    out = open(args.out, "a")

    skipped_long = 0
    pending = []
    for record in load_records(args.occurrences):
        if record["occ_id"] in done:
            continue
        if len(record["tokens"]) > args.max_tokens:
            skipped_long += 1
            continue
        pending.append(record)
    print("skipped for length:", skipped_long)
    print("to annotate:", len(pending))

    for chunk in chunked(pending, args.chunk_size):
        samples = []
        meta_by_id = {}
        for record in chunk:
            sample, def_to_sense = build_sample(record)
            if sample is None:
                continue
            if len(sample.candidate_definitions) == 1:
                sense_key, synset_name = list(def_to_sense.values())[0]
                out.write(json.dumps({
                    "occ_id": sample.sample_id,
                    "pred_sense_key": sense_key,
                    "pred_synset": synset_name,
                    "top_prob": 1.0,
                    "probs": {sense_key: 1.0},
                }) + "\n")
                continue
            samples.append(sample)
            meta_by_id[sample.sample_id] = def_to_sense

        results = predict(
            module,
            tokenizer,
            samples,
            text_encoding_strategy=args.text_encoding_strategy,
            token_batch_size=args.token_batch_size,
            progress_bar=True,
        )

        for sample, probs in results:
            def_to_sense = meta_by_id[sample.sample_id]
            pairs = []
            for cand, p in zip(sample.candidate_definitions, probs):
                sense_key, synset_name = def_to_sense[cand.text]
                pairs.append((sense_key, synset_name, float(p)))
            pairs.sort(key=lambda x: x[2], reverse=True)
            best = pairs[0]
            out.write(json.dumps({
                "occ_id": sample.sample_id,
                "pred_sense_key": best[0],
                "pred_synset": best[1],
                "top_prob": best[2],
                "probs": {sense_key: p for sense_key, synset_name, p in pairs},
            }) + "\n")
        out.flush()

    out.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--occurrences", required=True, help="Occurrences jsonl from extract.py")
    parser.add_argument("--out", required=True, help="Output annotations jsonl")
    parser.add_argument("--checkpoint", required=True, help="Path to consec_semcor.ckpt")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--token-batch-size", type=int, default=4096)
    parser.add_argument("--chunk-size", type=int, default=2000)
    parser.add_argument("--max-tokens", type=int, default=120)
    parser.add_argument("--text-encoding-strategy", default="relative-positions")
    return parser.parse_args()


if __name__ == "__main__":
    main()
