import argparse
import gc
import json
import os

import numpy as np
import torch
from tqdm import tqdm


def load_records(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def target_token_indices(offsets, start, end):
    return [i for i, (s, e) in enumerate(offsets) if s < end and e > start and e > s]


@torch.inference_mode()
def extract_checkpoint(model_name, step, records, out_dir, layers, batch_size, max_length, cache_dir):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out_path = os.path.join(out_dir, "step" + str(step) + ".npz")
    if os.path.exists(out_path):
        print("already extracted, skipping", out_path, flush=True)
        return out_path

    revision = "step" + str(step)
    print("loading", model_name, revision, flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision, cache_dir=cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, revision=revision, torch_dtype=torch.float16, cache_dir=cache_dir)
    model.to("cuda")
    model.eval()

    max_layer = model.config.num_hidden_layers
    for layer in layers:
        if not 1 <= layer <= max_layer:
            raise ValueError("layer " + str(layer) + " out of range 1.." + str(max_layer))

    hidden = model.config.hidden_size
    vectors = np.zeros((len(records), len(layers), hidden), dtype=np.float16)
    n_lost = 0

    for begin in tqdm(range(0, len(records), batch_size), desc=revision, dynamic_ncols=True):
        batch = records[begin:begin + batch_size]
        encoded = tokenizer([r["sentence"] for r in batch], return_tensors="pt", padding=True,
                            truncation=True, max_length=max_length, return_offsets_mapping=True)
        offsets = encoded.pop("offset_mapping")
        encoded = {k: v.to("cuda") for k, v in encoded.items()}
        out = model(**encoded, output_hidden_states=True)
        picked = [out.hidden_states[layer] for layer in layers]
        del out
        for j, record in enumerate(batch):
            idx = target_token_indices(offsets[j].tolist(), record["target_start"], record["target_end"])
            if not idx:
                n_lost += 1
                continue
            pooled = torch.stack([h[j, idx, :].float().mean(dim=0) for h in picked])
            vectors[begin + j] = pooled.cpu().numpy().astype(np.float16)
        del picked

    if n_lost:
        print("targets lost to truncation:", n_lost, "of", len(records), flush=True)

    np.savez_compressed(out_path, vectors=vectors, step=step, layers=np.array(layers))
    print("saved", out_path, vectors.shape, flush=True)

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return out_path


def main():
    args = parse_args()

    records = load_records(args.dataset)
    print("records:", len(records), flush=True)
    os.makedirs(args.out_dir, exist_ok=True)

    layers = [int(x) for x in args.layers.split(",")]
    steps = [int(x) for x in args.steps.split(",")]

    for step in steps:
        extract_checkpoint(args.model, step, records, args.out_dir, layers,
                           args.batch_size, args.max_length, args.cache_dir)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--steps", required=True, help="comma separated, e.g. 0,512,1000")
    parser.add_argument("--layers", default="1,8,16,24,32")
    parser.add_argument("--model", default="EleutherAI/pythia-6.9b")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--cache-dir", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    main()
