"""Model loading and the logit-lens computation itself.

Pythia is a GPTNeoXForCausalLM: ``hidden_states[L]`` (from
``output_hidden_states=True``) is the *pre-final-norm* residual stream after
transformer block L (``hidden_states[0]`` is the embedding output), and the real
forward pass only ever applies ``final_layer_norm`` + ``embed_out`` once, to the
last block's output. The logit lens reapplies that same final norm + unembedding
to an *earlier* layer's residual, treating it as if it were the last -- a cheap
readout of "what would the model predict if it stopped here". At the true last
layer this must reproduce the model's real logits exactly, which is checked in
``self_check`` below.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import torch

log = logging.getLogger(__name__)


def pick_device(requested: "str | None" = None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def model_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def load_tokenizer(model_name: str, revision: str, cache_dir: Path):
    """Tokenizer only -- Pythia's vocab is identical across checkpoints, so this is
    loaded once up front rather than pulling a full ~14GB model snapshot just to
    read its tokenizer files."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision, cache_dir=str(cache_dir))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model_and_tokenizer(model_name: str, revision: str, cache_dir: Path, device: torch.device):
    from transformers import AutoModelForCausalLM

    tokenizer = load_tokenizer(model_name, revision, cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, revision=revision, dtype=model_dtype(device), cache_dir=str(cache_dir))
    model.to(device)
    model.eval()
    return model, tokenizer


def logit_lens_logprobs(model, hidden: torch.Tensor) -> torch.Tensor:
    """hidden: (..., hidden_size) pre-final-norm residual at some layer.
    Returns log-softmax over the vocabulary, via the model's own final layer norm
    and unembedding (GPTNeoX: ``gpt_neox.final_layer_norm`` + ``embed_out``)."""
    base = model.gpt_neox
    normed = base.final_layer_norm(hidden.float())
    logits = model.embed_out(normed.to(model.embed_out.weight.dtype))
    return torch.log_softmax(logits.float(), dim=-1)


@torch.inference_mode()
def self_check(model, out, num_layers: int) -> None:
    """Sanity check: logit-lens logprobs at the true last layer must match the
    model's real output logprobs. Logs a warning (does not raise) if they drift,
    since this guards against a future transformers refactor changing
    ``output_hidden_states`` semantics (see module docstring). ``out`` is a
    forward pass already run with ``output_hidden_states=True``, reused here
    rather than re-running the model."""
    real = torch.log_softmax(out.logits.float(), dim=-1)
    lens = logit_lens_logprobs(model, out.hidden_states[num_layers])
    max_diff = (real - lens).abs().max().item()
    if max_diff > 1e-2:
        log.warning("logit-lens self-check: max |real - lens| = %.4g at layer %d "
                    "(expected ~0; investigate before trusting results)", max_diff, num_layers)
    else:
        log.info("logit-lens self-check OK (max diff %.2e)", max_diff)


def purge_model_cache(model_name: str, cache_dir: Path) -> None:
    """Delete this model's HF snapshot from ``cache_dir`` (each revision is a full
    ~14GB checkpoint; keep only one on disk at a time). ``cache_dir`` here is the
    directory passed as ``from_pretrained(cache_dir=...)``, under which snapshots
    land directly as ``models--<org>--<name>`` (no extra ``hub/`` level)."""
    target = cache_dir / ("models--" + model_name.replace("/", "--"))
    if target.exists():
        shutil.rmtree(target)
        log.info("Purged HF cache %s", target)
