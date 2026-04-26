"""
init_from_gpt2.py  —  Initialize the Teaching Assistant model from GPT-2 weights
=================================================================================
This script downloads HuggingFace's `gpt2` (124M small variant) and copies
its weights into a TeachingAssistantModel that has matching geometry.

Why?
----
Phase 8 of the SLM-improvement plan: produce a *parallel* run that starts
from GPT-2 weights instead of from scratch, and compare both on the eval
harness.  This gives the report a clean "from-scratch vs. fine-tuned base"
comparison.

Caveats handled here
--------------------
1. GPT-2 uses **learned absolute position embeddings** (`wpe`), not RoPE.
   We KEEP our RoPE and simply skip `wpe`.  The model's other weights still
   transfer because the residual stream and FFNs don't depend on the
   positional scheme.  In practice this gives a useful warm start; the model
   adapts to RoPE quickly during fine-tuning.
2. GPT-2's weight matrices are stored by HuggingFace as `Conv1D` (which is a
   transposed `nn.Linear`).  For c_attn / c_proj / c_fc we **transpose**
   weights before copying.
3. GPT-2 uses biases everywhere (`bias=True`); our default is `bias=False`.
   We require `bias=True` here so the bias parameters exist.
4. GPT-2 uses LayerNorm and GELU (4·n_embd hidden), so we require
   `use_rmsnorm=False` and `ffn_variant="gelu"`.
5. Tied weights: after loading we explicitly retie `wte` ↔ `lm_head`.

Usage
-----
    python model/init_from_gpt2.py --output checkpoints/gpt2_init.pt

Then resume training from that checkpoint:

    python model/train.py --preset 3090 --model-size 125M \\
        --resume checkpoints/gpt2_init.pt \\
        --out-dir checkpoints_gpt2 \\
        --log-curves runs/curves_gpt2.json
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import torch

from transformer import TeachingAssistantModel, TransformerConfig


# Weight names that need a transpose because HF stores them as Conv1D
# (which is shape (in, out)) and we have nn.Linear (which is (out, in)).
_HF_CONV1D_PARAMS = (
    "attn.c_attn.weight",
    "attn.c_proj.weight",
    "mlp.c_fc.weight",
    "mlp.c_proj.weight",
)


def gpt2_compatible_config() -> TransformerConfig:
    """Return a TransformerConfig that matches GPT-2 small's geometry."""
    return TransformerConfig(
        vocab_size=50_257,
        n_layer=12,
        n_head=12,
        n_embd=768,
        block_size=512,        # we keep our 512 cap, even though GPT-2 has 1024
        dropout=0.1,
        bias=True,             # GPT-2 has biases on every Linear
        use_rmsnorm=False,
        ffn_variant="gelu",
        label_smoothing=0.0,
        stochastic_depth=0.0,
    )


def load_gpt2_into(
    model: TeachingAssistantModel,
    hf_model_id: str = "gpt2",
    verbose: bool = True,
) -> None:
    """
    Copy weights from `hf_model_id` (default GPT-2 small) into `model`.

    Mutates `model` in place.  The model's TransformerConfig must satisfy:
      vocab_size = 50257, n_embd / n_head / n_layer match the HF model,
      bias=True, use_rmsnorm=False, ffn_variant="gelu".
    """
    try:
        from transformers import GPT2LMHeadModel
    except ImportError as e:
        raise ImportError("This script requires `pip install transformers`.") from e

    cfg = model.config
    assert cfg.use_rmsnorm is False, "GPT-2 init requires LayerNorm (use_rmsnorm=False)"
    assert cfg.ffn_variant == "gelu", "GPT-2 init requires ffn_variant='gelu'"
    assert cfg.bias is True,         "GPT-2 init requires bias=True"
    assert cfg.vocab_size == 50_257, "GPT-2 init requires vocab_size=50257"

    if verbose:
        print(f"Loading {hf_model_id} from HuggingFace…")
    hf_model = GPT2LMHeadModel.from_pretrained(hf_model_id)
    hf_state = hf_model.state_dict()

    if verbose:
        n_hf = sum(v.numel() for v in hf_state.values()) / 1e6
        print(f"HF state has {len(hf_state)} tensors, {n_hf:.1f}M parameters total.")

    own_state = model.state_dict()
    n_loaded  = 0
    n_skipped = 0

    # ── Token embeddings (no transpose; same shape) ───────────────────────────
    own_state["transformer.wte.weight"].copy_(hf_state["transformer.wte.weight"])
    n_loaded += 1
    if verbose:
        print("  loaded transformer.wte.weight (token embeddings)")
    # We deliberately ignore transformer.wpe.weight (GPT-2's learned absolute
    # positions) — our model uses RoPE.

    # ── Per-block weights ────────────────────────────────────────────────────
    for i in range(cfg.n_layer):
        prefix_hf  = f"transformer.h.{i}"
        prefix_own = f"transformer.h.{i}"

        # Map HF names → ours.  Note our blocks expose .attn (CausalSelfAttention,
        # which has c_attn and c_proj as nn.Linear) and .ffn (FeedForward, which
        # has c_fc and c_proj as nn.Linear).  Same naming as HF, so the only
        # transformation we need is `.transpose(0, 1)` for the Conv1D params.
        mapping = {
            f"{prefix_hf}.ln_1.weight":         f"{prefix_own}.ln_1.weight",
            f"{prefix_hf}.ln_1.bias":           f"{prefix_own}.ln_1.bias",
            f"{prefix_hf}.attn.c_attn.weight":  f"{prefix_own}.attn.c_attn.weight",
            f"{prefix_hf}.attn.c_attn.bias":    f"{prefix_own}.attn.c_attn.bias",
            f"{prefix_hf}.attn.c_proj.weight":  f"{prefix_own}.attn.c_proj.weight",
            f"{prefix_hf}.attn.c_proj.bias":    f"{prefix_own}.attn.c_proj.bias",
            f"{prefix_hf}.ln_2.weight":         f"{prefix_own}.ln_2.weight",
            f"{prefix_hf}.ln_2.bias":           f"{prefix_own}.ln_2.bias",
            f"{prefix_hf}.mlp.c_fc.weight":     f"{prefix_own}.ffn.c_fc.weight",
            f"{prefix_hf}.mlp.c_fc.bias":       f"{prefix_own}.ffn.c_fc.bias",
            f"{prefix_hf}.mlp.c_proj.weight":   f"{prefix_own}.ffn.c_proj.weight",
            f"{prefix_hf}.mlp.c_proj.bias":     f"{prefix_own}.ffn.c_proj.bias",
        }
        for hf_name, own_name in mapping.items():
            if hf_name not in hf_state:
                if verbose:
                    print(f"  [skip] missing in HF: {hf_name}")
                n_skipped += 1
                continue
            if own_name not in own_state:
                if verbose:
                    print(f"  [skip] missing in own: {own_name}")
                n_skipped += 1
                continue
            src = hf_state[hf_name]
            # Conv1D weight needs transpose to match nn.Linear (out, in)
            if hf_name.endswith(_HF_CONV1D_PARAMS):
                src = src.t().contiguous()
            dst = own_state[own_name]
            if src.shape != dst.shape:
                print(f"  [shape mismatch] {hf_name} {tuple(src.shape)} vs "
                      f"{own_name} {tuple(dst.shape)} — skipping")
                n_skipped += 1
                continue
            dst.copy_(src)
            n_loaded += 1

    # ── Final LayerNorm and LM head ──────────────────────────────────────────
    for hf_name, own_name in (
        ("transformer.ln_f.weight", "transformer.ln_f.weight"),
        ("transformer.ln_f.bias",   "transformer.ln_f.bias"),
    ):
        if hf_name in hf_state and own_name in own_state:
            own_state[own_name].copy_(hf_state[hf_name])
            n_loaded += 1

    # Re-tie lm_head ↔ wte (state_dict copy doesn't preserve tying)
    model.lm_head.weight = model.transformer["wte"].weight

    if verbose:
        print(f"\nLoaded {n_loaded} tensors, skipped {n_skipped}.")


def main():
    p = argparse.ArgumentParser(description="Initialize the model from GPT-2 weights.")
    p.add_argument("--hf-model-id", default="gpt2",
                   help="HuggingFace model id (gpt2, gpt2-medium, …)")
    p.add_argument("--output", required=True,
                   help="Output checkpoint path (compatible with train.py --resume)")
    p.add_argument("--n-layer",  type=int, default=None)
    p.add_argument("--n-head",   type=int, default=None)
    p.add_argument("--n-embd",   type=int, default=None)
    p.add_argument("--block-size", type=int, default=512)
    args = p.parse_args()

    cfg = gpt2_compatible_config()
    if args.n_layer:    cfg.n_layer    = args.n_layer
    if args.n_head:     cfg.n_head     = args.n_head
    if args.n_embd:     cfg.n_embd     = args.n_embd
    if args.block_size: cfg.block_size = args.block_size
    cfg.__post_init__()

    print("Building TeachingAssistantModel with GPT-2-compatible geometry:")
    for k, v in asdict(cfg).items():
        print(f"  {k:20s} = {v}")

    model = TeachingAssistantModel(cfg)
    load_gpt2_into(model, hf_model_id=args.hf_model_id)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "step":            0,
        "model_state":     model.state_dict(),
        "optimizer_state": None,                     # let trainer re-init
        "model_config":    asdict(cfg),
        "train_config":    {},                       # trainer overrides
        "metrics":         {"source": f"gpt2_init:{args.hf_model_id}"},
    }
    torch.save(payload, out_path)
    print(f"\nSaved GPT-2-initialized checkpoint to: {out_path}")
    print("Resume training with:")
    print(f"  python model/train.py --resume {out_path} --model-size 125M --preset 3090")


if __name__ == "__main__":
    main()
