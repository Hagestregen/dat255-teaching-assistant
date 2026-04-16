"""
train.py  —  Training loop for the Teaching Assistant transformer
=================================================================
This script fine-tunes the model on the QA dataset using:
  - AdamW optimizer with weight decay
  - Cosine learning rate schedule with warmup
  - Gradient clipping to prevent exploding gradients
  - Periodic validation and checkpointing
  - Optional Weights & Biases (wandb) logging

HOW FINE-TUNING DIFFERS FROM TRAINING FROM SCRATCH:
─────────────────────────────────────────────────────
Training from scratch: start with random weights, train on a huge corpus
  (Wikipedia + books) to learn language in general.

Fine-tuning: start with (optionally pre-trained) weights, train on a
  task-specific dataset (QA pairs) with a small learning rate.

In our case: we train from scratch on the QA dataset. The model is small
enough (6-8M params) that it can learn a consistent QA style in ~1-2 hours
on a GPU. If you want better initial language quality, you could start from
GPT-2 weights (like your nanoGPT test notebook does) and fine-tune — but
for the report, training from scratch satisfies the "own model" requirement.

LEARNING RATE SCHEDULE — WHY COSINE WITH WARMUP?
──────────────────────────────────────────────────
Constant LR: simple but suboptimal. Too high → diverge. Too low → slow.

Warmup: start with a tiny LR (near 0), linearly increase to max_lr over
  the first warmup_steps. The model's weights are random at the start, so
  a big LR would shoot them into a bad region immediately.

Cosine decay: after warmup, decay LR following a cosine curve from max_lr
  down to min_lr. The cosine shape means it decreases quickly in the middle
  (fast progress) and slowly near the end (careful refinement near the optimum).
  This outperforms linear decay in practice.

GRADIENT CLIPPING — WHY?
──────────────────────────
If gradients become very large (can happen with long sequences or bad batch),
the optimizer would make an enormous step and likely diverge. Clipping the
gradient norm to max_grad_norm (typically 1.0) prevents this.

It works by computing the global gradient norm (like the length of a vector
of all gradients concatenated), and if it's > threshold, scaling ALL gradients
down proportionally so the norm equals the threshold exactly.
"""

import os
import time
import math
import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

import torch
from torch.optim import AdamW

from transformer import TeachingAssistantModel, TransformerConfig
from dataset import Tokenizer, build_datasets, make_dataloader


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: Training Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainingConfig:
    # Paths
    out_dir:              str   = "checkpoints"
    data_chunks_json:     str   = "rag_index/chunks.json"
    synthetic_jsonl:      str   = "data/synthetic_qa.jsonl"

    # Training loop
    max_steps:            int   = 5000    # total optimizer steps
    batch_size:           int   = 16      # sequences per batch
    max_length:           int   = 512     # context window (must match model)
    grad_accum_steps:     int   = 4       # accumulate gradients before stepping
                                          # effective_batch = batch_size * grad_accum_steps

    # Optimizer
    learning_rate:        float = 3e-4    # peak LR after warmup
    min_lr:               float = 3e-5    # LR at end of cosine decay (= 10% of max)
    weight_decay:         float = 0.1     # L2 penalty on weights (not biases/norms)
    beta1:                float = 0.9     # Adam β₁
    beta2:                float = 0.95    # Adam β₂ (slightly different from default 0.999)
    max_grad_norm:        float = 1.0     # gradient clipping threshold

    # LR schedule
    warmup_steps:         int   = 200     # steps to linearly increase from 0 to lr
    decay_steps:          int   = 4500    # steps to cosine-decay from lr to min_lr

    # Logging and saving
    log_interval:         int   = 10      # log loss every N steps
    val_interval:         int   = 200     # evaluate on val set every N steps
    save_interval:        int   = 500     # save checkpoint every N steps
    use_wandb:            bool  = False   # set True if you want W&B logging

    # Reproducibility
    seed:                 int   = 42


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: LR schedule
# ─────────────────────────────────────────────────────────────────────────────

def get_lr(step: int, cfg: TrainingConfig) -> float:
    """
    Compute learning rate at a given step.

    Phase 1 (0 → warmup_steps):      linear ramp 0 → max_lr
    Phase 2 (warmup → decay_steps):  cosine decay max_lr → min_lr
    Phase 3 (decay_steps → ∞):       constant min_lr

    The cosine formula:
        lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(π * progress))
    where progress goes from 0 to 1 during the decay phase.
    At progress=0: lr = max_lr
    At progress=1: lr = min_lr
    The cosine curve gives a smooth S-shape that works better than linear.
    """
    if step < cfg.warmup_steps:
        # Linear warmup
        return cfg.learning_rate * (step + 1) / cfg.warmup_steps

    if step >= cfg.decay_steps:
        # After decay: hold at min_lr
        return cfg.min_lr

    # Cosine decay phase
    progress = (step - cfg.warmup_steps) / (cfg.decay_steps - cfg.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: Optimizer setup
# ─────────────────────────────────────────────────────────────────────────────

def configure_optimizer(model: TeachingAssistantModel, cfg: TrainingConfig) -> AdamW:
    """
    Configure AdamW with weight decay applied only to certain parameters.

    Why AdamW?
    Adam: adaptive learning rates per parameter (great for sparse gradients,
      handles different scales in different layers).
    W = Weight Decay: adds L2 regularization (keeps weights small, reduces
      overfitting). AdamW fixes a bug in Adam's L2 implementation — it applies
      the decay directly to weights, not via the gradient.

    Which parameters should NOT have weight decay?
    - Biases: 1D parameters, decaying them hurts performance.
    - LayerNorm gains/biases: same reason.
    - Embeddings: decaying embeddings can degrade word representations.

    Which SHOULD have weight decay?
    - All 2D+ weights (linear layers, attention projections, FFN weights).
    """
    decay_params    = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # 2D params: weight matrices → apply weight decay
        # 1D params: biases, layernorm → no weight decay
        if param.dim() >= 2:
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    print(f"Optimizer: {len(decay_params)} param tensors with decay, "
          f"{len(no_decay_params)} without")

    optimizer = AdamW([
        {"params": decay_params,    "weight_decay": cfg.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ], lr=cfg.learning_rate, betas=(cfg.beta1, cfg.beta2))

    return optimizer


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: Evaluation (perplexity on validation set)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, val_loader, device: str, max_batches: int = 50) -> dict:
    """
    Compute validation loss and perplexity.

    Perplexity = exp(cross_entropy_loss).

    A perplexity of N roughly means: on average, the model is as uncertain
    as if it were choosing uniformly from N options at each token.
    - Random model over 50k vocab: perplexity ~50,000
    - Well-trained small model on QA: perplexity 5-20 is reasonable
    - GPT-2 on general text: perplexity ~35

    We run at most max_batches to keep evaluation fast.
    """
    model.eval()
    total_loss = 0.0
    n_batches  = 0

    for batch in val_loader:
        if n_batches >= max_batches:
            break
        input_ids = batch["input_ids"].to(device)
        labels    = batch["labels"].to(device)
        _, loss   = model(input_ids, labels)
        if loss is not None:
            total_loss += loss.item()
            n_batches  += 1

    avg_loss = total_loss / max(n_batches, 1)
    return {
        "val_loss":        avg_loss,
        "val_perplexity":  math.exp(avg_loss),
    }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: Checkpoint save/load
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(model, optimizer, step: int, metrics: dict, cfg: TrainingConfig):
    """Save model weights, optimizer state, and training metadata."""
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    path = Path(cfg.out_dir) / f"step_{step:06d}.pt"
    torch.save({
        "step":          step,
        "model_state":   model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "model_config":  asdict(model.config),
        "train_config":  asdict(cfg),
        "metrics":       metrics,
    }, path)
    print(f"  Checkpoint saved: {path}")
    return path


def load_checkpoint(path: str, device: str):
    """Load a checkpoint. Returns (model, optimizer_state, step, metrics)."""
    ckpt = torch.load(path, map_location=device)
    model_cfg = TransformerConfig(**ckpt["model_config"])
    model = TeachingAssistantModel(model_cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    return model, ckpt["optimizer_state"], ckpt["step"], ckpt.get("metrics", {})


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: Main training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(model_config: TransformerConfig = None, train_config: TrainingConfig = None):
    """
    Full training loop.

    Gradient accumulation:
    If your GPU doesn't have enough memory for a large batch, you can simulate
    a larger batch by running N smaller batches and accumulating (summing) the
    gradients before calling optimizer.step(). This is what grad_accum_steps does.

    The training loop structure:
      for each micro-batch:
          forward → compute loss / grad_accum_steps (scale for consistency)
          backward → accumulate gradients
          if accum steps done:
              clip gradients
              optimizer step
              zero gradients
              update LR
    """
    torch.manual_seed(train_config.seed)

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Training on: {device}")

    if model_config is None:
        model_config = TransformerConfig()
    if train_config is None:
        train_config = TrainingConfig()

    # ── Build datasets and dataloaders ────────────────────────────────────────
    tokenizer = Tokenizer()
    train_ds, val_ds, test_ds = build_datasets(
        chunks_json_path=train_config.data_chunks_json,
        synthetic_jsonl_path=train_config.synthetic_jsonl,
        tokenizer=tokenizer,
        max_length=train_config.max_length,
    )

    train_loader = make_dataloader(train_ds, train_config.batch_size, shuffle=True)
    val_loader   = make_dataloader(val_ds,   train_config.batch_size, shuffle=False)

    # ── Model and optimizer ───────────────────────────────────────────────────
    model     = TeachingAssistantModel(model_config).to(device)
    optimizer = configure_optimizer(model, train_config)

    # Optional: W&B logging
    if train_config.use_wandb:
        import wandb
        wandb.init(project="dat255-teaching-assistant",
                   config={**asdict(model_config), **asdict(train_config)})

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"\nStarting training: {train_config.max_steps} steps, "
          f"batch_size={train_config.batch_size}, "
          f"grad_accum={train_config.grad_accum_steps}")
    print(f"Effective batch size: {train_config.batch_size * train_config.grad_accum_steps}\n")

    step          = 0
    running_loss  = 0.0
    t0            = time.time()
    train_iter    = iter(train_loader)

    model.train()

    while step < train_config.max_steps:
        # ── Update learning rate ─────────────────────────────────────────────
        lr = get_lr(step, train_config)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # ── Gradient accumulation loop ───────────────────────────────────────
        loss_accum = 0.0
        for micro_step in range(train_config.grad_accum_steps):
            # Get next batch (cycle through dataset)
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            input_ids = batch["input_ids"].to(device)
            labels    = batch["labels"].to(device)

            # Forward pass
            _, loss = model(input_ids, labels)

            # Scale loss by grad_accum_steps so the effective gradient magnitude
            # is the same regardless of accumulation steps
            loss = loss / train_config.grad_accum_steps
            loss.backward()
            loss_accum += loss.item()

        # ── Gradient clipping ─────────────────────────────────────────────────
        # Compute the global norm of all gradients; clip if above threshold.
        # Returns the actual norm (useful for monitoring).
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), train_config.max_grad_norm
        )

        # ── Optimizer step ────────────────────────────────────────────────────
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)  # set_to_none saves a tiny bit of memory
        step += 1

        running_loss += loss_accum

        # ── Logging ───────────────────────────────────────────────────────────
        if step % train_config.log_interval == 0:
            elapsed = time.time() - t0
            avg_loss = running_loss / train_config.log_interval
            perplexity = math.exp(avg_loss)
            tokens_per_sec = (train_config.batch_size * train_config.grad_accum_steps
                              * train_config.max_length * train_config.log_interval / elapsed)

            print(f"step {step:5d} | loss {avg_loss:.4f} | "
                  f"ppl {perplexity:.1f} | "
                  f"lr {lr:.2e} | "
                  f"grad_norm {grad_norm:.3f} | "
                  f"{tokens_per_sec:.0f} tok/s")

            if train_config.use_wandb:
                import wandb
                wandb.log({"train/loss": avg_loss, "train/perplexity": perplexity,
                           "train/lr": lr, "train/grad_norm": grad_norm}, step=step)

            running_loss = 0.0
            t0 = time.time()

        # ── Validation ────────────────────────────────────────────────────────
        if step % train_config.val_interval == 0:
            val_metrics = evaluate(model, val_loader, device)
            print(f"  [VAL] loss {val_metrics['val_loss']:.4f} | "
                  f"ppl {val_metrics['val_perplexity']:.1f}")

            if train_config.use_wandb:
                import wandb
                wandb.log(val_metrics, step=step)

            model.train()

        # ── Checkpoint ────────────────────────────────────────────────────────
        if step % train_config.save_interval == 0:
            metrics = {"step": step, "lr": lr}
            save_checkpoint(model, optimizer, step, metrics, train_config)

    # ── Final save ────────────────────────────────────────────────────────────
    print("\nTraining complete!")
    final_path = save_checkpoint(model, optimizer, step,
                                 {"final": True}, train_config)
    print(f"Final model saved to: {final_path}")

    # Save the tokenizer config alongside (for inference convenience)
    tok_meta = {"vocab_size": tokenizer.vocab_size, "encoding": "gpt2"}
    with open(Path(train_config.out_dir) / "tokenizer_meta.json", "w") as f:
        json.dump(tok_meta, f)

    return model


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model_cfg = TransformerConfig(
        n_layer=6,      # 6 by default
        n_head=8,       # 8 by default
        n_embd=512,     # 512 by default
        block_size=512, # 512 by default
        dropout=0.1,    # 0.1 by default
    )

    train_cfg = TrainingConfig(
        max_steps=100,          # 5000 steps by default
        batch_size=8,           # 16 by default
        grad_accum_steps=4,     # effective batch = 32 by default
        learning_rate=3e-4,     # 3e-4 by default
        warmup_steps=200,       # 200 by default
        val_interval=250,       # 200 by default
        save_interval=500,      # 500 by default
        use_wandb=False,        # False by default
        out_dir="checkpoints",  # checkpoints by default
    )

    train(model_cfg, train_cfg)
