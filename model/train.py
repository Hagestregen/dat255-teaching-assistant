"""
train.py  —  Training loop for the Teaching Assistant transformer
=================================================================
This script fine-tunes the model on the QA dataset using:
  - AdamW optimizer with weight decay
  - Cosine learning rate schedule with warmup
  - Gradient clipping to prevent exploding gradients
  - Periodic validation and checkpointing
  - Optional Weights & Biases (wandb) logging
  - Optional bf16 autocast and torch.compile for throughput
  - Resume-from-checkpoint and early stopping

HOW FINE-TUNING DIFFERS FROM TRAINING FROM SCRATCH:
─────────────────────────────────────────────────────
Training from scratch: start with random weights, train on a huge corpus
  (Wikipedia + books) to learn language in general.

Fine-tuning: start with (optionally pre-trained) weights, train on a
  task-specific dataset (QA pairs) with a small learning rate.

In our case: we train from scratch on the QA dataset. The model is small
enough (~25–50M params) that it can learn a consistent QA style in a few
hours on a single GPU.  See `model/init_from_gpt2.py` (Phase 8) for an
optional GPT-2-initialized variant for comparison.

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

PHASE 1 ADDITIONS (this revision)
──────────────────────────────────
  * bf16 autocast        ~2× throughput on Ampere+ (3090 etc.) for free
  * torch.compile        eager → graph optimization, single-line speedup
  * weight-decay fix     embeddings exempt by *name* (param dim alone is wrong)
  * resume-from-ckpt     `--resume PATH` continues a run end-to-end
  * presets              `--preset {debug, laptop, 3090}` for sane defaults
  * early stopping       `--early-stopping-patience N` watches val loss
  * curve logging        train/val pairs persisted to JSON for later plots
"""

import argparse
import math
import json
import time
from contextlib import nullcontext
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import torch
from torch.optim import AdamW

from transformer import TeachingAssistantModel, TransformerConfig
from dataset import Tokenizer, build_datasets, make_dataloader
from torch.utils.data import DataLoader


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: Training Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainingConfig:
    # Paths
    out_dir:              str   = "checkpoints"
    data_chunks_json:     str   = "rag_index/chunks.json"
    synthetic_jsonl:      str   = "data/synthetic_qa.jsonl"
    splits_dir:           str   = "data/splits"   # frozen splits (preferred)

    # Training loop
    max_steps:            int   = 5000
    batch_size:           int   = 16
    max_length:           int   = 512
    grad_accum_steps:     int   = 4

    # Optimizer
    learning_rate:        float = 3e-4
    min_lr:               float = 3e-5
    weight_decay:         float = 0.1
    beta1:                float = 0.9
    beta2:                float = 0.95
    max_grad_norm:        float = 1.0

    # LR schedule
    warmup_steps:         int   = 200
    decay_steps:          int   = 4500

    # Logging and saving
    log_interval:         int   = 10
    val_interval:         int   = 200
    save_interval:        int   = 500
    use_wandb:            bool  = False
    log_curves_path:      Optional[str] = None  # JSON of {train: [...], val: [...]}

    # Phase 1: throughput / loop hygiene
    mixed_precision:      str   = "none"        # "none" | "bf16"
    compile_model:        bool  = False         # torch.compile
    resume_from:          Optional[str] = None  # checkpoint to resume from
    early_stopping_patience: int = 0            # # val intervals; 0 = off

    # Phase 2: sequence packing (path wired here, dataset implementation in Phase 2)
    use_packing:          bool  = False
    rag_context_prob:     float = 0.0           # P(include context) at train time (Phase 3)

    # Phase 6: regularization
    ema_decay:            float = 0.0           # 0 = off; 0.999 typical

    # Data-size ablation (Phase A2 of the experiment plan)
    max_train_examples:   Optional[int] = None  # cap train set; 0/None = keep all

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
    """
    if step < cfg.warmup_steps:
        return cfg.learning_rate * (step + 1) / cfg.warmup_steps
    if step >= cfg.decay_steps:
        return cfg.min_lr
    progress = (step - cfg.warmup_steps) / (cfg.decay_steps - cfg.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: Optimizer setup
# ─────────────────────────────────────────────────────────────────────────────

# Substrings that mark a parameter as "should NOT be weight-decayed".
# Embeddings appear as `transformer.wte.weight` (and possibly `wpe`); we
# also catch generic `embedding` substrings for safety with future changes.
_NO_DECAY_NAME_PARTS = ("wte", "wpe", "embedding")


def _is_no_decay(name: str, param: torch.Tensor) -> bool:
    """A parameter is NOT weight-decayed if it is 1D OR is an embedding."""
    if param.dim() < 2:
        return True
    lower = name.lower()
    return any(part in lower for part in _NO_DECAY_NAME_PARTS)


def configure_optimizer(model: TeachingAssistantModel, cfg: TrainingConfig) -> AdamW:
    """
    Configure AdamW with weight decay applied only to certain parameters.

    Why AdamW?
    Adam: adaptive learning rates per parameter.
    W = Weight Decay: AdamW applies decay directly to weights (not via the
      gradient like classic L2 regularization), which empirically matters.

    Which parameters should NOT have weight decay?
    - Biases / LayerNorm / RMSNorm gains: 1D, decaying them hurts performance.
    - **Embeddings**: even though they are 2D matrices, decaying them
      degrades word representations.  Standard practice is to exempt them
      explicitly, by NAME, not by shape (this was a bug in the old code).
    """
    decay_params:    list[torch.Tensor] = []
    no_decay_params: list[torch.Tensor] = []
    decay_names:     list[str] = []
    no_decay_names:  list[str] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if _is_no_decay(name, param):
            no_decay_params.append(param)
            no_decay_names.append(name)
        else:
            decay_params.append(param)
            decay_names.append(name)

    n_decay     = sum(p.numel() for p in decay_params)
    n_no_decay  = sum(p.numel() for p in no_decay_params)
    print(f"Optimizer: {len(decay_params)} param tensors with decay "
          f"({n_decay/1e6:.1f}M params), "
          f"{len(no_decay_params)} without "
          f"({n_no_decay/1e6:.1f}M params)")
    if no_decay_names:
        print(f"  no-decay names: {no_decay_names}")

    optimizer = AdamW(
        [
            {"params": decay_params,    "weight_decay": cfg.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=cfg.learning_rate, betas=(cfg.beta1, cfg.beta2),
    )
    return optimizer


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: Mixed-precision context helper
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Phase 6 helper: Exponential Moving Average of model weights
# ─────────────────────────────────────────────────────────────────────────────

class EMA:
    """
    Exponential Moving Average of model parameters:

        θ_ema ← α · θ_ema + (1 - α) · θ_live    after every optimizer step.

    Evaluating with the EMA weights tends to give a small but consistent
    improvement (think "smoother" minimum) — typical α=0.999 means the EMA
    averages over ≈1000 last steps.

    Usage:
        ema = EMA(model, decay=0.999)
        ...
        optimizer.step()
        ema.update(model)
        ...
        with ema.swapped_into(model):
            val_metrics = evaluate(...)
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        assert 0.0 < decay < 1.0, f"EMA decay must be in (0, 1), got {decay}"
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.detach().clone()

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        d = self.decay
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            shadow = self.shadow.get(name)
            if shadow is None:
                continue
            shadow.mul_(d).add_(p.detach(), alpha=1.0 - d)

    def state_dict(self) -> dict:
        return {name: t.detach().cpu().clone() for name, t in self.shadow.items()}

    def load_state_dict(self, state: dict):
        for name, t in state.items():
            if name in self.shadow:
                self.shadow[name].copy_(t.to(self.shadow[name].device))

    class _Swap:
        def __init__(self, ema: "EMA", model: torch.nn.Module):
            self.ema = ema
            self.model = model
            self.backup: dict[str, torch.Tensor] = {}

        def __enter__(self):
            for name, p in self.model.named_parameters():
                if name in self.ema.shadow:
                    self.backup[name] = p.detach().clone()
                    p.data.copy_(self.ema.shadow[name])
            return self

        def __exit__(self, exc_type, exc, tb):
            for name, p in self.model.named_parameters():
                if name in self.backup:
                    p.data.copy_(self.backup[name])
            self.backup.clear()
            return False

    def swapped_into(self, model: torch.nn.Module) -> "EMA._Swap":
        return EMA._Swap(self, model)


def _amp_ctx(device: str, mixed_precision: str):
    """
    Return an autocast context manager (or a no-op nullcontext).
    bf16 autocast keeps weights/optimizer in fp32 but runs the forward/
    backward in bf16 — about 2× faster on Ampere+ and uses less memory.
    Only enabled on CUDA; bf16 on CPU/MPS is unreliable.
    """
    if mixed_precision == "bf16" and device == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: Evaluation (perplexity on validation set)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
# ─────────────────────────────────────────────────────────────────────────────
# Run-metadata helpers (used to make every W&B run self-describing for the
# project report).  Cheap and side-effect-free; failures fall back to None.
# ─────────────────────────────────────────────────────────────────────────────

def _git_sha(short: bool = True) -> Optional[str]:
    """Return the current git commit SHA (short form), or None if unavailable."""
    import subprocess
    try:
        cmd = ["git", "rev-parse", "--short" if short else "HEAD", "HEAD" if short else ""]
        cmd = [c for c in cmd if c]
        sha = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
        # Also flag if the working tree is dirty so report runs can be filtered.
        dirty = subprocess.call(["git", "diff", "--quiet"],
                                stderr=subprocess.DEVNULL) != 0
        return f"{sha}{'-dirty' if dirty else ''}"
    except Exception:
        return None


def _splits_manifest_summary(splits_dir: str) -> Optional[dict]:
    """Read `manifest.json` from the frozen-splits dir and return a compact dict."""
    if not splits_dir:
        return None
    path = Path(splits_dir)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent.parent / splits_dir
    manifest_path = path / "manifest.json"
    if not manifest_path.exists():
        return None
    try:
        m = json.loads(manifest_path.read_text())
        return {
            "splits_seed":     m.get("seed"),
            "splits_val_frac": m.get("val_frac"),
            "splits_test_frac": m.get("test_frac"),
            "splits_counts":   m.get("counts"),
            "splits_tasks":    m.get("tasks"),
            "splits_data_sha256": m.get("synthetic_jsonl_sha256"),
        }
    except Exception:
        return None


def _generate_samples(model, tokenizer, prompts: list[str], device: str,
                      max_new_tokens: int = 120, temperature: float = 0.7) -> list[str]:
    """
    Generate one completion per prompt and return decoded text (sans prompt).
    Used for the W&B `samples` table so the report can show qualitative
    progress every val step without needing to re-run inference.
    """
    raw = _unwrap(model)
    was_training = raw.training
    raw.eval()
    out: list[str] = []
    with torch.no_grad():
        for prompt in prompts:
            ids = torch.tensor([tokenizer.encode(prompt)], device=device, dtype=torch.long)
            gen = raw.generate(
                ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=50, top_p=0.95,
                stop_token=tokenizer.eot_id,
                use_cache=True,
            )
            new_ids = gen[0, ids.size(1):].tolist()
            # Trim at first eot_id if present so we don't show post-EOS noise.
            if tokenizer.eot_id in new_ids:
                new_ids = new_ids[:new_ids.index(tokenizer.eot_id)]
            out.append(tokenizer.decode(new_ids))
    if was_training:
        raw.train()
    return out


# Five fixed prompts (one per task, plus a no-task baseline) — kept identical
# across runs so the W&B sample table is directly comparable between models.
_DEFAULT_SAMPLE_PROMPTS = [
    "<|explain|> Question: What is overfitting and how do we detect it?\nAnswer:",
    "<|quiz|> Question: Which of the following best describes dropout?\n"
        "A) A way to mask out tokens at random\n"
        "B) A regularizer that randomly zeros activations\n"
        "C) A learning-rate schedule\n"
        "D) An optimizer\nAnswer:",
    "<|review|> Question: Explain the bias-variance tradeoff.\n"
        "Student answer: It's about the model being too simple or too complex.\nAnswer:",
    "<|flashcard|> Question: Define cross-entropy loss.\nAnswer:",
    "Question: How is gradient descent used to train a neural network?\nAnswer:",
]


def evaluate(model, val_loader, device: str, cfg: TrainingConfig, max_batches: int = 50) -> dict:
    """Compute validation loss and perplexity (autocast-aware)."""
    was_training = model.training
    model.eval()
    total_loss = 0.0
    n_batches  = 0

    for batch in val_loader:
        if n_batches >= max_batches:
            break
        input_ids = batch["input_ids"].to(device)
        labels    = batch["labels"].to(device)
        with _amp_ctx(device, cfg.mixed_precision):
            _, loss = model(input_ids, labels)
        if loss is not None:
            total_loss += loss.item()
            n_batches  += 1

    if was_training:
        model.train()
    avg_loss = total_loss / max(n_batches, 1)
    return {
        "val_loss":        avg_loss,
        "val_perplexity":  math.exp(avg_loss),
    }


def evaluate_per_task(model, val_ds, device: str, cfg: TrainingConfig,
                      batch_size: int = 16, max_batches_per_task: int = 50) -> dict:
    """
    Per-task-type validation loss + perplexity.

    Why we report this separately
    -----------------------------
    The training set has 4 task types (explanation, quiz, review, flashcard)
    and the model can perform well on average while being terrible at one
    of them.  The aggregate `val_loss` hides that.  We report per-task loss
    so the report can show, e.g., "the model is excellent at explanations
    (loss 1.2) but underfits quiz format (loss 3.4)".

    Implementation note: we use torch.utils.data.Subset to slice the
    already-tokenized val dataset by task type rather than re-tokenising.
    The aligned raw_examples list on QADataset gives us the per-index types.
    """
    from torch.utils.data import Subset

    raw = getattr(val_ds, "raw_examples", None)
    if not raw:
        return {}

    # Bucket dataset indices by example type.
    by_type: dict[str, list[int]] = {}
    for i, ex in enumerate(raw):
        t = ex.get("type", "explanation")
        by_type.setdefault(t, []).append(i)

    out: dict[str, float] = {}
    for task, idxs in sorted(by_type.items()):
        if not idxs:
            continue
        loader = DataLoader(
            Subset(val_ds, idxs), batch_size=batch_size,
            shuffle=False, num_workers=0, pin_memory=True,
        )
        m = evaluate(model, loader, device, cfg, max_batches=max_batches_per_task)
        out[f"val_loss_{task}"]       = m["val_loss"]
        out[f"val_perplexity_{task}"] = m["val_perplexity"]
        out[f"val_n_{task}"]          = len(idxs)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: Checkpoint save/load
# ─────────────────────────────────────────────────────────────────────────────

def _unwrap(model):
    """Return the underlying nn.Module from a torch.compile wrapper if any."""
    return getattr(model, "_orig_mod", model)


def save_checkpoint(model, optimizer, step: int, metrics: dict, cfg: TrainingConfig,
                    name: Optional[str] = None, ema: Optional["EMA"] = None) -> Path:
    """
    Save model weights, optimizer state, and training metadata.
    If `ema` is provided, also save the EMA shadow weights.
    """
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    if name is None:
        name = f"step_{step:06d}.pt"
    path = Path(cfg.out_dir) / name
    raw = _unwrap(model)
    payload = {
        "step":             step,
        "model_state":      raw.state_dict(),
        "optimizer_state":  optimizer.state_dict(),
        "model_config":     asdict(raw.config),
        "train_config":     asdict(cfg),
        "metrics":          metrics,
    }
    if ema is not None:
        payload["ema_state"] = ema.state_dict()
    torch.save(payload, path)
    print(f"  Checkpoint saved: {path}")
    return path


def load_checkpoint(path: str, device: str):
    """
    Load a checkpoint.  Returns (model, optimizer_state, step, metrics).
    Tolerates configs containing fields the current TransformerConfig
    doesn't know about (forward-compat).
    """
    ckpt = torch.load(path, map_location=device)
    cfg_dict  = dict(ckpt["model_config"])
    # Drop unknown keys so old checkpoints still load against newer code
    # (and vice versa).
    valid_keys = set(TransformerConfig.__dataclass_fields__.keys())
    extra = [k for k in cfg_dict if k not in valid_keys]
    if extra:
        print(f"  load_checkpoint: dropping unknown TransformerConfig fields: {extra}")
        for k in extra:
            cfg_dict.pop(k)
    model_cfg = TransformerConfig(**cfg_dict)
    model = TeachingAssistantModel(model_cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    return model, ckpt.get("optimizer_state"), ckpt.get("step", 0), ckpt.get("metrics", {})


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7: Main training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(model_config: Optional[TransformerConfig] = None,
          train_config:  Optional[TrainingConfig]   = None):
    """
    Full training loop with bf16 autocast, torch.compile, resume, and
    early stopping (all opt-in via TrainingConfig).
    """
    if train_config is None:
        train_config = TrainingConfig()
    if model_config is None:
        model_config = TransformerConfig()

    torch.manual_seed(train_config.seed)

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available()
              else "cpu")
    print(f"Training on: {device}")

    # ── Build datasets and dataloaders ────────────────────────────────────────
    tokenizer = Tokenizer()
    train_ds, val_ds, test_ds = build_datasets(
        chunks_json_path=train_config.data_chunks_json,
        synthetic_jsonl_path=train_config.synthetic_jsonl,
        tokenizer=tokenizer,
        max_length=train_config.max_length,
        rag_context_prob=train_config.rag_context_prob,
        max_train_examples=train_config.max_train_examples,
        splits_dir=train_config.splits_dir,
    )

    if train_config.use_packing:
        # Phase 2 wires this up; keep a clear error for now if not implemented.
        try:
            from dataset import PackedQADataset, make_packed_dataloader  # noqa: F401
            print("Sequence packing: enabled")
            packed_train = PackedQADataset(
                train_ds.raw_examples, tokenizer,
                max_length=train_config.max_length,
                rag_context_prob=train_config.rag_context_prob,
            )
            train_loader = make_packed_dataloader(packed_train, train_config.batch_size, shuffle=True)
        except ImportError:
            print("Sequence packing requested but PackedQADataset not available — falling back to unpacked.")
            train_loader = make_dataloader(train_ds, train_config.batch_size, shuffle=True)
    else:
        train_loader = make_dataloader(train_ds, train_config.batch_size, shuffle=True)

    val_loader = make_dataloader(val_ds, train_config.batch_size, shuffle=False)

    # ── Model and optimizer (with optional resume) ────────────────────────────
    start_step        = 0
    best_val_loss     = float("inf")
    best_ema_val_loss = float("inf")
    no_improve        = 0
    history           = {"train": [], "val": []}

    if train_config.resume_from:
        print(f"Resuming from: {train_config.resume_from}")
        model, opt_state, start_step, ckpt_metrics = load_checkpoint(
            train_config.resume_from, device
        )
        # Use the saved model_config to avoid drift with the run we're resuming.
        model_config = model.config
        optimizer = configure_optimizer(model, train_config)
        if opt_state is not None:
            try:
                optimizer.load_state_dict(opt_state)
                print(f"  Loaded optimizer state. Continuing at step {start_step}.")
            except Exception as e:
                print(f"  Warning: optimizer state could not be loaded ({e}). "
                      f"Re-initializing optimizer state.")
        if isinstance(ckpt_metrics, dict) and "best_val_loss" in ckpt_metrics:
            best_val_loss = float(ckpt_metrics["best_val_loss"])
            print(f"  Restored best_val_loss = {best_val_loss:.4f}")
    else:
        model     = TeachingAssistantModel(model_config).to(device)
        optimizer = configure_optimizer(model, train_config)

    # ── Optional EMA shadow weights (Phase 6) ─────────────────────────────────
    ema: Optional[EMA] = None
    if train_config.ema_decay > 0:
        ema = EMA(_unwrap(model), decay=train_config.ema_decay)
        # If we resumed and the checkpoint stored EMA state, load it
        if train_config.resume_from:
            try:
                _ckpt = torch.load(train_config.resume_from, map_location=device)
                if "ema_state" in _ckpt:
                    ema.load_state_dict(_ckpt["ema_state"])
                    print(f"  Loaded EMA shadow weights from {train_config.resume_from}")
                del _ckpt
            except Exception as e:
                print(f"  Warning: could not load EMA state: {e}")
        print(f"EMA enabled (decay = {train_config.ema_decay})")

    # ── Optional torch.compile ────────────────────────────────────────────────
    if train_config.compile_model:
        try:
            print("Compiling model with torch.compile (first step will be slow)…")
            model = torch.compile(model)
        except Exception as e:
            print(f"  torch.compile failed: {e}. Continuing without compilation.")

    # ── Mixed precision banner ────────────────────────────────────────────────
    use_amp = train_config.mixed_precision == "bf16" and device == "cuda"
    print(f"Mixed precision: {'bf16 autocast' if use_amp else 'fp32'}")

    # ── Optional W&B logging ──────────────────────────────────────────────────
    if train_config.use_wandb:
        import wandb
        # Compose a self-describing config so every W&B run captures git
        # state, dataset version and parameter counts.  This is what the
        # report cites — without it we can't reproduce results next month.
        n_params       = sum(p.numel() for p in _unwrap(model).parameters())
        n_train_params = sum(p.numel() for p in _unwrap(model).parameters() if p.requires_grad)
        run_meta = {
            "git_sha":                 _git_sha(),
            "n_params":                n_params,
            "n_trainable_params":      n_train_params,
            "n_train_examples":        len(getattr(train_ds, "raw_examples", [])),
            "n_val_examples":          len(getattr(val_ds,   "raw_examples", [])),
            "n_test_examples":         len(getattr(test_ds,  "raw_examples", [])),
            "tokenizer_vocab_size":    tokenizer.vocab_size,
        }
        manifest_meta = _splits_manifest_summary(train_config.splits_dir) or {}
        wandb.init(
            project="dat255-teaching-assistant",
            config={
                **asdict(model_config),
                **asdict(train_config),
                **run_meta,
                **manifest_meta,
            },
        )

    print(f"\nStarting training: {train_config.max_steps} steps "
          f"(starting at {start_step}), "
          f"batch_size={train_config.batch_size}, "
          f"grad_accum={train_config.grad_accum_steps}")
    print(f"Effective batch size: "
          f"{train_config.batch_size * train_config.grad_accum_steps}")
    n_params = sum(p.numel() for p in _unwrap(model).parameters())
    print(f"Model parameters: {n_params/1e6:.2f}M\n")

    step          = start_step
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
        for _micro_step in range(train_config.grad_accum_steps):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            input_ids = batch["input_ids"].to(device)
            labels    = batch["labels"].to(device)

            with _amp_ctx(device, train_config.mixed_precision):
                _, loss = model(input_ids, labels)

            # Scale loss so the effective gradient magnitude is consistent
            # regardless of accumulation steps.
            loss = loss / train_config.grad_accum_steps
            loss.backward()
            loss_accum += loss.item()

        # ── Gradient clipping ────────────────────────────────────────────────
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), train_config.max_grad_norm
        )

        # ── Optimizer step ───────────────────────────────────────────────────
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if ema is not None:
            ema.update(_unwrap(model))
        step += 1
        running_loss += loss_accum

        # ── Logging ──────────────────────────────────────────────────────────
        if step % train_config.log_interval == 0:
            elapsed     = time.time() - t0
            avg_loss    = running_loss / train_config.log_interval
            perplexity  = math.exp(avg_loss)
            tokens_per_sec = (
                train_config.batch_size * train_config.grad_accum_steps
                * train_config.max_length * train_config.log_interval
                / max(elapsed, 1e-6)
            )
            print(f"step {step:5d} | loss {avg_loss:.4f} | "
                  f"ppl {perplexity:.1f} | "
                  f"lr {lr:.2e} | "
                  f"grad_norm {grad_norm:.3f} | "
                  f"{tokens_per_sec:.0f} tok/s")

            history["train"].append({
                "step": step, "loss": avg_loss, "perplexity": perplexity,
                "lr": lr, "grad_norm": float(grad_norm),
                "tokens_per_sec": tokens_per_sec,
            })

            if train_config.use_wandb:
                import wandb
                wandb.log({
                    "train/loss": avg_loss, "train/perplexity": perplexity,
                    "train/lr": lr, "train/grad_norm": float(grad_norm),
                    "train/tokens_per_sec": tokens_per_sec,
                }, step=step)

            running_loss = 0.0
            t0 = time.time()

        # ── Validation ───────────────────────────────────────────────────────
        if step % train_config.val_interval == 0:
            # Live-weight metrics
            val_metrics = evaluate(model, val_loader, device, train_config)
            log_str = (f"  [VAL] loss {val_metrics['val_loss']:.4f} | "
                       f"ppl {val_metrics['val_perplexity']:.1f}")

            # Per-task breakdown (live weights).  Cheap on the small val set
            # and very useful for the report's "task strength" subsection.
            try:
                per_task = evaluate_per_task(
                    model, val_ds, device, train_config,
                    batch_size=train_config.batch_size,
                )
                val_metrics.update(per_task)
                # Compact one-line summary so the log stays readable.
                if per_task:
                    task_strs = [f"{k.replace('val_loss_',''):s}={v:.2f}"
                                 for k, v in per_task.items() if k.startswith("val_loss_")]
                    log_str += "  [task " + " ".join(task_strs) + "]"
            except Exception as e:
                print(f"  [warn] per-task eval failed: {e}")

            # Optional: also evaluate with EMA shadow weights and report both
            if ema is not None:
                with ema.swapped_into(_unwrap(model)):
                    ema_metrics = evaluate(model, val_loader, device, train_config)
                    try:
                        ema_per_task = evaluate_per_task(
                            model, val_ds, device, train_config,
                            batch_size=train_config.batch_size,
                        )
                    except Exception:
                        ema_per_task = {}
                val_metrics["ema_val_loss"]       = ema_metrics["val_loss"]
                val_metrics["ema_val_perplexity"] = ema_metrics["val_perplexity"]
                # Re-key the EMA per-task metrics so they don't collide.
                for k, v in ema_per_task.items():
                    val_metrics[k.replace("val_", "ema_val_")] = v
                log_str += (f" | EMA loss {ema_metrics['val_loss']:.4f} "
                            f"ppl {ema_metrics['val_perplexity']:.1f}")
            print(log_str)

            history["val"].append({"step": step, **val_metrics})
            if train_config.use_wandb:
                import wandb
                # W&B prefers nested namespaces ("val/loss_explanation") over
                # flat keys ("val_loss_explanation"); normalise here.
                wandb_payload = {}
                for k, v in val_metrics.items():
                    if k.startswith("val_"):
                        wandb_payload[f"val/{k[4:]}"] = v
                    elif k.startswith("ema_val_"):
                        wandb_payload[f"val_ema/{k[8:]}"] = v
                    else:
                        wandb_payload[k] = v
                wandb.log(wandb_payload, step=step)

                # Sample-generation table: 5 fixed prompts, one per task type
                # plus a no-task baseline.  Logged at every val step so the
                # W&B UI lets you scrub through training and watch outputs
                # improve.  This is the most direct qualitative evidence we
                # can include in the report.
                try:
                    samples = _generate_samples(
                        model, tokenizer, _DEFAULT_SAMPLE_PROMPTS, device,
                        max_new_tokens=120, temperature=0.7,
                    )
                    table = wandb.Table(columns=["step", "prompt", "completion"])
                    for prompt, completion in zip(_DEFAULT_SAMPLE_PROMPTS, samples):
                        table.add_data(step, prompt, completion)
                    wandb.log({"samples": table}, step=step)
                except Exception as e:
                    print(f"  [warn] sample generation failed: {e}")

            # Track best model and persist on improvement
            if val_metrics["val_loss"] < best_val_loss:
                best_val_loss = val_metrics["val_loss"]
                no_improve    = 0
                save_checkpoint(
                    model, optimizer, step,
                    {"best": True, "best_val_loss": best_val_loss, **val_metrics},
                    train_config, name="best.pt", ema=ema,
                )
            else:
                no_improve += 1
                if (train_config.early_stopping_patience > 0
                    and no_improve >= train_config.early_stopping_patience):
                    print(f"  Early stopping triggered at step {step}: "
                          f"no val improvement for {no_improve} validation intervals.")
                    break

            # Also track the best EMA val checkpoint so we can recover the
            # EMA snapshot at its peak (which is often a few hundred steps
            # past the live-val peak — see GPT-2 warm-start runs).
            ema_val = val_metrics.get("ema_val_loss")
            if ema is not None and ema_val is not None and ema_val < best_ema_val_loss:
                best_ema_val_loss = ema_val
                save_checkpoint(
                    model, optimizer, step,
                    {"best_ema": True,
                     "best_ema_val_loss": best_ema_val_loss,
                     **val_metrics},
                    train_config, name="best_ema.pt", ema=ema,
                )

        # ── Periodic checkpoint ──────────────────────────────────────────────
        if step % train_config.save_interval == 0:
            metrics = {"step": step, "lr": lr, "best_val_loss": best_val_loss}
            save_checkpoint(model, optimizer, step, metrics, train_config, ema=ema)

    # ── Final save and bookkeeping ────────────────────────────────────────────
    print("\nTraining complete!")
    final_path = save_checkpoint(
        model, optimizer, step,
        {"final": True, "best_val_loss": best_val_loss}, train_config,
        name="final.pt", ema=ema,
    )
    print(f"Final model saved to: {final_path}")

    tok_meta = {"vocab_size": tokenizer.vocab_size, "encoding": "gpt2_teaching"}
    with open(Path(train_config.out_dir) / "tokenizer_meta.json", "w") as f:
        json.dump(tok_meta, f)

    if train_config.log_curves_path:
        Path(train_config.log_curves_path).parent.mkdir(parents=True, exist_ok=True)
        with open(train_config.log_curves_path, "w") as f:
            json.dump(history, f, indent=2)
        print(f"Curves saved to: {train_config.log_curves_path}")

    return model


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8: Presets (training-recipe + model-size)
# ─────────────────────────────────────────────────────────────────────────────

# Training-recipe presets: tuned for typical hardware, not for small smoke tests.
TRAINING_PRESETS: dict[str, dict] = {
    "debug": dict(
        max_steps=200, batch_size=4, grad_accum_steps=2,
        max_length=256,
        log_interval=10, val_interval=50, save_interval=200,
        warmup_steps=20, decay_steps=180,
        mixed_precision="none", compile_model=False,
        early_stopping_patience=0,
    ),
    "laptop": dict(
        max_steps=2000, batch_size=4, grad_accum_steps=4,
        max_length=512,
        log_interval=20, val_interval=200, save_interval=500,
        warmup_steps=100, decay_steps=1900,
        mixed_precision="none", compile_model=False,
        early_stopping_patience=4,
    ),
    "3090": dict(
        max_steps=8000, batch_size=16, grad_accum_steps=4,
        max_length=512,
        log_interval=20, val_interval=400, save_interval=1000,
        warmup_steps=200, decay_steps=7500,
        mixed_precision="bf16", compile_model=True,
        early_stopping_patience=5,
    ),
}

# Model-size presets (parameter counts approximate, with weight tying).
MODEL_PRESETS: dict[str, dict] = {
    "25M":  dict(n_layer=6,  n_head=8,  n_embd=512, block_size=512, dropout=0.1),
    "50M":  dict(n_layer=8,  n_head=8,  n_embd=640, block_size=512, dropout=0.1),
    "125M": dict(n_layer=12, n_head=12, n_embd=768, block_size=512, dropout=0.1),
}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9: CLI
# ─────────────────────────────────────────────────────────────────────────────

def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train the Teaching Assistant transformer.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Training preset
    p.add_argument("--preset",     choices=list(TRAINING_PRESETS), default="laptop",
                   help="Training-recipe preset")
    p.add_argument("--model-size", choices=list(MODEL_PRESETS) + ["custom"], default="25M",
                   help="Model-size preset")

    # Training overrides
    p.add_argument("--max-steps",        type=int)
    p.add_argument("--batch-size",       type=int)
    p.add_argument("--grad-accum-steps", type=int)
    p.add_argument("--max-length",       type=int)
    p.add_argument("--learning-rate",    type=float)
    p.add_argument("--warmup-steps",     type=int)
    p.add_argument("--val-interval",     type=int)
    p.add_argument("--save-interval",    type=int)
    p.add_argument("--out-dir",          type=str)
    p.add_argument("--data-chunks-json", type=str)
    p.add_argument("--synthetic-jsonl",  type=str)
    p.add_argument("--splits-dir",       type=str,
                   help="Frozen train/val/test split directory (preferred over "
                        "--synthetic-jsonl). Default: data/splits")
    p.add_argument("--mixed-precision",  choices=["none", "bf16"])
    p.add_argument("--compile",          action="store_true", dest="compile_model",
                   help="Enable torch.compile")
    p.add_argument("--resume",           type=str, dest="resume_from",
                   help="Resume from checkpoint path")
    p.add_argument("--packed",           action="store_true", dest="use_packing",
                   help="Use sequence-packed dataset (Phase 2)")
    p.add_argument("--rag-context-prob", type=float,
                   help="P(include retrieved context in train prompts), Phase 3")
    p.add_argument("--ema-decay",        type=float, help="EMA decay (0=off)")
    p.add_argument("--max-train-examples", type=int,
                   help="Cap the train split size (for data-size ablations)")
    p.add_argument("--early-stopping-patience", type=int)
    p.add_argument("--use-wandb",        action="store_true")
    p.add_argument("--log-curves",       type=str, dest="log_curves_path",
                   help="Where to write JSON of train/val curves")
    p.add_argument("--seed",             type=int)

    # Model overrides
    p.add_argument("--n-layer",          type=int)
    p.add_argument("--n-head",           type=int)
    p.add_argument("--n-embd",           type=int)
    p.add_argument("--block-size",       type=int)
    p.add_argument("--dropout",          type=float)
    p.add_argument("--use-rmsnorm",      action="store_true",
                   help="Use RMSNorm instead of LayerNorm (Phase 4)")
    p.add_argument("--ffn-variant",      choices=["gelu", "swiglu"],
                   help="FFN variant (Phase 4)")
    p.add_argument("--label-smoothing",  type=float,
                   help="Cross-entropy label smoothing (Phase 6)")
    p.add_argument("--stochastic-depth", type=float,
                   help="Layer-drop probability for top block (Phase 6)")

    return p


def _build_configs_from_args(args: argparse.Namespace) -> tuple[TransformerConfig, TrainingConfig]:
    """Compose TrainingConfig + TransformerConfig from preset + CLI overrides."""
    train_kw = dict(TRAINING_PRESETS[args.preset])
    overrides_train = {
        "max_steps":               args.max_steps,
        "batch_size":              args.batch_size,
        "grad_accum_steps":        args.grad_accum_steps,
        "max_length":              args.max_length,
        "learning_rate":           args.learning_rate,
        "warmup_steps":            args.warmup_steps,
        "val_interval":            args.val_interval,
        "save_interval":           args.save_interval,
        "out_dir":                 args.out_dir,
        "data_chunks_json":        args.data_chunks_json,
        "synthetic_jsonl":         args.synthetic_jsonl,
        "splits_dir":              args.splits_dir,
        "mixed_precision":         args.mixed_precision,
        "compile_model":           args.compile_model or None,
        "resume_from":             args.resume_from,
        "use_packing":             args.use_packing or None,
        "rag_context_prob":        args.rag_context_prob,
        "ema_decay":               args.ema_decay,
        "max_train_examples":      args.max_train_examples,
        "early_stopping_patience": args.early_stopping_patience,
        "use_wandb":               args.use_wandb or None,
        "log_curves_path":         args.log_curves_path,
        "seed":                    args.seed,
    }
    for k, v in overrides_train.items():
        if v is not None:
            train_kw[k] = v
    train_cfg = TrainingConfig(**train_kw)

    if args.model_size == "custom":
        model_kw = {}
    else:
        model_kw = dict(MODEL_PRESETS[args.model_size])
    overrides_model = {
        "n_layer":          args.n_layer,
        "n_head":           args.n_head,
        "n_embd":           args.n_embd,
        "block_size":       args.block_size,
        "dropout":          args.dropout,
        "use_rmsnorm":      args.use_rmsnorm or None,
        "ffn_variant":      args.ffn_variant,
        "label_smoothing":  args.label_smoothing,
        "stochastic_depth": args.stochastic_depth,
    }
    for k, v in overrides_model.items():
        if v is not None:
            model_kw[k] = v
    model_cfg = TransformerConfig(**model_kw)

    return model_cfg, train_cfg


def main():
    args = _build_argparser().parse_args()
    model_cfg, train_cfg = _build_configs_from_args(args)
    print("=" * 72)
    print("Resolved TrainingConfig:")
    for k, v in asdict(train_cfg).items():
        print(f"  {k:25s} = {v}")
    print("Resolved TransformerConfig:")
    for k, v in asdict(model_cfg).items():
        print(f"  {k:25s} = {v}")
    print("=" * 72)
    train(model_cfg, train_cfg)


if __name__ == "__main__":
    main()
