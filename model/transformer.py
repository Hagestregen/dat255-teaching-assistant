"""
model.py  —  Decoder-only Transformer with Rotary Positional Embeddings (RoPE)
===============================================================================
This is the "brain" of the teaching assistant. It is a GPT-style model:
tokens go in, logits (raw scores over the vocabulary) come out.

Architecture overview
─────────────────────
  Input tokens (integers)
       │
  [Token Embedding]          ← lookup table: int → n_embd-dim float vector
       │
  N × [Transformer Block]    ← each block = attention + feedforward
       │
  [LayerNorm]
       │
  [LM Head]                  ← linear: n_embd → vocab_size logits

What makes this different from vanilla GPT?
───────────────────────────────────────────
We use RoPE (Rotary Positional Embeddings) instead of learned absolute
positional embeddings. See the RotaryEmbedding class below for a full
explanation of why and how.

Why a decoder-only model for QA?
─────────────────────────────────
An encoder (like BERT) reads the whole sentence at once — great for
classification. A decoder generates text left-to-right, which is what we
need: given a question, *produce* an answer token by token.
"""

import math
from dataclasses import dataclass
from typing import Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TransformerConfig:
    """
    All hyperparameters in one place. Change these to scale the model.

    Rule of thumb for laptop/Colab training:
      n_layer=4, n_head=4, n_embd=256  → ~8M params, trains in minutes on CPU
      n_layer=6, n_head=8, n_embd=512  → ~50M params, needs a GPU, ~hours
    """
    vocab_size: int   = 50_261   # GPT-2 tokenizer (tiktoken 'gpt2' encoding) + 4 task tokens
    n_layer:    int   = 6        # number of transformer blocks stacked
    n_head:     int   = 8        # number of attention heads per block
    n_embd:     int   = 512      # embedding dimension (must be divisible by n_head)
    block_size: int   = 512      # maximum context window in tokens
    dropout:    float = 0.1      # dropout probability (set 0.0 at inference)
    bias:       bool  = False    # False = slightly faster, usually fine

    # Phase 4: free architecture upgrades (RMSNorm, SwiGLU)
    use_rmsnorm:     bool  = False    # True = RMSNorm everywhere (LLaMA-style)
    ffn_variant:     str   = "gelu"   # "gelu" (default) | "swiglu"

    # Phase 6: regularization
    label_smoothing: float = 0.0      # cross-entropy label smoothing (0 = off)
    stochastic_depth: float = 0.0     # max layer-drop prob (0 = off)

    def __post_init__(self):
        assert self.n_embd % self.n_head == 0, (
            f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})"
        )
        assert self.ffn_variant in ("gelu", "swiglu"), (
            f"ffn_variant must be 'gelu' or 'swiglu', got {self.ffn_variant!r}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: Rotary Positional Embeddings (RoPE)
# ─────────────────────────────────────────────────────────────────────────────

class RotaryEmbedding(nn.Module):
    """
    WHAT IS RoPE AND WHY USE IT?
    ─────────────────────────────
    The problem: attention is "permutation-invariant" by default. If you
    shuffle the input tokens, the attention scores are identical — the model
    can't tell WHERE each token sits in the sequence. We must inject position
    information somehow.

    Classic approach (GPT-2, original Transformer):
        token_embedding[i] += position_embedding[i]
    Add a learned vector to each position. Works but has a fixed max length.

    RoPE approach: instead of changing the embeddings, ROTATE the Query and
    Key vectors inside attention using angles that depend on each token's
    position. The key insight: when you compute Q·K (the attention score),
    the rotation difference between two tokens encodes their RELATIVE distance
    automatically — regardless of where they appear in absolute terms.

    Why RoPE is better for this project:
      1. No extra parameters (deterministic rotation matrix).
      2. Encodes relative position, which is what attention actually needs.
      3. Can generalize to longer sequences at inference (e.g. longer RAG contexts).
      4. Standard in LLaMA, Mistral, Gemma — you're learning modern practice.

    HOW THE ROTATION WORKS (the math, demystified):
    ─────────────────────────────────────────────────
    Each attention head has dimension head_dim. Think of this as head_dim/2
    pairs of numbers: (x₀, x₁), (x₂, x₃), (x₄, x₅), ...

    For a token at position p, rotate each pair i by angle  p × θᵢ:
        x'₂ᵢ   = x₂ᵢ   × cos(p θᵢ)  −  x₂ᵢ₊₁ × sin(p θᵢ)
        x'₂ᵢ₊₁ = x₂ᵢ   × sin(p θᵢ)  +  x₂ᵢ₊₁ × cos(p θᵢ)

    where θᵢ = 1 / (10000^(2i / head_dim))   — a geometric sequence of
    frequencies. Low-frequency pairs encode coarse position (sentence-level),
    high-frequency pairs encode fine position (token-level). The model can
    read both scales.

    The beautiful property: when we compute Q·K, the dot product of two
    rotated vectors naturally becomes a function of (position_q − position_k):
        rotate(q, p) · rotate(k, p') = f(q, k, p − p')
    So the model learns relative distances for free.

    HOW RoPE INTERACTS WITH RAG:
    ──────────────────────────────
    When we prepend retrieved chunks as context to the question, the combined
    sequence is: [chunk₁ tokens] [chunk₂ tokens] [Question: ... Answer:]
    RoPE encodes position within THIS combined sequence. The model sees the
    chunks at positions 0..N and the question at positions N+1..M.
    This is fine — the model learns during fine-tuning that the relevant
    context appears before the question marker. You do NOT need to reset
    positions for each chunk; the continuous numbering is intentional.
    """

    def __init__(self, head_dim: int, max_seq_len: int):
        super().__init__()
        assert head_dim % 2 == 0, "head_dim must be even for RoPE"

        # Base frequencies: θᵢ = 1 / 10000^(2i / head_dim)
        # Shape: (head_dim/2,)
        theta = 1.0 / (10_000 ** (torch.arange(0, head_dim, 2).float() / head_dim))

        # All positions 0 .. max_seq_len-1
        positions = torch.arange(max_seq_len).float()

        # freqs[p, i] = p * θᵢ  —  shape: (max_seq_len, head_dim/2)
        freqs = torch.outer(positions, theta)

        # Duplicate each frequency for the pair (cos applies to both dims in a pair)
        # cos_table[p] has shape (head_dim,)
        cos_table = freqs.cos().repeat_interleave(2, dim=-1)
        sin_table = freqs.sin().repeat_interleave(2, dim=-1)

        # Register as non-trainable buffers (moved with model.to(device) automatically)
        self.register_buffer("cos_table", cos_table)
        self.register_buffer("sin_table", sin_table)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """
        Helper for the 2D rotation formula.

        Given x = [x₀, x₁, x₂, x₃, ...], returns [-x₁, x₀, -x₃, x₂, ...]
        Combined with:  x' = x * cos + rotate_half(x) * sin
        this implements the full 2D rotation per pair.
        """
        x1 = x[..., 0::2]  # even indices
        x2 = x[..., 1::2]  # odd indices
        return torch.stack([-x2, x1], dim=-1).flatten(-2)

    def forward(self, q: torch.Tensor, k: torch.Tensor, start_pos: int = 0):
        """
        Apply rotary embeddings to query and key tensors.

        q, k: (batch, n_head, seq_len, head_dim)
        Returns rotated (q_rot, k_rot) with the same shape.

        `start_pos` lets a caller advance into the rotation tables — needed
        for KV-cached generation, where we feed only the latest token as a
        length-1 sequence but its true position is (cache_len + 0).

        Note: we do NOT rotate V (values). Values carry content, not position.
        """
        T = q.size(2)
        cos = self.cos_table[start_pos:start_pos + T].unsqueeze(0).unsqueeze(0)
        sin = self.sin_table[start_pos:start_pos + T].unsqueeze(0).unsqueeze(0)
        q_rot = q * cos + self._rotate_half(q) * sin
        k_rot = k * cos + self._rotate_half(k) * sin
        return q_rot, k_rot


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: Causal Self-Attention
# ─────────────────────────────────────────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    """
    Multi-head causal (masked) self-attention with RoPE.

    "Causal" means: token at position i can ONLY attend to positions 0..i.
    No peeking at the future. Enforced via is_causal=True in SDPA, which
    masks future scores to -∞ before softmax → they become probability 0.

    Why multiple heads? Each head can specialize in different relationships:
    - Head A might focus on "what word came 2 tokens ago?"
    - Head B might focus on "what semantically similar word is nearby?"
    - Head C might focus on subject-verb agreement across long distances.
    The outputs are concatenated, so the model gets all views simultaneously.

    We use PyTorch 2.0's scaled_dot_product_attention (SDPA), which dispatches
    to Flash Attention when possible — O(n) memory instead of O(n²).
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.n_head   = config.n_head
        self.n_embd   = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout  = config.dropout

        # One linear for Q, K, V together (efficiency)
        self.c_attn    = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj    = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.resid_drop = nn.Dropout(config.dropout)

        self.rope = RotaryEmbedding(self.head_dim, config.block_size)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass with optional KV cache.

        kv_cache: (cached_k, cached_v) of shape (B, n_head, cached_len, head_dim)
                  or None.  When present, the new token's K/V are appended to
                  the cache and the cache is returned (or None when disabled).

        Returns (output, new_kv_cache).
        """
        B, T, C = x.shape

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        def split_heads(t):
            return t.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)

        # Position offset = number of tokens already cached.  RoPE rotates
        # the *new* tokens at their absolute positions, not at 0..T-1.
        start_pos = kv_cache[0].size(2) if kv_cache is not None else 0
        q, k = self.rope(q, k, start_pos=start_pos)

        if kv_cache is not None:
            cached_k, cached_v = kv_cache
            k = torch.cat([cached_k, k], dim=2)
            v = torch.cat([cached_v, v], dim=2)
            # When the new query length is 1 (decode step) and the key length
            # is larger, SDPA's is_causal=True would mask everything but the
            # very first key — which is wrong.  Disable causal mask: the new
            # query token at position cache_len naturally attends to all of
            # 0..cache_len, none of which are in the future.
            is_causal = (T == k.size(2))
        else:
            is_causal = True
        # Always seed the cache from the current (k, v) so that the very first
        # forward pass under `use_cache=True` produces a usable cache for the
        # next decode step.  Setting this to None on the first call broke
        # generation (caches[0][0].size(2) crashed in `generate`).
        new_cache = (k, v)

        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.c_proj(y)), new_cache


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: Normalization layers (LayerNorm vs. RMSNorm)
# ─────────────────────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    """
    Root-Mean-Square Layer Normalization (Zhang & Sennrich 2019).

    RMSNorm replaces LayerNorm with a simpler operation: it just rescales by
    the RMS of the activations and multiplies by a learned gain.  No mean
    subtraction, no learned bias.  Compared to LayerNorm:
      - ~7–10 % faster (one less reduction)
      - empirically equivalent (or slightly better) quality
      - used in LLaMA, Mistral, Gemma, Qwen, and basically every modern LLM

    Formula:  y = x / sqrt(mean(x²) + eps) * gain
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps    = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute RMS in fp32 for stability when running under bf16 autocast.
        x_fp32 = x.float()
        rms = x_fp32.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x_fp32 * rms).type_as(x) * self.weight


def _make_norm(config: TransformerConfig) -> nn.Module:
    """Construct LayerNorm or RMSNorm based on config.use_rmsnorm."""
    if getattr(config, "use_rmsnorm", False):
        return RMSNorm(config.n_embd)
    return nn.LayerNorm(config.n_embd)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: Feed-Forward Network — GELU and SwiGLU variants
# ─────────────────────────────────────────────────────────────────────────────

class FeedForward(nn.Module):
    """
    GELU FFN (the original Transformer / GPT-2 design).

    Position-wise FFN: applied independently to each token after attention.
    Attention = "which tokens should I look at?"
    FFN       = "now that I know, what do I do with that information?"

    Most factual knowledge is thought to be stored in the FFN weights
    (each neuron acts like a fuzzy key-value pair).

    Hidden dim = 4 × n_embd is the standard from the original Transformer paper.
    GELU activation: smoother than ReLU, better for transformers in practice.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        self.act     = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(self.act(self.c_fc(x))))


class SwiGLUFeedForward(nn.Module):
    """
    SwiGLU FFN (Shazeer 2020), now standard in LLaMA and friends.

    Architecture: two parallel projections, gate and value, multiplied
    elementwise after a SiLU activation on the gate, then projected down:

        out = c_proj( SiLU(w_gate(x)) * w_value(x) )

    The hidden dim is set to round(8/3 · n_embd) (rounded up to a multiple
    of 64 for hardware-friendliness), which keeps the *parameter count*
    identical to a 4·n_embd GELU FFN — so SwiGLU is a strict quality upgrade
    at the same cost.

    The down-projection is named `c_proj` to share the same scaled init as
    the GELU FFN (see `_init_weights` post-hook).
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        # 8/3 ≈ 2.67 keeps the param count equal to the 4·n_embd GELU variant.
        # Round up to a multiple of 64 for tensor-core friendliness.
        target  = int(round(8 * config.n_embd / 3))
        hidden  = ((target + 63) // 64) * 64
        self.w_gate  = nn.Linear(config.n_embd, hidden, bias=config.bias)
        self.w_value = nn.Linear(config.n_embd, hidden, bias=config.bias)
        self.c_proj  = nn.Linear(hidden, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(F.silu(self.w_gate(x)) * self.w_value(x)))


def _make_ffn(config: TransformerConfig) -> nn.Module:
    """Construct the FFN variant requested by config.ffn_variant."""
    variant = getattr(config, "ffn_variant", "gelu")
    if variant == "swiglu":
        return SwiGLUFeedForward(config)
    return FeedForward(config)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: Transformer Block
# ─────────────────────────────────────────────────────────────────────────────

class Block(nn.Module):
    """
    One transformer block.

    Structure: Pre-LayerNorm → Attention → Residual
               Pre-LayerNorm → FFN       → Residual

    Residual connections: x = x + f(norm(x))
    Without residuals, gradients vanish in deep networks. With them, each
    block learns a small *additive correction* to the token representations,
    and gradients can flow straight back to the input.

    Pre-norm (normalize BEFORE the sub-layer, not after): more stable training
    than post-norm, used in GPT-2 and all modern transformer variants.
    """

    def __init__(self, config: TransformerConfig, layer_idx: int = 0, n_layer: Optional[int] = None):
        super().__init__()
        self.ln_1 = _make_norm(config)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = _make_norm(config)
        self.ffn  = _make_ffn(config)

        # Stochastic depth (Phase 6): linear schedule from 0 (first layer) to
        # config.stochastic_depth (last layer).  Only active during training.
        nl = n_layer if n_layer is not None else config.n_layer
        max_p = getattr(config, "stochastic_depth", 0.0)
        self.drop_prob = max_p * (layer_idx + 1) / max(nl, 1)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # Stochastic depth: with prob drop_prob, skip this block entirely
        # during training.  Disabled at eval and when KV-caching (we still
        # need to advance the cache).
        if (self.training
                and self.drop_prob > 0
                and kv_cache is None
                and torch.rand((), device=x.device).item() < self.drop_prob):
            return x, None

        # When kept under stochastic depth, the survivors compensate by
        # scaling their contribution (timm-style DropPath).
        scale = 1.0 / (1.0 - self.drop_prob) if (self.training and self.drop_prob > 0) else 1.0

        attn_out, new_cache = self.attn(self.ln_1(x), kv_cache=kv_cache)
        x = x + attn_out * scale
        x = x + self.ffn(self.ln_2(x)) * scale
        return x, new_cache


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7: Full Model
# ─────────────────────────────────────────────────────────────────────────────

class TeachingAssistantModel(nn.Module):
    """
    The complete decoder-only transformer.

    Weight tying:
    The token embedding matrix (vocab_size × n_embd) and the LM head
    (n_embd → vocab_size) share the SAME weights. This:
      - Cuts ~25M parameters for vocab_size=50257, n_embd=512.
      - Makes sense intuitively: the embedding and un-embedding of a token
        should represent the same "meaning."
      - Is standard practice (GPT-2, LLaMA, etc.)
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict({
            'wte':  nn.Embedding(config.vocab_size, config.n_embd),  # word token embeddings
            'drop': nn.Dropout(config.dropout),
            'h':    nn.ModuleList([Block(config, layer_idx=i, n_layer=config.n_layer)
                                   for i in range(config.n_layer)]),
            'ln_f': _make_norm(config),                               # final norm
        })

        # LM head projects n_embd → vocab_size. Tied with wte.
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer['wte'].weight = self.lm_head.weight  # weight tying

        # Initialize all weights
        self.apply(self._init_weights)
        # Special scaled-down init for output projections (GPT-2 trick):
        # prevents the residual stream from growing too large with depth.
        for name, p in self.named_parameters():
            if name.endswith('c_proj.weight'):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        print(f"Initialized model: {self.count_parameters():,} trainable parameters")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        input_ids: torch.Tensor,                  # (B, T) integer token indices
        targets:   Optional[torch.Tensor] = None, # (B, T) targets; -1 = ignore
        kv_caches: Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]] = None,
        return_caches: bool = False,
    ):
        """
        Forward pass.

        Returns (logits, loss).  When `return_caches=True`, returns
        (logits, loss, list_of_kv_caches) for use by KV-cached generation.

        `kv_caches` is a list of length n_layer (one (K, V) tuple per block,
        or None if that block is the first call).  When provided, each block
        appends its new K/V to its cache and returns the updated cache.

        Training format reminder:
          "Question: What is dropout?\nAnswer: Dropout randomly zeroes..."
        We compute loss only over the Answer tokens (targets for Question
        positions are set to -1 so they're ignored).  See dataset.py.
        """
        B, T = input_ids.shape
        # When KV-caching, T may be 1 even though the real position is large.
        # The model will still encode positions correctly via RoPE's start_pos.
        max_pos = T + (kv_caches[0][0].size(2) if kv_caches and kv_caches[0] is not None else 0)
        assert max_pos <= self.config.block_size, (
            f"Seq len {max_pos} > block_size {self.config.block_size}"
        )

        x = self.transformer['drop'](self.transformer['wte'](input_ids))

        new_caches: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = []
        for i, block in enumerate(self.transformer['h']):
            past = kv_caches[i] if kv_caches is not None else None
            x, new_cache = block(x, kv_cache=past)
            new_caches.append(new_cache)

        x = self.transformer['ln_f'](x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            # Autoregressive shift: position i's logits predict the token at
            # position i+1.  Without this shift the model is asked to predict
            # the current token, which is trivially solvable through the
            # residual stream + causal self-attention (it can attend to
            # itself).  That collapses train loss to near-zero in a few hundred
            # steps while teaching the model nothing about generation.
            shift_logits  = logits[:, :-1, :].contiguous()
            shift_targets = targets[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_targets.view(-1),
                ignore_index=-1,
                label_smoothing=getattr(self.config, "label_smoothing", 0.0),
            )

        if return_caches:
            return logits, loss, new_caches
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids:      torch.Tensor,
        max_new_tokens: int   = 200,
        temperature:    float = 0.8,
        top_k:          int   = 50,
        top_p:          float = 0.95,
        stop_token:     Optional[int] = None,
        use_cache:      bool  = True,
    ) -> torch.Tensor:
        """
        Autoregressive generation: generate up to max_new_tokens new tokens.

        Temperature:  divides logits before softmax.
                      Low (0.3) = deterministic/safe, High (1.5) = creative/risky.
        Top-k:        only consider the top-k most likely tokens.
        Top-p:        nucleus sampling — keep the smallest set of tokens
                      whose cumulative probability ≥ p.

        KV cache (Phase 5):
        With `use_cache=True` (default), the prompt is processed once with a
        normal forward pass and then each subsequent token re-uses cached K/V
        from previous attention computations.  This makes generation 10–60×
        faster — every per-token step shrinks from O(prompt_len + step) work
        to O(1) per layer.

        When the cache would exceed block_size, we restart from the latest
        block_size tokens (rare in practice for our use case where
        max_new_tokens=200 and prompts fit comfortably).
        """
        was_training = self.training
        self.eval()
        block_size = self.config.block_size

        # ── Initial forward pass over the prompt ─────────────────────────────
        ctx = input_ids[:, -block_size:]
        caches: Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]] = None
        if use_cache:
            logits, _, caches = self(ctx, return_caches=True)
        else:
            logits, _ = self(ctx)

        for step in range(max_new_tokens):
            # ── Sample next token from the last position's logits ────────────
            next_tok = self._sample_next(
                logits[:, -1, :], temperature=temperature, top_k=top_k, top_p=top_p,
            )
            input_ids = torch.cat([input_ids, next_tok], dim=1)

            if stop_token is not None and (next_tok == stop_token).all():
                break
            if step == max_new_tokens - 1:
                break

            # ── Advance one token, using the cache when possible ─────────────
            if use_cache and caches is not None:
                cache_len = caches[0][0].size(2)
                if cache_len + 1 <= block_size:
                    logits, _, caches = self(next_tok, kv_caches=caches,
                                             return_caches=True)
                else:
                    # Cache full → restart with a fresh window.  Slower but
                    # correct.  Practically rare for this project.
                    ctx = input_ids[:, -block_size:]
                    logits, _, caches = self(ctx, return_caches=True)
            else:
                ctx = input_ids[:, -block_size:]
                logits, _ = self(ctx)

        if was_training:
            self.train()
        return input_ids

    @staticmethod
    def _sample_next(
        logits:      torch.Tensor,
        temperature: float = 0.8,
        top_k:       int   = 50,
        top_p:       float = 0.95,
    ) -> torch.Tensor:
        """Apply temperature, top-k, top-p, then multinomial sampling."""
        logits = logits / temperature
        if top_k:
            k_val = min(top_k, logits.size(-1))
            threshold = logits.topk(k_val).values[:, -1, None]
            logits = logits.masked_fill(logits < threshold, float('-inf'))
        if top_p:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            probs = F.softmax(sorted_logits, dim=-1)
            cum_probs = torch.cumsum(probs, dim=-1)
            remove = cum_probs - probs > top_p
            sorted_logits[remove] = float('-inf')
            logits.scatter_(1, sorted_idx, sorted_logits)
        return torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)


# ─────────────────────────────────────────────────────────────────────────────
# Quick sanity check
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Small config for fast testing
    cfg = TransformerConfig(n_layer=4, n_head=4, n_embd=256, block_size=256)
    model = TeachingAssistantModel(cfg)

    x = torch.randint(0, cfg.vocab_size, (2, 32))
    logits, loss = model(x, x)
    print(f"Logits: {logits.shape}")   # (2, 32, 50261)
    print(f"Loss:   {loss.item():.3f}")  # should be ~10.82 (= ln(50261))

    prompt = torch.randint(0, cfg.vocab_size, (1, 8))
    out = model.generate(prompt, max_new_tokens=5)
    print(f"Generated: {out.shape}")  # (1, 13)
