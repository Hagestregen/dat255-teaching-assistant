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
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


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
    vocab_size: int   = 50_257   # GPT-2 tokenizer (tiktoken 'gpt2' encoding)
    n_layer:    int   = 6        # number of transformer blocks stacked
    n_head:     int   = 8        # number of attention heads per block
    n_embd:     int   = 512      # embedding dimension (must be divisible by n_head)
    block_size: int   = 512      # maximum context window in tokens
    dropout:    float = 0.1      # dropout probability (set 0.0 at inference)
    bias:       bool  = False    # False = slightly faster, usually fine

    def __post_init__(self):
        assert self.n_embd % self.n_head == 0, (
            f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})"
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

    def forward(self, q: torch.Tensor, k: torch.Tensor):
        """
        Apply rotary embeddings to query and key tensors.

        q, k: (batch, n_head, seq_len, head_dim)
        Returns rotated (q_rot, k_rot) with the same shape.

        Note: we do NOT rotate V (values). Values carry content, not position.
        """
        T = q.size(2)
        cos = self.cos_table[:T].unsqueeze(0).unsqueeze(0)  # (1, 1, T, head_dim)
        sin = self.sin_table[:T].unsqueeze(0).unsqueeze(0)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        # Project to Q, K, V  →  split along last dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        # (B, T, C) → (B, n_head, T, head_dim)
        def split_heads(t):
            return t.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)
        q, k = self.rope(q, k)  # inject positional info via rotation

        # Flash Attention: memory-efficient, causal mask applied automatically
        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )

        # (B, n_head, T, head_dim) → (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.c_proj(y))


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: Feed-Forward Network (FFN)
# ─────────────────────────────────────────────────────────────────────────────

class FeedForward(nn.Module):
    """
    Position-wise FFN: applied independently to each token after attention.

    Attention = "which tokens should I look at?"
    FFN = "now that I know, what do I do with that information?"

    Most of the model's factual knowledge is thought to be stored in the FFN
    weights (each neuron can be thought of as a fuzzy key-value store).

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


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: Transformer Block
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

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.ffn  = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))  # attention sub-layer
        x = x + self.ffn(self.ln_2(x))   # feedforward sub-layer
        return x


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: Full Model
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
            'h':    nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            'ln_f': nn.LayerNorm(config.n_embd),  # final layer norm
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
        input_ids: torch.Tensor,        # (B, T) integer token indices
        targets:   torch.Tensor = None  # (B, T) targets; -1 = ignore (padding)
    ):
        """
        Forward pass.

        Returns (logits, loss). Loss is None if targets not provided.

        The training format looks like:
          "Question: What is dropout?\nAnswer: Dropout randomly zeroes..."
        We compute loss only over the Answer tokens (targets for Question
        positions are set to -1 so they're ignored). See dataset.py.
        """
        B, T = input_ids.shape
        assert T <= self.config.block_size, f"Seq len {T} > block_size {self.config.block_size}"

        # Look up token embeddings — no positional embeddings added (RoPE handles it inside attention)
        x = self.transformer['drop'](self.transformer['wte'](input_ids))

        for block in self.transformer['h']:
            x = block(x)

        x = self.transformer['ln_f'](x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,  # ignore padding positions
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids:      torch.Tensor,
        max_new_tokens: int   = 200,
        temperature:    float = 0.8,
        top_k:          int   = 50,
        top_p:          float = 0.95,
        stop_token:     int   = None,
    ) -> torch.Tensor:
        """
        Autoregressive generation: generate up to max_new_tokens new tokens.

        Temperature:  divides logits before softmax.
                      Low (0.3) = deterministic/safe, High (1.5) = creative/risky.
        Top-k:        only consider the top-k most likely tokens.
        Top-p:        nucleus sampling — keep the smallest set of tokens
                      whose cumulative probability ≥ p. This adapts to the
                      model's confidence: if it's very sure, p=0.95 might only
                      keep 2-3 tokens; if unsure, it keeps more options open.
        """
        self.eval()
        for _ in range(max_new_tokens):
            # Crop to context window
            ctx = input_ids[:, -self.config.block_size:]
            logits, _ = self(ctx)
            logits = logits[:, -1, :] / temperature  # (B, vocab_size)

            # Top-k: zero out everything outside top-k
            if top_k:
                k_val = min(top_k, logits.size(-1))
                threshold = logits.topk(k_val).values[:, -1, None]
                logits = logits.masked_fill(logits < threshold, float('-inf'))

            # Top-p (nucleus): keep tokens until cumulative prob >= top_p
            if top_p:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                # Remove tokens once cumulative prob exceeds top_p
                remove = cum_probs - F.softmax(sorted_logits, dim=-1) > top_p
                sorted_logits[remove] = float('-inf')
                logits.scatter_(1, sorted_idx, sorted_logits)

            next_tok = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            input_ids = torch.cat([input_ids, next_tok], dim=1)

            if stop_token is not None and (next_tok == stop_token).all():
                break

        return input_ids


# ─────────────────────────────────────────────────────────────────────────────
# Quick sanity check
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Small config for fast testing
    cfg = TransformerConfig(n_layer=4, n_head=4, n_embd=256, block_size=256)
    model = TeachingAssistantModel(cfg)

    x = torch.randint(0, cfg.vocab_size, (2, 32))
    logits, loss = model(x, x)
    print(f"Logits: {logits.shape}")   # (2, 32, 50257)
    print(f"Loss:   {loss.item():.3f}")  # should be ~10.82 (= ln(50257))

    prompt = torch.randint(0, cfg.vocab_size, (1, 8))
    out = model.generate(prompt, max_new_tokens=5)
    print(f"Generated: {out.shape}")  # (1, 13)
