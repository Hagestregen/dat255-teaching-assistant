"""
rag_pipeline.py  —  RAG-augmented Teaching Assistant
======================================================
This is where your trained transformer meets the RAG system you've already
built (chunker.py → embedder.py → retriever.py).

WHAT RAG DOES AND WHY YOUR SMALL MODEL NEEDS IT:
──────────────────────────────────────────────────
Your transformer has ~6-50M parameters. GPT-3 has 175B. The gap in
"memorized knowledge" is enormous. Your model will learn QA PATTERNS
from training, but it cannot memorize the DAT255 curriculum in detail.

RAG (Retrieval-Augmented Generation) solves this by:
  1. At inference time, encode the question as a vector.
  2. Search your HNSW index for the most similar lecture chunks.
  3. Prepend those chunks to the prompt as "Context: ..."
  4. Let the model generate an answer that can COPY from the context.

This is like open-book vs closed-book exam:
  - Model alone = closed-book: relies only on what it "memorized"
  - Model + RAG  = open-book: can look up relevant notes each time

The model doesn't need to memorize facts — it just needs to learn to
read context and extract/rephrase the relevant information. Even a
small model can do this well if fine-tuned properly.

A/B COMPARISON (your project's key experiment):
────────────────────────────────────────────────
  Model A: Just your trained transformer (no RAG)
  Model B: Your trained transformer + RAG
  Model C: GPT-4 (frontier baseline)

All three receive the same questions. You evaluate with the same metrics.
The comparison shows what RAG adds to a small model.
"""

import sys
from pathlib import Path
from typing import List, Optional

import torch

# Ensure project root is importable whether this file is run standalone or
# imported from app/gradio_app.py (which may already have set the path).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from model.transformer import TeachingAssistantModel, TransformerConfig
from model.dataset import Tokenizer, format_rag_prompt


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: Load trained model from checkpoint
# ─────────────────────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, device: str = None) -> tuple:
    """
    Load a trained model from a checkpoint file.

    Returns (model, tokenizer, config)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(checkpoint_path, map_location=device)
    from dataclasses import fields
    config = TransformerConfig(**{
        k: v for k, v in ckpt["model_config"].items()
        if k in [f.name for f in fields(TransformerConfig)]
    })
    model = TeachingAssistantModel(config)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    tokenizer = Tokenizer()
    print(f"Model loaded from {checkpoint_path} ({config.n_layer}L, {config.n_embd}d)")
    return model, tokenizer, config


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: Plain generation (no RAG) — "Model A"
# ─────────────────────────────────────────────────────────────────────────────

def answer_question(
    question:    str,
    model:       TeachingAssistantModel,
    tokenizer:   Tokenizer,
    device:      str      = "cpu",
    max_tokens:  int      = 200,
    temperature: float    = 0.7,
    top_k:       int      = 50,
    top_p:       float    = 0.95,
) -> str:
    """
    Generate an answer using the model alone, no retrieved context.

    The prompt is just: "Question: {q}\nAnswer:"
    The model must rely entirely on what it learned during training.
    """
    prompt    = f"Question: {question}\nAnswer:"
    input_ids = tokenizer.encode(prompt)
    x         = torch.tensor([input_ids], dtype=torch.long).to(device)

    out = model.generate(
        x,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        stop_token=tokenizer.eot_id,
    )

    # Decode only the newly generated tokens (after the prompt)
    generated_ids = out[0, len(input_ids):].tolist()
    answer = tokenizer.decode(generated_ids)

    # Clean up: remove stop token text and trailing whitespace
    answer = answer.replace("<|endoftext|>", "").strip()
    return answer


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: RAG-augmented generation — "Model B"
# ─────────────────────────────────────────────────────────────────────────────

class RAGPipeline:
    """
    Full RAG pipeline: query → retrieve → generate.

    HOW THE CONTEXT FITS IN THE PROMPT:
    ─────────────────────────────────────
    The model has a fixed context window (e.g. 512 tokens). We need to fit:
      - Retrieved chunks (context)
      - Question
      - Generated answer

    We budget roughly:
      - 300 tokens for context (from retrieved chunks)
      - 50 tokens for the question
      - 150 tokens for the generated answer
      - 12 tokens for formatting overhead

    If retrieved chunks are longer, we truncate them. We always prioritize
    the highest-scoring chunk (most relevant).

    HOW RoPE HANDLES THE PREPENDED CONTEXT:
    ─────────────────────────────────────────
    With RoPE, the combined sequence [chunk1][chunk2][Question:...][Answer:...]
    is just one sequence with continuous positions 0, 1, 2, ...

    The model sees:
      position 0-N:   context tokens
      position N+1-M: question tokens
      position M+1-:  the answer it generates

    This works because RoPE encodes RELATIVE positions. The model has seen
    similar patterns during training (question + answer), and it generalizes
    to having context before the question.

    For even better RAG performance: include context-grounded QA examples
    in training (the "context" field from synthetic_qa.jsonl). Then the model
    explicitly learns to use the provided context.
    """

    def __init__(
        self,
        checkpoint_path: str,
        index_dir:       str  = "rag_index",
        device:          str  = None,
        top_k:           int  = 3,
        context_budget:  int  = 300,  # max tokens for retrieved context
    ):
        self.device         = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.top_k          = top_k
        self.context_budget = context_budget

        # Load our trained model
        self.model, self.tokenizer, self.model_config = load_model(
            checkpoint_path, self.device
        )

        # Load the RAG retriever (your existing retriever.py)
        # sys.path.insert(0, str(Path(__file__).parent))
        from rag.retriever import Retriever
        self.retriever = Retriever(index_dir=index_dir)

        print(f"RAG pipeline ready. top_k={top_k}, context_budget={context_budget} tokens")

    def retrieve(self, question: str) -> List[dict]:
        """
        Retrieve top-k most relevant chunks for a question.

        Returns list of dicts with keys: text, score, source, breadcrumb
        Sorted by relevance score (highest first).
        """
        return self.retriever.query(question, top_k=self.top_k)

    def _build_context_string(self, chunks: List[dict]) -> str:
        """
        Assemble retrieved chunks into a context string.

        We truncate based on token count to stay within the context budget.
        Higher-scoring chunks come first so they're least likely to be truncated.
        """
        context_parts = []
        tokens_used   = 0

        for chunk in chunks:
            text = chunk["text"]
            # Quick estimate: 1 token ≈ 4 characters (rough GPT-2 average)
            token_estimate = len(text) // 4
            if tokens_used + token_estimate > self.context_budget:
                # Truncate this chunk to fit
                chars_left = (self.context_budget - tokens_used) * 4
                if chars_left > 100:
                    text = text[:chars_left] + "..."
                    context_parts.append(text)
                break
            context_parts.append(text)
            tokens_used += token_estimate

        return "\n---\n".join(context_parts)

    def answer(
        self,
        question:    str,
        temperature: float = 0.7,
        top_k:       int   = 50,
        top_p:       float = 0.95,
        max_tokens:  int   = 200,
        verbose:     bool  = False,
    ) -> dict:
        """
        Full RAG pipeline: retrieve → assemble prompt → generate.

        Returns a dict with:
          answer:   the generated text
          chunks:   the retrieved chunks used
          prompt:   the full prompt sent to the model (for debugging)
        """
        # Step 1: Retrieve
        chunks = self.retrieve(question)

        if verbose:
            print(f"\n[RAG] Retrieved {len(chunks)} chunks:")
            for c in chunks:
                print(f"  score={c['score']:.3f} | {c['breadcrumb']} | {c['source']}")

        # Step 2: Assemble context + prompt
        context = self._build_context_string(chunks)
        prompt  = format_rag_prompt(context, question)

        if verbose:
            print(f"\n[RAG] Prompt preview:\n{prompt[:300]}...")

        # Step 3: Generate
        input_ids = self.tokenizer.encode(prompt)

        # Safety check: if prompt is already too long, truncate the context
        max_prompt_len = self.model_config.block_size - max_tokens - 10
        if len(input_ids) > max_prompt_len:
            # Re-build with shorter context
            context = context[:max_prompt_len * 3]  # rough char estimate
            prompt  = format_rag_prompt(context, question)
            input_ids = self.tokenizer.encode(prompt)[:max_prompt_len]

        x = torch.tensor([input_ids], dtype=torch.long).to(self.device)
        out = self.model.generate(
            x,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            stop_token=self.tokenizer.eot_id,
        )

        generated_ids = out[0, len(input_ids):].tolist()
        answer = self.tokenizer.decode(generated_ids)
        answer = answer.replace("<|endoftext|>", "").strip()

        return {
            "answer": answer,
            "chunks": chunks,
            "prompt": prompt,
            "n_prompt_tokens": len(input_ids),
        }

    def compare_ab(self, question: str, **kwargs) -> dict:
        """
        Run both Model A (no RAG) and Model B (with RAG) on the same question.
        This is your key experiment for the report.
        """
        answer_a = answer_question(
            question, self.model, self.tokenizer, self.device, **kwargs
        )
        result_b = self.answer(question, **kwargs)

        return {
            "question":  question,
            "answer_a":  answer_a,   # without RAG
            "answer_b":  result_b["answer"],  # with RAG
            "chunks_used": result_b["chunks"],
        }


# ─────────────────────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Replace with your actual checkpoint path
    CHECKPOINT = "checkpoints/step_005000.pt"
    INDEX_DIR  = "rag_index"

    if not Path(CHECKPOINT).exists():
        print("No checkpoint found. Run train.py first.")
        exit()

    rag = RAGPipeline(CHECKPOINT, INDEX_DIR, top_k=3)

    question = "What is dropout and why is it used in neural networks?"

    print("=" * 60)
    print(f"Question: {question}")
    print("=" * 60)

    result = rag.compare_ab(question, verbose=True)
    print(f"\n[Model A — no RAG]:\n{result['answer_a']}")
    print(f"\n[Model B — with RAG]:\n{result['answer_b']}")
