# llm_utils.py
"""
Shared helpers for LLM generation and RAG context assembly.

Imported by quiz.py, feedback.py, and flashcard.py so each module
only contains its own prompts and data structures.
"""

from __future__ import annotations
from typing import Optional


def call_llm(
    *,
    pipe=None,
    model=None,
    tokenizer=None,
    device: str       = "cpu",
    messages: list    | None = None,   # used by HuggingFace pipeline (chat format)
    prompt:   str     | None = None,   # used by custom checkpoint (raw completion)
    max_new_tokens: int   = 200,
    temperature:    float = 0.7,
    top_p:          float = 0.95,
) -> Optional[str]:
    """
    Call whichever backend is available and return the generated text.

    Priority: pipe (HuggingFace instruct model) > model+tokenizer (custom checkpoint).
    Returns None on failure so callers can retry or surface an error.
    """
    if pipe is not None and messages is not None:
        try:
            result = pipe(
                messages,
                max_new_tokens  = max_new_tokens,
                do_sample       = True,
                temperature     = temperature,
                top_p           = top_p,
                return_full_text= False,
            )
            raw = result[0]["generated_text"]
            return (raw[-1].get("content", "") if isinstance(raw, list) else str(raw)).strip()
        except Exception as e:
            print(f"[llm_utils] pipeline error: {e}")
            return None

    if model is not None and tokenizer is not None and prompt is not None:
        import torch
        input_ids = tokenizer.encode(prompt)
        x = torch.tensor([input_ids], dtype=torch.long).to(device)
        try:
            out = model.generate(
                x,
                max_new_tokens = max_new_tokens,
                temperature    = temperature,
                top_k          = 50,
                top_p          = top_p,
                stop_token     = tokenizer.eot_id,
            )
            generated = tokenizer.decode(out[0, len(input_ids):].tolist())
            # return generated.replace("", "").strip()
            return generated.replace("<|endoftext|>", "").strip()
        except Exception as e:
            print(f"[llm_utils] model error: {e}")
            return None

    return None


def get_context_chunks(
    retriever,
    topic:            str,
    prefetched_chunk: dict | None = None,
    top_k:            int         = 2,
) -> list[dict]:
    """
    Return up to `top_k` chunks for the given topic.

    If a `prefetched_chunk` is supplied (e.g. chosen by the progress tracker
    or a random scope picker), it becomes the primary chunk and one additional
    related chunk is fetched to give the model more context.
    """
    if prefetched_chunk is not None:
        extra  = retriever.query(topic, top_k=1)
        extras = [c for c in extra if c != prefetched_chunk][:1]
        return [prefetched_chunk] + extras
    return retriever.query(topic, top_k=top_k)