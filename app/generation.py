# generation.py
"""
Shared generation utilities used by all tabs.

Public API:
    answer_question(question, use_rag, temperature, *, pipeline, pretrained_pipe, retriever)
        -> {"answer": str, "chunks": list}

    get_random_topic_from_retriever(retriever)
        -> str
"""

from __future__ import annotations
import random
from typing import Optional
import re
from rag.retriever import Retriever

_PROBE_SEEDS = [
    "neural network architecture",
    "loss function training",
    "regularization techniques",
    "attention transformer",
    "convolutional layers",
    "recurrent networks",
    "optimisation learning rate",
    "batch normalisation",
    "activation functions",
    "dropout training",
    "gradient vanishing exploding",
    "data augmentation",
    "embedding representation",
    "transfer learning fine-tuning",
    "evaluation metrics classification",
    "reinforcement learning reward",
    "generative adversarial networks",
    "variational autoencoder latent",
    "hyperparameter tuning",
    "cross-entropy softmax",
]


def get_random_topic_from_retriever(retriever: Retriever) -> str:
    """Pick a random probe topic, query the retriever, and return the breadcrumb leaf."""
    seed = random.choice(_PROBE_SEEDS)
    try:
        chunks = retriever.query(seed, top_k=1)
        if chunks and chunks[0].get("breadcrumb"):
            topic = chunks[0]["breadcrumb"].split(">")[-1].strip()
            if topic:
                return topic
    except Exception:
        pass
    return seed


def get_random_chunk_in_scope(retriever: Retriever, breadcrumb_prefix: str | None) -> dict | None:
    """
    Pick a random chunk whose breadcrumb starts with the given prefix.
    If prefix is None or empty, picks from all chunks.
    """
    pool = retriever.chunks  # assumes retriever exposes its chunk list
    # print(f"  [get_random_chunk_in_scope] pool: {pool!r}")
    if breadcrumb_prefix:
        pool = [
            c for c in pool
            if c.get("metadata", {}).get("breadcrumb", "").startswith(breadcrumb_prefix)
        ]
    return random.choice(pool) if pool else None

def _build_chat_messages(question: str, context: str = "") -> list[dict]:
    system = (
        "You are a helpful teaching assistant for a university machine learning course. "
        "Answer the student's question clearly and concisely. "
        "If context is provided, base your answer on it; otherwise use your knowledge."
    )
    user_content = question
    if context:
        user_content = f"Context from the course material:\n{context}\n\nQuestion: {question}"
    return [
        {"role": "system", "content": system},
        {"role": "user",   "content": user_content},
    ]


def answer_question(
    question:       str,
    use_rag:        bool  = True,
    temperature:    float = 0.7,
    *,
    pipeline        = None,
    pretrained_pipe = None,
    retriever       = None,
) -> dict:
    """
    Generate an answer.

    Returns {"answer": str, "chunks": list[dict]}.
    chunks is empty when RAG is disabled or unavailable.
    """
    if not question.strip():
        return {"answer": "Please enter a question.", "chunks": []}

    if pipeline is not None:
        if use_rag:
            result = pipeline.answer(question, temperature=temperature, max_tokens=300)
            return {"answer": result["answer"], "chunks": result.get("chunks", [])}
        else:
            from rag_pipeline import answer_question as _aq
            answer = _aq(question, pipeline.model, pipeline.tokenizer,
                         pipeline.device, temperature=temperature)
            return {"answer": answer, "chunks": []}

    if pretrained_pipe is not None:
        chunks: list[dict] = []

        if use_rag and retriever is not None:
            try:
                chunks  = retriever.query(question, top_k=3)
                context = "\n---\n".join(c["text"][:300] for c in chunks[:3])
            except Exception:
                context = ""
        else:
            context = ""

        messages = _build_chat_messages(question, context if use_rag else "")

        try:
            result = pretrained_pipe(
                messages,
                max_new_tokens=300,
                do_sample=True,
                temperature=temperature,
                top_p=0.95,
                return_full_text=False,
            )
            raw = result[0]["generated_text"]
            answer = raw[-1].get("content", "") if isinstance(raw, list) else str(raw)
        except Exception as e:
            answer = f"Generation error: {e}"

        return {"answer": answer.strip(), "chunks": chunks}

    return {"answer": "No model loaded.", "chunks": []}

# def clean_context(raw: str) -> str:
#     """Remove retriever metadata headers like [Source > Chapter > Section]."""
#     # Drop lines that are purely the bracketed path header
#     lines = raw.splitlines()
#     cleaned = [
#         line for line in lines
#         if not re.match(r'^\s*\[.+>\s*.+\]\s*$', line)
#     ]
#     return "\n".join(cleaned).strip()
def build_context(chunks: list[dict]) -> str:
    """
    Build the context string for the model.
    Breadcrumb is shown as a labelled source line, clearly separated
    from the actual content so the model doesn't echo it into questions.
    """
    parts = []
    for c in chunks[:2]:
        breadcrumb = c.get("metadata", {}).get("breadcrumb", "")
        body       = c["text"]
        if breadcrumb:
            parts.append(f"Source: {breadcrumb}\n\n{body}")
        else:
            parts.append(body)
    return "\n\n---\n\n".join(parts)[:600]