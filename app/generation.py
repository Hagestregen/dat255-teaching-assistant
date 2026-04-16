"""
generation.py  —  Shared generation utilities
==============================================
Single place for answer generation logic so all tabs (Ask, Quiz, Feedback,
Flashcards) reuse the same code regardless of whether we have a custom
checkpoint or a pretrained HuggingFace pipeline.

Public API
----------
answer_question(question, use_rag, temperature, pipeline, pretrained_pipe, retriever)
    → {"answer": str, "chunks": list}

get_random_topic_from_retriever(retriever)
    → str  (a topic derived from an actual index chunk, for quiz diversity)
"""

from __future__ import annotations
import random
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Seed queries used to pull diverse topics from the RAG index.
# We shoot a varied probe at the retriever and use whatever breadcrumb comes
# back as the generation topic — so questions actually reflect index content.
# ─────────────────────────────────────────────────────────────────────────────
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


def get_random_topic_from_retriever(retriever) -> str:
    """
    Pick a random seed, query the retriever, and return the breadcrumb of
    the top chunk as the quiz/flashcard topic.  This ensures questions are
    spread across the actual index content rather than always clustering
    around whichever chunks rank highest for "machine learning".
    """
    seed = random.choice(_PROBE_SEEDS)
    try:
        chunks = retriever.query(seed, top_k=1)
        if chunks and chunks[0].get("breadcrumb"):
            # breadcrumb is usually "lecture_X > section_title" — take the
            # last component so the topic is concrete and specific.
            breadcrumb = chunks[0]["breadcrumb"]
            topic = breadcrumb.split(">")[-1].strip()
            if topic:
                return topic
    except Exception:
        pass
    return seed  # fall back to the seed itself


# ─────────────────────────────────────────────────────────────────────────────
# Answer generation
# ─────────────────────────────────────────────────────────────────────────────

def _build_chat_messages(question: str, context: str = "") -> list[dict]:
    """Build a chat-format message list for instruction-tuned models."""
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
    pipeline=None,        # custom RAGPipeline
    pretrained_pipe=None, # HuggingFace text-generation pipeline
    retriever=None,       # standalone Retriever (pretrained mode)
) -> dict:
    """
    Generate an answer using whichever backend is available.

    Returns
    -------
    {"answer": str, "chunks": list[dict]}
        chunks is empty when RAG is disabled or unavailable.
    """
    if not question.strip():
        return {"answer": "Please enter a question.", "chunks": []}

    # ── Custom checkpoint path ─────────────────────────────────────────────
    if pipeline is not None:
        if use_rag:
            result = pipeline.answer(question, temperature=temperature, max_tokens=300)
            return {"answer": result["answer"], "chunks": result.get("chunks", [])}
        else:
            from rag_pipeline import answer_question as _aq
            answer = _aq(question, pipeline.model, pipeline.tokenizer,
                         pipeline.device, temperature=temperature)
            return {"answer": answer, "chunks": []}

    # ── Pretrained HuggingFace pipeline path ──────────────────────────────
    if pretrained_pipe is not None:
        active_retriever = retriever
        chunks: list[dict] = []

        if use_rag and active_retriever is not None:
            try:
                chunks = active_retriever.query(question, top_k=3)
                context = "\n---\n".join(c["text"][:300] for c in chunks[:3])
            except Exception:
                context = ""
        else:
            context = ""

        messages = _build_chat_messages(question, context if use_rag else "")

        try:
            # Instruction-tuned models accept a messages list directly
            result = pretrained_pipe(
                messages,
                max_new_tokens=300,
                do_sample=True,
                temperature=temperature,
                top_p=0.95,
                return_full_text=False,
            )
            # The pipeline returns the assistant's message content
            raw = result[0]["generated_text"]
            if isinstance(raw, list):
                # Some pipelines return the full conversation as a list
                answer = raw[-1].get("content", "")
            else:
                answer = str(raw)
        except Exception as e:
            answer = f"Generation error: {e}"

        return {"answer": answer.strip(), "chunks": chunks}

    return {"answer": "No model loaded.", "chunks": []}