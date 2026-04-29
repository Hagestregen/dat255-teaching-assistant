# feedback.py
"""
Long-answer question generation and answer review/scoring.

Flow:
  1. generate_question  — pick a course chunk, produce an exam-style question.
  2. Student writes a free-form answer in the UI.
  3. review_answer      — score 1-5 and give specific feedback.
"""

from __future__ import annotations
import re
from typing import Optional

from generation import build_context
from llm_utils import call_llm, get_context_chunks


# =============================================================================
# Question generation
# =============================================================================

# Raw-completion prompt for custom checkpoint models
_QUESTION_COMPLETION_PROMPT = """\
You are an examiner for a machine learning course.
Read the lecture material below and write ONE open-ended exam question.
The question should require a 3-5 sentence answer and be answerable from the text.

Lecture material:
{context}

Output ONLY the question, nothing else."""


def _question_chat_messages(topic: str, context: str) -> list:
    system = (
        "You are an examiner for a university machine learning course. "
        "Generate ONE clear, open-ended exam question that a student must answer "
        "in 3-5 sentences. The question must be directly answerable from the "
        "provided course material. Output only the question — no preamble, no numbering."
    )
    user = (
        f"Topic: {topic}\n\n"
        f"Course material:\n{context}\n\n"
        "Generate one exam question about this material."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def generate_question(
    retriever,
    topic:            str        = "",
    pipe                         = None,
    model                        = None,
    tokenizer                    = None,
    device:     str              = "cpu",
    temperature: float           = 0.7,
    prefetched_chunk: dict | None = None,
) -> Optional[str]:
    """
    Generate an exam-style open-ended question from course material.

    `topic` is used both as the retrieval query and as context for the prompt.
    `prefetched_chunk` pins the primary context chunk when already selected
    upstream (e.g. by the progress tracker or random scope picker).

    Returns the question string, or None on failure.
    """
    import random

    query = topic.strip() or random.choice([
        "neural network training", "loss function optimisation",
        "regularisation dropout", "attention transformer architecture",
        "convolutional layers", "batch normalisation", "gradient descent",
        "evaluation metrics", "recurrent networks", "activation functions",
    ])

    try:
        chunks      = get_context_chunks(retriever, query, prefetched_chunk)
        context     = build_context(chunks)
        topic_label = (
            chunks[0].get("breadcrumb", query).split(">")[-1].strip()
            if chunks else query
        )
    except Exception as e:
        print(f"[feedback] context retrieval error: {e}")
        return None

    if not context.strip():
        return None

    messages = _question_chat_messages(topic_label, context)
    prompt   = _QUESTION_COMPLETION_PROMPT.format(context=context)

    generated = call_llm(
        pipe=pipe, model=model, tokenizer=tokenizer, device=device,
        messages=messages, prompt=prompt,
        max_new_tokens=80, temperature=temperature, top_p=0.9,
    )
    if not generated:
        print("[feedback] question generation returned empty text")
        return None

    # Strip any "Question:" prefix the model might add
    question = re.sub(r'^(?:question|q)[:\s]+', '', generated, flags=re.IGNORECASE).strip()
    return question if len(question) > 15 else None


# =============================================================================
# Answer review
# =============================================================================

# Raw-completion prompt for custom checkpoint models
_REVIEW_COMPLETION_PROMPT = """\
Context: {context}
Question: {question}
Student answer: {student_answer}
Review:"""


def _review_chat_messages(question: str, student_answer: str, context: str) -> list:
    system = (
        "You are a strict but fair teaching assistant for a machine learning course. "
        "Review the student answer. Score it 1-5 and give one specific piece of "
        "feedback: what they got right and what they missed. "
        "Format: Score: X/5. <feedback>"
    )
    parts = []
    if context:
        parts.append(f"Context from course material:\n{context}")
    parts += [f"Question: {question}", f"Student answer: {student_answer}", "Review:"]
    return [
        {"role": "system", "content": system},
        {"role": "user",   "content": "\n\n".join(parts)},
    ]


def review_answer(
    question:       str,
    student_answer: str,
    retriever,
    pipe      = None,
    model     = None,
    tokenizer = None,
    device:   str   = "cpu",
    temperature: float = 0.4,
) -> str:
    """
    Review a student's free-text answer and return a formatted markdown string.

    Retrieves context chunks related to the question, then calls the LLM to
    produce a score (1-5) and targeted feedback.  The formatted result is
    suitable for direct display in a gr.Markdown component.
    """
    context = ""
    if retriever:
        try:
            chunks  = retriever.query(question, top_k=2)
            context = "\n---\n".join(c["text"][:200] for c in chunks[:2])
        except Exception:
            pass

    messages = _review_chat_messages(question, student_answer, context)
    prompt   = _REVIEW_COMPLETION_PROMPT.format(
        context=context[:400] if context else "",
        question=question,
        student_answer=student_answer,
    )

    raw = call_llm(
        pipe=pipe, model=model, tokenizer=tokenizer, device=device,
        messages=messages, prompt=prompt,
        max_new_tokens=200, temperature=temperature, top_p=0.9,
    )
    if raw is None:
        return "Generation error — please try again."

    return _format_review(raw)


# =============================================================================
# Parsing helpers
# =============================================================================

def _parse_score(text: str) -> tuple[int, str]:
    """Return (score, feedback_text) extracted from a review string."""
    m = re.search(r'[Ss]core[:\s]+(\d)[/\s]*5', text)
    if m:
        return int(m.group(1)), text[m.end():].strip().lstrip('.').strip()
    if text and text[0].isdigit():
        return int(text[0]), text[1:].strip().lstrip('/5').lstrip('.').strip()
    return 0, text


def _format_review(raw: str) -> str:
    """Convert raw review text into a star-rated markdown string."""
    score, feedback = _parse_score(raw)
    stars = "★" * score + "☆" * (5 - score)
    return f"**Score: {score}/5** {stars}\n\n{feedback}"


def parse_score_from_markdown(formatted_md: str) -> int:
    """Extract the numeric score from a formatted review markdown string."""
    m = re.search(r'Score:\s*(\d)', formatted_md)
    return int(m.group(1)) if m else 0