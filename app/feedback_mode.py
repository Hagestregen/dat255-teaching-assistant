"""
feedback_mode.py  —  Long-answer review and scoring
=====================================================
Flow:
  1. The assistant generates an exam-style question from your course material.
  2. You write a free-form answer.
  3. The assistant reviews your answer: score (1-5) + specific feedback.

This is NOT agentic — it's two sequential forward passes with carefully-shaped
prompts.  The "intelligence" is in the prompt templates.

HOW THE REVIEW PROMPT WORKS:
──────────────────────────────
The model was optionally fine-tuned on answer-review examples from
data_generation.py, so it has seen:

  Context: {lecture text}
  Question: {question}
  Student answer: {partial/wrong answer}
  Review: Score: 3/5. You correctly identified X but missed Y...

At inference we send the same format and the model completes "Review:".

IF YOUR MODEL IS TOO SMALL TO DO THIS WELL:
────────────────────────────────────────────
Small models (4-8 layers, 256-512 dim) sometimes generate incoherent feedback.
You can fall back to GPT-4o-mini for the scoring step and keep your model for
quiz/explanation generation.  That's a valid design choice — note it in the
report as a hybrid architecture.
"""

import re
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: Question generation
# ─────────────────────────────────────────────────────────────────────────────

def _build_question_gen_messages(context: str, topic: str) -> list:
    """Chat messages for instruction-tuned models to generate an exam question."""
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
    return [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]


_QUESTION_GEN_PROMPT = """You are an examiner for a machine learning course.
Read the lecture material below and write ONE open-ended exam question.
The question should require a 3-5 sentence answer and be answerable from the text.

Lecture material:
{context}

Output ONLY the question, nothing else."""


def generate_question_for_feedback(
    retriever,
    topic:          str   = "",
    pretrained_pipe       = None,
    model                 = None,
    tokenizer             = None,
    device:         str   = "cpu",
    temperature:    float = 0.7,
) -> Optional[str]:
    """
    Generate an exam-style question from course material.

    Retrieves a relevant chunk (using topic or a random probe), then prompts
    the model to produce one open-ended question.  Returns the question string,
    or None on failure.
    """
    import random

    # Pick a retrieval query
    if topic.strip():
        query = topic.strip()
    else:
        probes = [
            "neural network training", "loss function optimisation",
            "regularisation dropout", "attention transformer architecture",
            "convolutional layers", "batch normalisation", "gradient descent",
            "evaluation metrics", "recurrent networks", "activation functions",
        ]
        query = random.choice(probes)

    # Retrieve context
    try:
        chunks  = retriever.query(query, top_k=2)
        context = "\n---\n".join(c["text"][:400] for c in chunks[:2])
        topic_label = chunks[0].get("breadcrumb", query).split(">")[-1].strip() if chunks else query
    except Exception:
        return None

    if not context.strip():
        return None

    # ── Pretrained (instruction-tuned) model ──────────────────────────────────
    if pretrained_pipe is not None:
        messages = _build_question_gen_messages(context, topic_label)
        try:
            result = pretrained_pipe(
                messages,
                max_new_tokens=80,
                do_sample=True,
                temperature=temperature,
                top_p=0.9,
                return_full_text=False,
            )
            raw = result[0]["generated_text"]
            if isinstance(raw, list):
                raw = raw[-1].get("content", "")
            question = str(raw).strip()
            # Strip any leading labels like "Question: " that some models add
            question = re.sub(r'^(?:question|q)[:\s]+', '', question, flags=re.IGNORECASE)
            return question if len(question) > 15 else None
        except Exception as e:
            print(f"  [feedback] question gen error: {e}")
            return None

    # ── Custom checkpoint ─────────────────────────────────────────────────────
    if model is not None and tokenizer is not None:
        import torch
        prompt    = _QUESTION_GEN_PROMPT.format(context=context)
        input_ids = tokenizer.encode(prompt)
        x         = torch.tensor([input_ids], dtype=torch.long).to(device)
        out       = model.generate(
            x, max_new_tokens=80, temperature=temperature,
            top_k=50, top_p=0.9, stop_token=tokenizer.eot_id,
        )
        generated = tokenizer.decode(out[0, len(input_ids):].tolist())
        question  = generated.replace("<|endoftext|>", "").strip().split("\n")[0]
        question  = re.sub(r'^(?:question|q)[:\s]+', '', question, flags=re.IGNORECASE)
        return question if len(question) > 15 else None

    return None


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: Review prompt construction
# ─────────────────────────────────────────────────────────────────────────────

def build_review_prompt(
    question:         str,
    student_answer:   str,
    context:          str = "",
    reference_answer: str = "",
) -> str:
    """
    Build the prompt for the answer review step.

    Includes retrieved context so the model can fact-check the student's
    answer against the course material.
    """
    parts = []
    if context:
        parts.append(f"Context: {context[:400]}")
    if reference_answer:
        parts.append(f"Reference answer: {reference_answer}")
    parts.append(f"Question: {question}")
    parts.append(f"Student answer: {student_answer}")
    parts.append("Review:")
    return "\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: Review with custom checkpoint
# ─────────────────────────────────────────────────────────────────────────────

def review_answer_with_model(
    question:       str,
    student_answer: str,
    rag_pipeline,
    max_tokens:     int   = 150,
    temperature:    float = 0.4,   # lower = more consistent scoring
) -> dict:
    """
    Use your trained transformer + RAG to review a student's long answer.

    Returns: {"score": int (1-5), "feedback": str, "raw": str}
    """
    import torch

    chunks  = rag_pipeline.retrieve(question)
    context = "\n---\n".join(c["text"][:200] for c in chunks[:2])

    prompt    = build_review_prompt(question, student_answer, context)
    input_ids = rag_pipeline.tokenizer.encode(prompt)
    x         = torch.tensor([input_ids], dtype=torch.long).to(rag_pipeline.device)

    out = rag_pipeline.model.generate(
        x,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_k=40,
        top_p=0.9,
        stop_token=rag_pipeline.tokenizer.eot_id,
    )

    raw = rag_pipeline.tokenizer.decode(out[0, len(input_ids):].tolist())
    raw = raw.replace("<|endoftext|>", "").strip()

    return _parse_review(raw)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: Shared parsing and formatting
# ─────────────────────────────────────────────────────────────────────────────

def _parse_review(raw: str) -> dict:
    """Parse 'Score: X/5. Feedback text' from model output."""
    score    = 0
    feedback = raw
    m = re.search(r'[Ss]core[:\s]+(\d)[/\s]*5', raw)
    if m:
        score    = int(m.group(1))
        feedback = raw[m.end():].strip().lstrip('.').strip()
    elif raw and raw[0].isdigit():
        score    = int(raw[0])
        feedback = raw[1:].strip().lstrip('/5').lstrip('.').strip()
    return {"score": score, "feedback": feedback, "raw": raw}


def format_review_for_display(review: dict, question: str = "", student_answer: str = "") -> str:
    """Format a review result as Markdown for display in Gradio."""
    score    = review.get("score", 0)
    feedback = review.get("feedback", "No feedback generated.")
    stars    = "★" * score + "☆" * (5 - score)
    return f"**Score: {score}/5** {stars}\n\n{feedback}"