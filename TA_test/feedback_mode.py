"""
feedback_mode.py  —  Long-answer review and scoring
=====================================================
When the student writes a long answer, the model:
  1. Retrieves relevant context for the question (RAG)
  2. Reads the student's answer
  3. Generates a score (1-5) and specific feedback

This is NOT agentic — it's a single forward pass with a carefully-shaped prompt.
The "intelligence" is in the prompt template, not a planning loop.

HOW THE PROMPT WORKS:
──────────────────────
The model was (optionally) fine-tuned on answer review examples from
data_generation.py. So it has seen:

  Context: {lecture text}
  Question: {question}
  Student answer: {partial/wrong answer}
  Review: Score: 3/5. You correctly identified X but missed Y...

At inference we send the same format and the model completes the "Review:" part.

IF YOUR MODEL IS TOO SMALL TO DO THIS WELL:
────────────────────────────────────────────
Small models (4-8 layers, 256-512 dim) sometimes generate incoherent feedback.
In that case, use GPT-4o-mini for the feedback/scoring step and reserve your
model for quiz generation and explanation. This is explicitly noted as a
valid design choice — you can say in the report that the review function uses
a stronger model as an "oracle grader" while your model handles generation.

This is actually common in practice (e.g., AI tutoring systems use different
specialized models for different tasks).
"""

from typing import Optional


def build_review_prompt(
    question: str,
    student_answer: str,
    context: str = "",
    reference_answer: str = "",
) -> str:
    """
    Build the prompt for the answer review mode.

    We include:
    - Retrieved context (so the model can check facts)
    - The original question
    - The student's answer
    - Optionally, a reference answer (if available from the test set)

    The model should output: "Score: X/5. [specific feedback]"
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


def review_answer_with_model(
    question:        str,
    student_answer:  str,
    rag_pipeline,
    max_tokens:      int   = 150,
    temperature:     float = 0.4,   # lower temp = more consistent scoring
) -> dict:
    """
    Use your trained transformer + RAG to review a student's long answer.

    Returns: {"score": int (1-5), "feedback": str, "raw": str}
    """
    import torch, re

    # Retrieve context for the question
    chunks = rag_pipeline.retrieve(question)
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

    # Parse "Score: X/5. Feedback..." pattern
    score    = 0
    feedback = raw
    m = re.search(r'[Ss]core[:\s]+(\d)[/\s]*5', raw)
    if m:
        score    = int(m.group(1))
        feedback = raw[m.end():].strip().lstrip('.').strip()
    elif raw[0].isdigit():
        score = int(raw[0])
        feedback = raw[1:].strip().lstrip('/5').lstrip('.').strip()

    return {"score": score, "feedback": feedback, "raw": raw}


def review_answer_with_gpt(
    question:        str,
    student_answer:  str,
    api_key:         str,
    context:         str = "",
    reference:       str = "",
    model:           str = "gpt-4o-mini",
) -> dict:
    """
    Use GPT-4 to review the answer. More reliable than the small model for this task.
    Use this as your primary reviewer and your trained model as fallback/comparison.
    """
    import openai, json
    client = openai.OpenAI(api_key=api_key)

    system = ("You are a strict but fair ML course teaching assistant. "
              "Review student answers concisely. Score 1-5 and give one specific "
              "piece of feedback about what they got right and what they missed.")

    prompt = build_review_prompt(question, student_answer, context, reference)
    prompt += "\n\nRespond with JSON: {\"score\": <1-5>, \"feedback\": \"...\"}"

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.3,
            max_tokens=200,
            response_format={"type": "json_object"},
        )
        data = json.loads(resp.choices[0].message.content)
        return {
            "score":    data.get("score", 0),
            "feedback": data.get("feedback", ""),
            "raw":      resp.choices[0].message.content,
        }
    except Exception as e:
        return {"score": 0, "feedback": f"Review failed: {e}", "raw": ""}


def format_review_for_display(review: dict, question: str, student_answer: str) -> str:
    """Format a review result for display in the terminal or Gradio."""
    score    = review.get("score", 0)
    feedback = review.get("feedback", "No feedback generated.")

    stars  = "★" * score + "☆" * (5 - score)
    return (
        f"**Score: {score}/5** {stars}\n\n"
        f"{feedback}"
    )
