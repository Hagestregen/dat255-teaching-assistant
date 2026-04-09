"""
data_generation.py  —  Synthetic training data covering all interaction modes
===============================================================================
Your model needs to have seen examples of all three output styles during training:

  1. EXPLANATION style:   "Explain X" → multi-paragraph walkthrough
  2. QUIZ GENERATION:     "Generate a quiz question about X" → JSON
  3. ANSWER REVIEW:       "Review this answer: ..." → score + feedback

You do NOT need to write any of these examples by hand. This script uses
GPT-4o-mini to generate them from your course chunks automatically.

WHY TRAINING ON EXPLANATIONS MATTERS:
──────────────────────────────────────
A model trained only on strict Q/A pairs learns a pattern:
  "Question: {short question}?" → "Answer: {one or two sentences}"

When asked "Can you walk me through how backpropagation works?", it tends
to produce a short, clipped response because that's all it has seen.

If you also train on:
  "Explain backpropagation in detail." → "Backpropagation is the algorithm
   used to compute gradients in a neural network. It works by applying the
   chain rule of calculus backwards through the computational graph..."

...the model learns that some prompts call for longer, more structured output.

HOWEVER: With RAG, even a poorly-trained model can explain things reasonably
well, because the retrieved context gives it the raw material to work with.
The training data mainly shapes the OUTPUT REGISTER (tone, length, structure),
not the factual content.

IN PRACTICE: The QA datasets already contain some explanation-style answers
(questions like "What is X?" with 3-5 sentence answers). You don't need a
huge amount of extra explanation training data — just enough to teach the
register. 200-400 explanation examples is usually sufficient.
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Prompt templates for each mode
# ─────────────────────────────────────────────────────────────────────────────

EXPLAIN_PROMPT = """You are creating training data for a machine learning teaching assistant.
Based on the following lecture material, create ONE explanation example.

Lecture material:
{text}

Generate a natural student request and a detailed explanation response.
The request should be conversational, like a student asking for help.

Respond ONLY with JSON:
{{
  "request": "Can you explain [specific concept from the text]?",
  "explanation": "A clear, 3-5 sentence explanation of the concept, written as if talking to a student."
}}"""


ANSWER_REVIEW_PROMPT = """You are creating training data for a machine learning teaching assistant.
Based on the following lecture material, create a question, a sample student answer, and a review.

Lecture material:
{text}

Create:
1. A question about a concept in the text
2. A partially correct student answer (missing 1-2 important details)
3. A review that scores it 1-5 and gives specific constructive feedback

Respond ONLY with JSON:
{{
  "question": "The question text?",
  "reference_answer": "The complete correct answer.",
  "student_answer": "A partially correct answer that misses something.",
  "review": {{
    "score": 3,
    "feedback": "You correctly identified X. However, you missed Y, which is important because Z."
  }}
}}"""


QUIZ_GENERATION_PROMPT = """You are creating training data for a machine learning teaching assistant.
Based on the following lecture material, create a quiz generation example.

The training example should show: given a topic + context, generate a JSON quiz question.

Lecture material:
{text}

Respond ONLY with JSON containing the full interaction:
{{
  "generate_request": "Generate a multiple-choice question about [concept from text].",
  "quiz_output": {{
    "question": "The question?",
    "options": ["Correct answer", "Plausible wrong answer 1", "Plausible wrong answer 2", "Plausible wrong answer 3"],
    "correct_index": 0,
    "explanation": "One sentence explaining why option 0 is correct."
  }}
}}"""


# ─────────────────────────────────────────────────────────────────────────────
# Training format strings for each mode
# ─────────────────────────────────────────────────────────────────────────────

def format_explanation_example(request: str, explanation: str, context: str = "") -> str:
    """
    Format an explanation training example.

    At inference, the model will see:
      Context: {retrieved_text}
      Request: Can you explain dropout?
      Response:
    And needs to produce a multi-sentence explanation.
    """
    if context:
        return f"Context: {context}\nRequest: {request}\nResponse: {explanation}<|endoftext|>"
    return f"Request: {request}\nResponse: {explanation}<|endoftext|>"


def format_review_example(question: str, student_answer: str, review_text: str, context: str = "") -> str:
    """
    Format a review/feedback training example.

    The model learns to act as a teacher reviewing a student's answer.
    At inference, you'll send:
      Context: {retrieved text about the topic}
      Question: {the question that was asked}
      Student answer: {what the user wrote}
      Review:
    """
    if context:
        return (f"Context: {context}\n"
                f"Question: {question}\n"
                f"Student answer: {student_answer}\n"
                f"Review: {review_text}<|endoftext|>")
    return (f"Question: {question}\n"
            f"Student answer: {student_answer}\n"
            f"Review: {review_text}<|endoftext|>")


def format_qa_example(question: str, answer: str, context: str = "") -> str:
    """Standard QA format (from dataset.py, repeated here for completeness)."""
    if context:
        return f"Context: {context}\nQuestion: {question}\nAnswer: {answer}<|endoftext|>"
    return f"Question: {question}\nAnswer: {answer}<|endoftext|>"


# ─────────────────────────────────────────────────────────────────────────────
# Generation functions
# ─────────────────────────────────────────────────────────────────────────────

def generate_all_modes(
    chunks_json_path: str,
    output_jsonl_path: str,
    api_key: str,
    model: str = "gpt-4o-mini",
    max_chunks: int = 150,
    # How many examples of each type to generate
    n_qa: int = 3,           # QA pairs per chunk
    n_explain: int = 1,      # explanation examples per chunk
    n_review: int = 1,       # answer review examples per chunk
    n_quiz_gen: int = 1,     # quiz generation examples per chunk
):
    """
    Generate training data for ALL interaction modes from your course chunks.

    Total output per chunk: up to n_qa + n_explain + n_review + n_quiz_gen examples.
    With max_chunks=150 and defaults: ~900 training examples total.

    This takes ~10-15 minutes and costs ~$1-2 with gpt-4o-mini.
    Run ONCE and commit the output file to your repo.
    """
    import openai
    client = openai.OpenAI(api_key=api_key)

    with open(chunks_json_path) as f:
        chunks = json.load(f)

    selected = random.sample(chunks, min(max_chunks, len(chunks)))
    results  = []

    def call_gpt(prompt: str) -> Optional[dict]:
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=800,
                response_format={"type": "json_object"},
            )
            return json.loads(resp.choices[0].message.content)
        except Exception as e:
            print(f"    GPT call failed: {e}")
            return None

    for i, chunk in enumerate(selected):
        text = chunk["text"][:700]  # keep within token budget
        breadcrumb = chunk.get("metadata", {}).get("breadcrumb", "")
        topic = breadcrumb or "machine learning"

        print(f"  [{i+1}/{len(selected)}] {topic[:50]}")

        # ── QA pairs ──────────────────────────────────────────────────────
        from dataset import generate_synthetic_qa  # reuse existing function
        # We'll just call the templates directly here for self-containment
        qa_prompt = f"""Generate {n_qa} question-answer pairs from this text.
Respond ONLY with a JSON array: [{{"question": "...", "answer": "..."}}]

Text: {text}"""
        qa_data = call_gpt(qa_prompt)
        if qa_data:
            pairs = qa_data if isinstance(qa_data, list) else qa_data.get("pairs", [qa_data])
            for pair in pairs[:n_qa]:
                if "question" in pair and "answer" in pair:
                    results.append({
                        "type":     "qa",
                        "text":     format_qa_example(pair["question"], pair["answer"], context=text[:300]),
                        "question": pair["question"],
                        "answer":   pair["answer"],
                        "source":   chunk.get("source", ""),
                        "topic":    topic,
                    })

        # ── Explanation ───────────────────────────────────────────────────
        explain_data = call_gpt(EXPLAIN_PROMPT.format(text=text))
        if explain_data and "request" in explain_data and "explanation" in explain_data:
            results.append({
                "type":        "explanation",
                "text":        format_explanation_example(
                    explain_data["request"],
                    explain_data["explanation"],
                    context=text[:300],
                ),
                "request":     explain_data["request"],
                "explanation": explain_data["explanation"],
                "source":      chunk.get("source", ""),
                "topic":       topic,
            })

        # ── Answer review ─────────────────────────────────────────────────
        review_data = call_gpt(ANSWER_REVIEW_PROMPT.format(text=text))
        if review_data and all(k in review_data for k in ["question", "student_answer", "review"]):
            review = review_data["review"]
            review_text = f"Score: {review.get('score', '?')}/5. {review.get('feedback', '')}"
            results.append({
                "type":           "review",
                "text":           format_review_example(
                    review_data["question"],
                    review_data["student_answer"],
                    review_text,
                    context=text[:300],
                ),
                "question":       review_data["question"],
                "student_answer": review_data["student_answer"],
                "review":         review_text,
                "reference":      review_data.get("reference_answer", ""),
                "source":         chunk.get("source", ""),
                "topic":          topic,
            })

        if (i + 1) % 20 == 0:
            print(f"  Saved {len(results)} examples so far...")

    # Save all examples
    Path(output_jsonl_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_jsonl_path, "w") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Print breakdown
    counts = {}
    for r in results:
        counts[r["type"]] = counts.get(r["type"], 0) + 1
    print(f"\nGenerated {len(results)} total examples:")
    for t, n in counts.items():
        print(f"  {t}: {n}")
    print(f"Saved to {output_jsonl_path}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Updated dataset.py integration
# ─────────────────────────────────────────────────────────────────────────────

def load_all_training_data(jsonl_path: str) -> List[Dict]:
    """
    Load all generated training examples.
    Each has a "text" field with the formatted training string.

    In dataset.py's QADataset, you can handle these by checking if the
    example has a "text" field (pre-formatted) vs "question"/"answer" fields.
    See the updated QADataset below.
    """
    results = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    print(f"Loaded {len(results)} training examples from {jsonl_path}")
    return results


class MultiModeDataset:
    """
    Drop-in replacement for QADataset that handles all training formats.

    The key difference: instead of building labels from a question/answer split,
    we detect the "Answer:", "Response:", or "Review:" prefix and mask everything
    before it. The model learns to complete any of these prompt types.
    """

    import re as _re

    # Markers that indicate where the model's output starts
    OUTPUT_MARKERS = [
        "Answer:",
        "Response:",
        "Review:",
    ]

    def __init__(self, examples: List[Dict], tokenizer, max_length: int = 512):
        import torch
        from torch.utils.data import Dataset

        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.data       = []
        skipped         = 0

        for ex in examples:
            item = self._process(ex)
            if item is not None:
                self.data.append(item)
            else:
                skipped += 1

        print(f"MultiModeDataset: {len(self.data)} examples, {skipped} skipped")

    def _find_output_start(self, full_ids: List[int], full_text: str) -> int:
        """
        Find the token index where the model's output starts.
        We tokenize just the prompt part to find its length.
        """
        for marker in self.OUTPUT_MARKERS:
            idx = full_text.find(marker)
            if idx != -1:
                # Tokenize everything up to and including the marker
                prefix = full_text[:idx + len(marker)]
                return len(self.tokenizer.encode(prefix))
        # Fallback: mask nothing (whole sequence is target)
        return 0

    def _process(self, example: Dict):
        import torch
        text = example.get("text", "")
        if not text:
            # Legacy format with question/answer fields
            q = example.get("question", "")
            a = example.get("answer", "")
            if not q or not a:
                return None
            from dataset import format_qa_example
            text = format_qa_example(q, a)

        full_ids = self.tokenizer.encode(text)
        if len(full_ids) > self.max_length:
            return None

        prompt_len = self._find_output_start(full_ids, text)

        labels     = [-1] * prompt_len + full_ids[prompt_len:]
        pad_len    = self.max_length - len(full_ids)
        input_ids  = full_ids + [self.tokenizer.pad_id] * pad_len
        labels     = labels   + [-1] * pad_len

        import torch
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels":    torch.tensor(labels,    dtype=torch.long),
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


if __name__ == "__main__":
    # Quick format test
    print(format_explanation_example(
        "Can you explain what dropout does?",
        "Dropout is a regularization technique that randomly zeroes a fraction of "
        "neuron outputs during each training step. This prevents neurons from "
        "co-adapting, which forces the network to learn more robust features.",
        context="Dropout prevents overfitting by randomly disabling neurons.",
    ))
    print()
    print(format_review_example(
        "What is backpropagation?",
        "It calculates gradients using the chain rule.",
        "Score: 3/5. You're correct that chain rule is involved, but your answer "
        "doesn't mention that backprop propagates error signals from output to "
        "input layer, nor that it's used to update weights via gradient descent.",
    ))
