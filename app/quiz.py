# quiz.py
"""
Multiple-choice quiz generation and evaluation.

Uses structured JSON generation: the model is prompted to output a strict
JSON schema which is then parsed and validated, with up to 3 retries and
progressively more permissive fallback parsing (markdown stripping, regex
field extraction) to handle common model output defects.
"""

import json
import re
from dataclasses import dataclass
from typing import List, Optional

import torch
from generation import build_context
import re

@dataclass
class QuizQuestion:
    """A single multiple-choice quiz question."""
    question:      str
    options:       List[str]    # exactly 4 options
    correct_index: int          # 0-3
    explanation:   str
    topic:         str = ""
    source_chunk:  str = ""

    @property
    def correct_answer(self) -> str:
        return self.options[self.correct_index]

    def display(self) -> str:
        letters = ["A", "B", "C", "D"]
        lines   = [f"Question: {self.question}\n"]
        for letter, opt in zip(letters, self.options):
            lines.append(f"  {letter}) {opt}")
        return "\n".join(lines)

    def display_with_answer(self) -> str:
        letters = ["A", "B", "C", "D"]
        lines   = [f"Question: {self.question}\n"]
        for i, (letter, opt) in enumerate(zip(letters, self.options)):
            marker = " [correct]" if i == self.correct_index else ""
            lines.append(f"  {letter}) {opt}{marker}")
        lines.append(f"\nExplanation: {self.explanation}")
        return "\n".join(lines)


# Prompt for raw-completion (custom checkpoint)
QUIZ_PROMPT_TEMPLATE = """You are creating a multiple-choice quiz for a machine learning course.
Based on the following lecture material, generate ONE quiz question.

Lecture material:
{context}

Generate a quiz question about: {topic}

You MUST respond with ONLY a JSON object in this exact format, nothing else:
{{
  "question": "The question text here?",
  "options": [
    "First option (correct answer)",
    "Second option (plausible distractor)",
    "Third option (plausible distractor)",
    "Fourth option (plausible distractor)"
  ],
  "correct_index": 0,
  "explanation": "One sentence explaining why option 0 is correct."
}}

Rules:
- The question must be answerable from the provided lecture material
- All 4 options must be plausible (no obviously wrong answers)
- correct_index is 0-3 indicating which option in the list is correct
- Distractors should be common misconceptions or related but incorrect concepts
- Do not output anything except the JSON object"""


def _quiz_chat_messages(topic: str, context: str) -> list:
    """Chat-format messages for instruction-tuned models (Qwen, Mistral, etc.)."""
    system = (
        "You are a quiz generator for a machine learning course. "
        "Your only output is a single valid JSON object — no prose, no markdown fences, "
        "no explanation outside the JSON. "
        "The JSON must have exactly these keys: "
        "\"question\" (string), \"options\" (array of exactly 4 non-empty strings), "
        "\"correct_index\" (integer 0-3), \"explanation\" (string)."
    )
    user = (
        f"Generate one multiple-choice quiz question about the topic: \"{topic}\"\n\n"
        f"Base the question on this lecture material:\n{context}\n\n"
        "Requirements:\n"
        "- All 4 options must be plausible (no obviously wrong answers)\n"
        "- Distractors should be common misconceptions, not placeholder text\n"
        "- correct_index is 0, 1, 2, or 3\n"
        "- Randomise which position holds the correct answer\n"
        "Output only the JSON object."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]


def _extract_json(text: str) -> Optional[dict]:
    """
    Extract a quiz JSON object from model output.

    Tries in order:
      A) Clean known defects (markdown fences, trailing commas, bad options arrays,
         missing correct_index) then json.loads
      B) Outermost { } block + same cleaning
      C) Regex field-by-field extraction as last resort
    """
    original = text.strip()

    def _clean(s: str) -> str:
        s = re.sub(r'```(?:json)?\s*', '', s)
        s = s.replace('```', '')
        s = re.sub(r',\s*([}\]])', r'\1', s)
        return s.strip()

    def _fix_options_array(s: str) -> str:
        opt_start = s.find('"options"')
        if opt_start == -1:
            return s
        arr_open  = s.find('[', opt_start)
        arr_close = s.find(']', arr_open) if arr_open != -1 else -1
        if arr_open == -1 or arr_close == -1:
            return s
        arr       = s[arr_open:arr_close + 1]
        fixed_arr = re.sub(r'"[^"]{1,30}"\s*:\s*("(?:[^"\\]|\\.)*")', r'\1', arr)
        return s[:arr_open] + fixed_arr + s[arr_close + 1:]

    def _inject_correct_index(s: str, full_text: str) -> str:
        if '"correct_index"' in s:
            return s
        m = re.search(r'[Cc]orrect(?:_index)?[^0-9]{0,15}([0-3])', full_text)
        if not m:
            return s
        idx        = m.group(1)
        last_brace = s.rfind('}')
        if last_brace == -1:
            return s
        prefix = s[:last_brace].rstrip()
        sep    = ',' if prefix and prefix[-1] not in ('{', ',') else ''
        return prefix + sep + f'\n  "correct_index": {idx}\n' + s[last_brace:]

    # Strategy A
    cleaned = _clean(original)
    cleaned = _fix_options_array(cleaned)
    cleaned = _inject_correct_index(cleaned, original)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Strategy B
    start = original.find('{')
    end   = original.rfind('}')
    if start != -1 and end != -1 and end > start:
        candidate = _clean(original[start:end + 1])
        candidate = _fix_options_array(candidate)
        candidate = _inject_correct_index(candidate, original)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    # Strategy C
    return _extract_fields_by_regex(original)


def _extract_fields_by_regex(text: str) -> Optional[dict]:
    """Last-resort field-by-field extraction when the JSON parser cannot recover."""
    q_m = re.search(r'"question"\s*:\s*"((?:[^"\\]|\\.)+)"', text)
    if not q_m:
        return None

    e_m  = re.search(r'"explanation"\s*:\s*"((?:[^"\\]|\\.)+)"', text)
    ci_m = (re.search(r'"correct_index"\s*:\s*([0-3])', text) or
            re.search(r'[Cc]orrect(?:_index)?[^0-3]{0,20}([0-3])', text))
    if not ci_m:
        return None

    opt_start = text.find('"options"')
    if opt_start == -1:
        return None
    arr_open = text.find('[', opt_start)
    if arr_open == -1:
        return None

    region  = text[arr_open: arr_open + 1000]
    options: list[str] = []
    for m in re.finditer(r'"((?:[^"\\]|\\.){5,})"', region):
        after = region[m.end(): m.end() + 5].strip()
        if after.startswith(':'):
            val_m = re.match(r'\s*:\s*"((?:[^"\\]|\\.)+)"', region[m.end():m.end() + 200])
            if val_m:
                options.append(val_m.group(1))
        else:
            options.append(m.group(1))
        if len(options) >= 4:
            break

    if len(options) < 4:
        return None

    return {
        "question":      q_m.group(1),
        "options":       options[:4],
        "correct_index": int(ci_m.group(1)),
        "explanation":   e_m.group(1) if e_m else "No explanation provided.",
    }


def _validate_quiz_json(data: dict) -> Optional[QuizQuestion]:
    if not isinstance(data, dict):
        return None

    question = data.get("question", "")
    options  = data.get("options", [])
    correct  = data.get("correct_index", -1)
    explain  = data.get("explanation", "")

    if not question or len(question) < 10:
        return None
    if not isinstance(options, list) or len(options) != 4:
        return None
    
    # Reject if the question contains retriever path metadata
    if re.search(r'\[.+>\s*.+\]', question):
        print(f"  [validation] rejected: question contains source metadata")
        return None

    PLACEHOLDER_FRAGMENTS = {
        "first option", "second option", "third option", "fourth option",
        "plausible distractor", "correct answer)", "question text here",
    }
    for opt in options:
        if not isinstance(opt, str) or len(opt.strip()) < 5:
            return None
        if any(frag in opt.lower() for frag in PLACEHOLDER_FRAGMENTS):
            print(f"  [validation] rejected placeholder option: {opt!r}")
            return None

    if isinstance(correct, str):
        try:
            correct = int(correct)
        except ValueError:
            return None
    if not isinstance(correct, int) or correct not in range(4):
        return None

    return QuizQuestion(
        question      = question,
        options       = options,
        correct_index = correct,
        explanation   = explain or "No explanation provided.",
    )


def generate_quiz_question_with_model(
    topic:       str,
    model,
    tokenizer,
    retriever,
    device:      str   = "cpu",
    max_retries: int   = 3,
    temperature: float = 0.8,
    prefetched_chunk:  dict | None = None,
) -> Optional[QuizQuestion]:
    if prefetched_chunk:
        # Use the pre-selected chunk as primary context, then fetch one more
        # related chunk to give the model a bit more material
        extra   = retriever.query(topic, top_k=1)
        chunks  = [prefetched_chunk] + [c for c in extra if c != prefetched_chunk][:1]
    else:
        chunks  = retriever.query(topic, top_k=2)
        
    context = build_context(chunks)
    prompt  = QUIZ_PROMPT_TEMPLATE.format(context=context, topic=topic)

    for attempt in range(max_retries):
        input_ids = tokenizer.encode(prompt)
        x         = torch.tensor([input_ids], dtype=torch.long).to(device)
        out       = model.generate(
            x,
            max_new_tokens=400,
            temperature=temperature if attempt == 0 else 0.5,
            top_k=50,
            top_p=0.95,
            stop_token=tokenizer.eot_id,
        )
        generated = tokenizer.decode(out[0, len(input_ids):].tolist())
        generated = generated.replace("<|endoftext|>", "").strip()

        data = _extract_json(generated)
        if data:
            quiz = _validate_quiz_json(data)
            if quiz:
                quiz.topic        = topic
                quiz.source_chunk = context
                return quiz

        print(f"  Attempt {attempt + 1} failed to produce valid JSON. Retrying...")

    print(f"  Failed after {max_retries} attempts.")
    return None


def generate_quiz_question_with_pretrained(
    topic:       str,
    pipe,
    retriever,
    max_retries: int   = 3,
    temperature: float = 0.7,
    prefetched_chunk:  dict | None = None,
) -> Optional[QuizQuestion]:
    print(f"  [pretrained] generating quiz question for topic: {topic!r}")
    # chunks   = retriever.query(topic, top_k=2)
    if prefetched_chunk:
        # Use the pre-selected chunk as primary context, then fetch one more
        # related chunk to give the model a bit more material
        extra   = retriever.query(topic, top_k=1)
        chunks  = [prefetched_chunk] + [c for c in extra if c != prefetched_chunk][:1]
    else:
        chunks  = retriever.query(topic, top_k=2)
        
    context = build_context(chunks)
    print(f"  [pretrained] context cleaned: {context!r}")
    messages = _quiz_chat_messages(topic, context)

    for attempt in range(max_retries):
        temp = temperature if attempt == 0 else 0.5
        try:
            result    = pipe(messages, max_new_tokens=400, do_sample=True,
                             temperature=temp, top_p=0.95, return_full_text=False)
            raw       = result[0]["generated_text"]
            generated = (raw[-1].get("content", "") if isinstance(raw, list) else str(raw)).strip()
        except Exception as e:
            print(f"  Pipeline error on attempt {attempt + 1}: {e}")
            continue

        print(f"  [pretrained] attempt {attempt + 1} raw: {repr(generated[:120])}")

        data = _extract_json(generated)
        if data:
            quiz = _validate_quiz_json(data)
            if quiz:
                quiz.topic        = topic
                quiz.source_chunk = context
                return quiz

        print(f"  Attempt {attempt + 1} failed. Retrying...")

    print(f"  Failed after {max_retries} attempts.")
    return None


def generate_quiz_question_with_gpt(
    topic:   str,
    context: str,
    api_key: str,
    model:   str = "gpt-4o-mini",
) -> Optional[QuizQuestion]:
    """Fallback to GPT-4 for reliable JSON output or training data generation."""
    import openai
    client = openai.OpenAI(api_key=api_key)
    prompt = QUIZ_PROMPT_TEMPLATE.format(context=context, topic=topic)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=400,
            response_format={"type": "json_object"},
        )
        data = json.loads(response.choices[0].message.content)
        return _validate_quiz_json(data)
    except Exception as e:
        print(f"GPT quiz generation failed: {e}")
        return None


def evaluate_quiz_question(quiz: QuizQuestion) -> dict:
    """
    Automated quality checks: structural validity, option uniqueness,
    and distractor plausibility (word-overlap similarity).
    """
    issues  = []
    metrics = {}

    if len(quiz.options) != 4:
        issues.append("Must have exactly 4 options")
    if not 0 <= quiz.correct_index <= 3:
        issues.append("correct_index must be 0-3")
    if len(quiz.question) < 15:
        issues.append("Question too short")
    if len(quiz.explanation) < 10:
        issues.append("Explanation too short")

    for i, o1 in enumerate(quiz.options):
        for j, o2 in enumerate(quiz.options):
            if i >= j:
                continue
            w1, w2 = set(o1.lower().split()), set(o2.lower().split())
            if w1 and w2:
                overlap = len(w1 & w2) / len(w1 | w2)
                if overlap > 0.8:
                    issues.append(f"Options {i} and {j} are too similar (Jaccard={overlap:.2f})")

    correct_words = set(quiz.correct_answer.lower().split())
    distractor_sims = []
    for i, opt in enumerate(quiz.options):
        if i == quiz.correct_index:
            continue
        opt_words = set(opt.lower().split())
        if correct_words and opt_words:
            distractor_sims.append(len(correct_words & opt_words) / len(correct_words | opt_words))

    avg_sim = sum(distractor_sims) / len(distractor_sims) if distractor_sims else 0
    metrics["avg_distractor_similarity"] = round(avg_sim, 3)
    if avg_sim < 0.05:
        issues.append(f"Distractors may be unrelated (avg word overlap: {avg_sim:.2f})")

    try:
        from bert_score import score as bert_score
        distractors = [quiz.options[i] for i in range(4) if i != quiz.correct_index]
        references  = [quiz.correct_answer] * len(distractors)
        _, _, F1    = bert_score(distractors, references, lang="en", verbose=False)
        metrics["bert_score_f1_distractors"] = float(F1.mean().item())
    except ImportError:
        pass

    metrics["valid"]  = len(issues) == 0
    metrics["issues"] = issues
    return metrics


class QuizSession:
    """Simple stateful quiz session for terminal use."""

    def __init__(self, rag_pipeline, topics: List[str] = None):
        self.rag     = rag_pipeline
        self.topics  = topics or [
            "dropout regularization", "backpropagation and gradients",
            "attention mechanism in transformers", "convolutional neural networks",
            "batch normalization", "gradient descent optimization",
            "overfitting and regularization", "recurrent neural networks",
            "loss functions", "activation functions",
        ]
        self.history = []
        self.current = None

    def next_question(self, topic: str = None) -> Optional[QuizQuestion]:
        import random
        topic = topic or random.choice(self.topics)
        self.current = generate_quiz_question_with_model(
            topic=topic, model=self.rag.model, tokenizer=self.rag.tokenizer,
            retriever=self.rag.retriever, device=self.rag.device,
        )
        return self.current

    def submit_answer(self, answer_index: int) -> dict:
        if self.current is None:
            return {"error": "No active question"}
        correct = (answer_index == self.current.correct_index)
        self.history.append((self.current, answer_index, correct))
        return {
            "correct":        correct,
            "correct_index":  self.current.correct_index,
            "correct_answer": self.current.correct_answer,
            "explanation":    self.current.explanation,
            "your_answer":    self.current.options[answer_index] if 0 <= answer_index < 4 else "Invalid",
            "score":          self.score(),
        }

    def score(self) -> dict:
        if not self.history:
            return {"correct": 0, "total": 0, "pct": 0.0}
        n = sum(1 for _, _, c in self.history if c)
        return {"correct": n, "total": len(self.history), "pct": 100 * n / len(self.history)}


if __name__ == "__main__":
    quiz = QuizQuestion(
        question      = "What does dropout do during training?",
        options       = [
            "Randomly zeroes neuron outputs to prevent co-adaptation",
            "Removes the worst-performing layers from the network",
            "Reduces the learning rate by a fixed factor each epoch",
            "Adds Gaussian noise to the input data",
        ],
        correct_index = 0,
        explanation   = "Dropout randomly sets a fraction of activations to zero, "
                        "forcing the network to learn redundant representations.",
        topic         = "dropout regularization",
    )
    print(quiz.display())
    print("\n--- With answer ---")
    print(quiz.display_with_answer())
    print(f"\nQuality: {evaluate_quiz_question(quiz)}")