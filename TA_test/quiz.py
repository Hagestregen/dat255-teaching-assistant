"""
quiz.py  —  Interactive Quiz Mode for the Teaching Assistant
=============================================================
This module implements the quiz feature: the model generates multiple-choice
questions from course topics, parses the structured output, and evaluates
both the user's answers and the quality of the generated questions.

DOES THE MODEL NEED AGENT BEHAVIOUR FOR QUIZZING?
───────────────────────────────────────────────────
Short answer: No — not for a basic quiz. But a little structure helps a lot.

"Agent behaviour" usually means: observe → think → act → observe loop,
often with tool use (web search, code execution, etc.).

For quizzing, you need something simpler: STRUCTURED GENERATION.
Instead of free-form text, you guide the model to output JSON so you can
reliably parse the question, options, and correct answer.

The trick: use a constrained prompt that tells the model to output ONLY
valid JSON in a specific schema. Then wrap in try/except json.loads().

For a stronger version (with retries, topic tracking, adaptive difficulty),
you'd add a thin loop — that's closer to agent-style but still simple.

ARCHITECTURE OF QUIZ MODE:
───────────────────────────
  1. User specifies a topic (or it's inferred from context)
  2. RAG retrieves relevant chunks about that topic
  3. Model generates a JSON quiz question from the context
  4. Parse and validate the JSON (retry if malformed)
  5. Present to user, collect answer
  6. Score and explain
  7. (Optional) Evaluate quiz quality with BERTScore
"""

import json
import re
from dataclasses import dataclass
from typing import List, Optional

import torch


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: Quiz data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class QuizQuestion:
    """A single multiple-choice quiz question."""
    question:      str
    options:       List[str]    # exactly 4 options
    correct_index: int          # 0-3
    explanation:   str          # why the answer is correct
    topic:         str = ""
    source_chunk:  str = ""     # the lecture text it was based on

    @property
    def correct_answer(self) -> str:
        return self.options[self.correct_index]

    def display(self) -> str:
        """Pretty-print the question for the terminal or Gradio."""
        letters = ["A", "B", "C", "D"]
        lines = [f"Question: {self.question}\n"]
        for i, (letter, opt) in enumerate(zip(letters, self.options)):
            marker = "→" if i == self.correct_index else " "
            lines.append(f"  {letter}) {opt}")
        return "\n".join(lines)

    def display_with_answer(self) -> str:
        letters = ["A", "B", "C", "D"]
        lines = [f"Question: {self.question}\n"]
        for i, (letter, opt) in enumerate(zip(letters, self.options)):
            marker = " ✓" if i == self.correct_index else ""
            lines.append(f"  {letter}) {opt}{marker}")
        lines.append(f"\nExplanation: {self.explanation}")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: Quiz generation
# ─────────────────────────────────────────────────────────────────────────────

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


def _extract_json(text: str) -> Optional[dict]:
    """
    Extract a JSON object from model output.

    The model might output extra text before/after the JSON. We try:
    1. Direct parse
    2. Find JSON between { and } using regex
    3. Give up and return None
    """
    text = text.strip()

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find JSON block with regex
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return None


def _validate_quiz_json(data: dict) -> Optional[QuizQuestion]:
    """
    Validate that parsed JSON has the right structure for a quiz question.
    Returns QuizQuestion if valid, None otherwise.
    """
    if not isinstance(data, dict):
        return None

    question = data.get("question", "")
    options  = data.get("options", [])
    correct  = data.get("correct_index", -1)
    explain  = data.get("explanation", "")

    # Basic validation
    if not question or len(question) < 10:
        return None
    if not isinstance(options, list) or len(options) != 4:
        return None
    if not all(isinstance(o, str) and len(o) > 2 for o in options):
        return None
    if not isinstance(correct, int) or correct not in range(4):
        return None

    return QuizQuestion(
        question=question,
        options=options,
        correct_index=correct,
        explanation=explain or "No explanation provided.",
    )


def generate_quiz_question_with_model(
    topic:      str,
    model,
    tokenizer,
    retriever,
    device:     str   = "cpu",
    max_retries: int  = 3,
    temperature: float = 0.8,
) -> Optional[QuizQuestion]:
    """
    Generate a quiz question using your trained transformer + RAG.

    Strategy:
    1. Retrieve relevant chunks for the topic.
    2. Build a constrained prompt asking for JSON output.
    3. Run generation with low temperature (more predictable output).
    4. Parse and validate the JSON.
    5. Retry up to max_retries times if output is malformed.

    Note on small model limitations:
    Small models often generate slightly malformed JSON (extra text,
    wrong quote types, missing commas). The retry loop + _extract_json
    handles most of these cases. If the model consistently fails,
    fall back to generate_quiz_question_with_gpt().
    """
    # Retrieve context
    chunks = retriever.query(topic, top_k=2)
    context = "\n---\n".join(c["text"] for c in chunks[:2])[:600]

    prompt = QUIZ_PROMPT_TEMPLATE.format(context=context, topic=topic)

    for attempt in range(max_retries):
        input_ids = tokenizer.encode(prompt)
        x = torch.tensor([input_ids], dtype=torch.long).to(device)

        out = model.generate(
            x,
            max_new_tokens=400,
            temperature=temperature if attempt == 0 else 0.5,  # lower temp on retry
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

        print(f"  Attempt {attempt+1} failed to produce valid JSON. Retrying...")

    print(f"  Failed to generate valid quiz after {max_retries} attempts.")
    return None


def generate_quiz_question_with_gpt(
    topic:   str,
    context: str,
    api_key: str,
    model:   str = "gpt-4o-mini",
) -> Optional[QuizQuestion]:
    """
    Generate a quiz question using GPT-4.

    Use this for:
    1. Generating your synthetic training data for quiz examples
    2. Fallback when your small model produces malformed JSON
    3. Baseline comparison in the evaluation

    GPT-4 reliably produces valid JSON even with complex topics.
    """
    import openai
    client = openai.OpenAI(api_key=api_key)

    prompt = QUIZ_PROMPT_TEMPLATE.format(context=context, topic=topic)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=400,
            response_format={"type": "json_object"},  # guaranteed JSON output
        )
        data = json.loads(response.choices[0].message.content)
        return _validate_quiz_json(data)
    except Exception as e:
        print(f"GPT quiz generation failed: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: Quiz quality evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_quiz_question(quiz: QuizQuestion) -> dict:
    """
    Automated quality checks for a generated quiz question.

    Checks:
    1. Structural: 4 options, valid correct_index, non-empty question/explanation.
    2. Uniqueness: options shouldn't be near-duplicates.
    3. Plausibility: distractors should be somewhat related to the answer
       (if distractors are totally unrelated, the quiz is too easy).
       We check this with simple word-overlap similarity.

    For a stronger evaluation, use BERTScore as described in the project plan:
       pip install bert-score
       from bert_score import score
       score(distractors, [correct] * len(distractors), lang="en")
    The idea: BERTScore between distractors and the correct answer should be
    moderate (0.6-0.85). Too low = distractors are unrelated (too easy).
    Too high = distractors are near-synonyms (ambiguous question).
    """
    issues  = []
    metrics = {}

    # Check 1: Structural validity
    if len(quiz.options) != 4:
        issues.append("Must have exactly 4 options")
    if not 0 <= quiz.correct_index <= 3:
        issues.append("correct_index must be 0-3")
    if len(quiz.question) < 15:
        issues.append("Question too short")
    if len(quiz.explanation) < 10:
        issues.append("Explanation too short")

    # Check 2: Option uniqueness (no near-duplicates)
    for i, o1 in enumerate(quiz.options):
        for j, o2 in enumerate(quiz.options):
            if i >= j:
                continue
            # Jaccard similarity of words
            w1, w2 = set(o1.lower().split()), set(o2.lower().split())
            if w1 and w2:
                overlap = len(w1 & w2) / len(w1 | w2)
                if overlap > 0.8:
                    issues.append(f"Options {i} and {j} are too similar (Jaccard={overlap:.2f})")

    # Check 3: Distractor plausibility (word overlap with correct answer)
    correct_words = set(quiz.correct_answer.lower().split())
    distractor_similarities = []
    for i, opt in enumerate(quiz.options):
        if i == quiz.correct_index:
            continue
        opt_words = set(opt.lower().split())
        if correct_words and opt_words:
            sim = len(correct_words & opt_words) / len(correct_words | opt_words)
            distractor_similarities.append(sim)

    avg_distractor_sim = sum(distractor_similarities) / len(distractor_similarities) if distractor_similarities else 0
    metrics["avg_distractor_similarity"] = round(avg_distractor_sim, 3)

    if avg_distractor_sim < 0.05:
        issues.append(f"Distractors may be unrelated to answer (avg word overlap: {avg_distractor_sim:.2f})")

    # Optional: BERTScore evaluation
    try:
        from bert_score import score as bert_score
        distractors = [quiz.options[i] for i in range(4) if i != quiz.correct_index]
        references  = [quiz.correct_answer] * len(distractors)
        _, _, F1 = bert_score(distractors, references, lang="en", verbose=False)
        metrics["bert_score_f1_distractors"] = float(F1.mean().item())
    except ImportError:
        pass  # bert-score not installed, skip

    metrics["valid"]  = len(issues) == 0
    metrics["issues"] = issues
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: Interactive quiz session
# ─────────────────────────────────────────────────────────────────────────────

class QuizSession:
    """
    A quiz session that tracks questions asked, answers given, and score.

    This is the interface between the quiz generator and the user/Gradio.
    It's NOT an agent — it's a simple state machine:
      idle → question_generated → answered → (next question or done)

    If you wanted true agent behaviour (e.g., adapting difficulty, generating
    follow-up questions based on what the user got wrong), you'd add:
    - A "memory" of topics covered and performance per topic
    - A planning step: "user struggles with backprop → generate more backprop Qs"
    - Possibly GPT-4 as a "metacognitive" layer that decides the next topic

    For this project, the simple loop is enough. You can note in the report
    that extending to adaptive difficulty would be future work.
    """

    def __init__(self, rag_pipeline, topics: List[str] = None):
        """
        rag_pipeline: your RAGPipeline instance
        topics: list of topics to quiz on (if None, derived from retriever breadcrumbs)
        """
        self.rag      = rag_pipeline
        self.topics   = topics or self._default_topics()
        self.history  = []   # list of (question, user_answer, correct, quiz_obj)
        self.current  = None # current QuizQuestion

    def _default_topics(self) -> List[str]:
        """Default ML topics for the quiz."""
        return [
            "dropout regularization",
            "backpropagation and gradients",
            "attention mechanism in transformers",
            "convolutional neural networks",
            "batch normalization",
            "gradient descent optimization",
            "overfitting and regularization",
            "recurrent neural networks",
            "loss functions",
            "activation functions",
        ]

    def next_question(self, topic: str = None) -> Optional[QuizQuestion]:
        """Generate the next quiz question."""
        import random
        if topic is None:
            topic = random.choice(self.topics)

        print(f"Generating quiz question about: {topic}")
        quiz = generate_quiz_question_with_model(
            topic=topic,
            model=self.rag.model,
            tokenizer=self.rag.tokenizer,
            retriever=self.rag.retriever,
            device=self.rag.device,
        )
        self.current = quiz
        return quiz

    def submit_answer(self, answer_index: int) -> dict:
        """
        Submit an answer (0-3) and get feedback.

        Returns dict with: correct, correct_index, explanation, score
        """
        if self.current is None:
            return {"error": "No active question"}

        correct = (answer_index == self.current.correct_index)
        self.history.append((self.current, answer_index, correct))

        return {
            "correct":         correct,
            "correct_index":   self.current.correct_index,
            "correct_answer":  self.current.correct_answer,
            "explanation":     self.current.explanation,
            "your_answer":     self.current.options[answer_index] if 0 <= answer_index < 4 else "Invalid",
            "score":           self.score(),
        }

    def score(self) -> dict:
        if not self.history:
            return {"correct": 0, "total": 0, "pct": 0.0}
        n_correct = sum(1 for _, _, c in self.history if c)
        total     = len(self.history)
        return {"correct": n_correct, "total": total, "pct": 100 * n_correct / total}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: Command-line quiz (for testing)
# ─────────────────────────────────────────────────────────────────────────────

def run_terminal_quiz(rag_pipeline, n_questions: int = 5):
    """
    Run an interactive quiz in the terminal for testing.
    The Gradio UI in app.py wraps the same QuizSession logic.
    """
    session = QuizSession(rag_pipeline)
    letters = ["A", "B", "C", "D"]

    print("\n" + "="*60)
    print("       DAT255 Quiz Mode")
    print("="*60)

    for q_num in range(1, n_questions + 1):
        print(f"\n--- Question {q_num}/{n_questions} ---")
        quiz = session.next_question()

        if quiz is None:
            print("Could not generate question. Skipping.")
            continue

        # Display question
        print(f"\n{quiz.question}\n")
        for i, (letter, opt) in enumerate(zip(letters, quiz.options)):
            print(f"  {letter}) {opt}")

        # Get user answer
        while True:
            raw = input("\nYour answer (A/B/C/D): ").strip().upper()
            if raw in letters:
                answer_idx = letters.index(raw)
                break
            print("Please enter A, B, C, or D")

        # Evaluate
        result = session.submit_answer(answer_idx)
        if result["correct"]:
            print("✓ Correct!")
        else:
            print(f"✗ Wrong. Correct answer: {letters[result['correct_index']]})"
                  f" {result['correct_answer']}")
        print(f"  Explanation: {result['explanation']}")
        print(f"  Score so far: {result['score']['correct']}/{result['score']['total']}")

    # Final score
    final = session.score()
    print(f"\n{'='*60}")
    print(f"Final Score: {final['correct']}/{final['total']} ({final['pct']:.0f}%)")
    print("="*60)


if __name__ == "__main__":
    # Quick structural test (no model needed)
    quiz = QuizQuestion(
        question="What does dropout do during training?",
        options=[
            "Randomly zeroes neuron outputs to prevent co-adaptation",
            "Removes the worst-performing layers from the network",
            "Reduces the learning rate by a fixed factor each epoch",
            "Adds Gaussian noise to the input data",
        ],
        correct_index=0,
        explanation="Dropout randomly sets a fraction of activations to zero, "
                    "forcing the network to learn redundant representations.",
        topic="dropout regularization",
    )

    print(quiz.display())
    print("\n--- With answer revealed ---")
    print(quiz.display_with_answer())

    metrics = evaluate_quiz_question(quiz)
    print(f"\nQuality metrics: {metrics}")
