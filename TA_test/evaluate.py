"""
evaluate.py  —  Multi-metric evaluation framework
===================================================
Evaluates all three models (A: transformer alone, B: transformer + RAG, C: GPT-4)
on the same test set and produces a comparison report.

METRICS EXPLAINED:
──────────────────
1. GPT-4 as Judge (1-5 score):
   The most meaningful metric. You send:
     (question, model_answer, reference_answer) → GPT-4 → score 1-5
   This captures correctness, coherence, and relevance holistically.
   This is now a standard evaluation technique in NLP research ("LLM-as-judge").

2. BLEU Score:
   Measures n-gram overlap between generated and reference answers.
   Originally for machine translation. Fast and simple, but rewards exact
   wording rather than meaning. A score of 0.1-0.3 is typical for QA.

3. Semantic Similarity (BERTScore / cosine similarity):
   Uses a pretrained encoder to compare meaning, not exact words.
   Better than BLEU for open-ended answers where many phrasings are correct.
   BERTScore F1 > 0.85 = very similar meaning.

4. Answer Length (proxy for detail):
   Short answers often miss important detail. We track word count as a
   sanity check — a 2-word answer to a complex question is probably wrong.

Run this AFTER you have:
  - A trained checkpoint
  - Your RAG index built
  - A test set (from dataset.py's build_datasets test split)
"""

import json
import math
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: GPT-4 judge
# ─────────────────────────────────────────────────────────────────────────────

JUDGE_PROMPT = """You are evaluating the quality of an AI teaching assistant's answer to a machine learning question.

Question: {question}
Reference answer: {reference}
Model answer: {answer}

Score the model answer on a scale of 1-5:
  1 = Completely wrong or irrelevant
  2 = Partially correct but has significant errors
  3 = Mostly correct but missing important details
  4 = Correct and reasonably complete
  5 = Excellent, accurate, and well-explained

Respond with ONLY a JSON object: {{"score": <1-5>, "reason": "<one sentence>"}}"""


def gpt4_judge(
    question:  str,
    reference: str,
    answer:    str,
    api_key:   str,
    model:     str = "gpt-4o-mini",
) -> dict:
    """
    Use GPT-4 to score an answer against a reference.
    Returns {"score": 1-5, "reason": "..."}
    """
    import openai
    client = openai.OpenAI(api_key=api_key)

    prompt = JUDGE_PROMPT.format(
        question=question, reference=reference, answer=answer
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,  # deterministic for evaluation
            max_tokens=100,
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"  Judge failed: {e}")
        return {"score": -1, "reason": f"error: {e}"}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: BLEU score
# ─────────────────────────────────────────────────────────────────────────────

def bleu_score(reference: str, hypothesis: str, max_n: int = 4) -> float:
    """
    Compute BLEU score (corpus BLEU for a single example).

    BLEU = geometric mean of n-gram precisions with brevity penalty.

    We implement a simple version here to avoid heavy dependencies.
    For production, use sacrebleu: pip install sacrebleu
    """
    import re
    from collections import Counter

    def tokenize(text):
        return re.findall(r'\b\w+\b', text.lower())

    ref_tokens  = tokenize(reference)
    hyp_tokens  = tokenize(hypothesis)

    if not hyp_tokens:
        return 0.0

    # Brevity penalty: penalize very short hypotheses
    bp = min(1.0, len(hyp_tokens) / max(len(ref_tokens), 1))
    if bp == 0:
        return 0.0

    log_score = 0.0
    for n in range(1, max_n + 1):
        if len(hyp_tokens) < n:
            return 0.0

        # Count n-gram matches
        ref_ngrams  = Counter(tuple(ref_tokens[i:i+n])  for i in range(len(ref_tokens)  - n + 1))
        hyp_ngrams  = Counter(tuple(hyp_tokens[i:i+n]) for i in range(len(hyp_tokens) - n + 1))

        match_count = sum(min(hyp_ngrams[ng], ref_ngrams[ng]) for ng in hyp_ngrams)
        total_count = sum(hyp_ngrams.values())

        if total_count == 0 or match_count == 0:
            return 0.0

        precision = match_count / total_count
        log_score += (1/max_n) * math.log(precision)

    return bp * math.exp(log_score)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: Semantic similarity (cosine via sentence-transformers)
# ─────────────────────────────────────────────────────────────────────────────

_semantic_model = None  # lazy load

def semantic_similarity(text_a: str, text_b: str) -> float:
    """
    Cosine similarity between sentence embeddings.

    Uses the same sentence-transformers model as your RAG embedder,
    which means no extra dependencies.

    1.0 = identical meaning, 0.0 = orthogonal, can be negative for opposites.
    Typical "good answer" similarity: 0.7-0.95.
    """
    global _semantic_model
    if _semantic_model is None:
        from sentence_transformers import SentenceTransformer
        _semantic_model = SentenceTransformer("all-MiniLM-L6-v2")

    import torch
    vecs = _semantic_model.encode([text_a, text_b], normalize_embeddings=True)
    return float((vecs[0] * vecs[1]).sum())


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: Full evaluation run
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EvalResult:
    question:       str
    reference:      str
    answer_a:       str   # transformer only
    answer_b:       str   # transformer + RAG
    answer_c:       str   # GPT-4

    # Scores per model
    judge_a:        float = 0.0
    judge_b:        float = 0.0
    judge_c:        float = 0.0

    bleu_a:         float = 0.0
    bleu_b:         float = 0.0
    bleu_c:         float = 0.0

    sem_a:          float = 0.0
    sem_b:          float = 0.0
    sem_c:          float = 0.0

    len_a:          int   = 0
    len_b:          int   = 0
    len_c:          int   = 0


def run_evaluation(
    test_examples: List[Dict],   # list of {"question": ..., "answer": ...}
    rag_pipeline,                # your RAGPipeline instance
    openai_api_key: str,
    output_path: str = "evaluation_results.json",
    max_examples: int = 100,
    gpt_judge_model: str = "gpt-4o-mini",
    gpt_answer_model: str = "gpt-4o-mini",
) -> List[EvalResult]:
    """
    Evaluate all three models on the test set.

    For each question:
      1. Get answer from Model A (transformer only)
      2. Get answer from Model B (transformer + RAG)
      3. Get answer from Model C (GPT-4)
      4. Score all three with GPT-4 judge, BLEU, and semantic similarity

    This can take a while (API calls per example × 3 models for judging).
    For 100 examples × 3 judge calls = 300 API calls ≈ $0.10 with gpt-4o-mini.
    """
    from rag_pipeline import answer_with_gpt4, answer_question

    results = []
    sample  = test_examples[:max_examples]

    print(f"Evaluating {len(sample)} examples...")

    for i, ex in enumerate(sample):
        q   = ex["question"]
        ref = ex["answer"]

        print(f"  [{i+1}/{len(sample)}] {q[:60]}...")

        # ── Get answers ────────────────────────────────────────────────────
        # Model A: no RAG
        a_a = answer_question(q, rag_pipeline.model, rag_pipeline.tokenizer,
                               rag_pipeline.device)
        # Model B: with RAG
        a_b = rag_pipeline.answer(q)["answer"]
        # Model C: GPT-4
        a_c = answer_with_gpt4(q, openai_api_key, model=gpt_answer_model)

        # ── BLEU scores ────────────────────────────────────────────────────
        bleu_a = bleu_score(ref, a_a)
        bleu_b = bleu_score(ref, a_b)
        bleu_c = bleu_score(ref, a_c)

        # ── Semantic similarity ────────────────────────────────────────────
        sem_a = semantic_similarity(ref, a_a)
        sem_b = semantic_similarity(ref, a_b)
        sem_c = semantic_similarity(ref, a_c)

        # ── GPT-4 judge scores ─────────────────────────────────────────────
        j_a = gpt4_judge(q, ref, a_a, openai_api_key, gpt_judge_model)
        j_b = gpt4_judge(q, ref, a_b, openai_api_key, gpt_judge_model)
        j_c = gpt4_judge(q, ref, a_c, openai_api_key, gpt_judge_model)

        result = EvalResult(
            question=q, reference=ref,
            answer_a=a_a, answer_b=a_b, answer_c=a_c,
            judge_a=j_a.get("score", 0), judge_b=j_b.get("score", 0), judge_c=j_c.get("score", 0),
            bleu_a=bleu_a,  bleu_b=bleu_b,  bleu_c=bleu_c,
            sem_a=sem_a,    sem_b=sem_b,    sem_c=sem_c,
            len_a=len(a_a.split()), len_b=len(a_b.split()), len_c=len(a_c.split()),
        )
        results.append(result)

    # ── Save results ──────────────────────────────────────────────────────────
    with open(output_path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"\nResults saved to {output_path}")

    # ── Print summary ─────────────────────────────────────────────────────────
    print_summary(results)
    return results


def print_summary(results: List[EvalResult]):
    """Print a table comparing the three models."""
    def avg(vals):
        vals = [v for v in vals if v > 0]
        return sum(vals) / len(vals) if vals else 0.0

    print("\n" + "="*65)
    print(f"{'Metric':<22} {'Model A':>10} {'Model B+RAG':>12} {'GPT-4':>10}")
    print("-"*65)
    print(f"{'GPT-4 Judge (1-5)':<22} "
          f"{avg([r.judge_a for r in results]):>10.2f} "
          f"{avg([r.judge_b for r in results]):>12.2f} "
          f"{avg([r.judge_c for r in results]):>10.2f}")
    print(f"{'BLEU':<22} "
          f"{avg([r.bleu_a for r in results]):>10.3f} "
          f"{avg([r.bleu_b for r in results]):>12.3f} "
          f"{avg([r.bleu_c for r in results]):>10.3f}")
    print(f"{'Semantic sim.':<22} "
          f"{avg([r.sem_a for r in results]):>10.3f} "
          f"{avg([r.sem_b for r in results]):>12.3f} "
          f"{avg([r.sem_c for r in results]):>10.3f}")
    print(f"{'Avg answer length':<22} "
          f"{avg([r.len_a for r in results]):>10.0f} "
          f"{avg([r.len_b for r in results]):>12.0f} "
          f"{avg([r.len_c for r in results]):>10.0f}")
    print("="*65)
    print("Model A = transformer only | B = +RAG | C = GPT-4 baseline")


if __name__ == "__main__":
    # Quick test of metrics without a model
    ref  = "Dropout randomly zeroes a fraction of neuron activations during training."
    hyp1 = "Dropout is a regularization method that randomly disables neurons."
    hyp2 = "Dropout makes training slower."

    print(f"BLEU (good answer):  {bleu_score(ref, hyp1):.3f}")
    print(f"BLEU (bad answer):   {bleu_score(ref, hyp2):.3f}")
    print(f"Sem  (good answer):  {semantic_similarity(ref, hyp1):.3f}")
    print(f"Sem  (bad answer):   {semantic_similarity(ref, hyp2):.3f}")
