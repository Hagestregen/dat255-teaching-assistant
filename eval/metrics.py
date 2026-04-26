"""
metrics.py  —  QA evaluation metrics
=====================================
Pure-Python implementations of the standard QA metrics so the eval harness
has zero extra dependencies.  ROUGE-L is computed from longest common
subsequence directly (no `rouge_score` package needed).

Provides:
  - normalize_text(s)          — lowercase, drop punctuation/articles
  - exact_match(pred, gold)    — 0 or 1
  - token_f1(pred, gold)       — unigram F1
  - rouge_l(pred, gold)        — F1 of longest common subsequence
  - aggregate_scores(records)  — mean per metric across a list of records
  - llm_judge(...)             — optional, uses HuggingFace pipeline

`llm_judge` is opt-in: it lazily loads a generator via
`model.data_generation.load_generator`, prompts it with a fixed rubric,
and parses the JSON response.  When the judge model isn't available
(no GPU, model not downloaded, etc.) it returns None and the run_eval
script silently skips that metric.
"""

from __future__ import annotations

import json
import re
import string
from collections import Counter
from typing import Iterable, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Text normalization
# ─────────────────────────────────────────────────────────────────────────────

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
_WHITESPACE = re.compile(r"\s+")
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize_text(s: str) -> str:
    """Lowercase, strip punctuation and articles, collapse whitespace."""
    if s is None:
        return ""
    s = s.lower()
    s = s.translate(_PUNCT_TABLE)
    s = _ARTICLES.sub(" ", s)
    s = _WHITESPACE.sub(" ", s).strip()
    return s


# ─────────────────────────────────────────────────────────────────────────────
# Surface-level metrics
# ─────────────────────────────────────────────────────────────────────────────

def exact_match(pred: str, gold: str) -> float:
    return float(normalize_text(pred) == normalize_text(gold))


def token_f1(pred: str, gold: str) -> float:
    p_toks = normalize_text(pred).split()
    g_toks = normalize_text(gold).split()
    if not p_toks and not g_toks:
        return 1.0
    if not p_toks or not g_toks:
        return 0.0
    common = Counter(p_toks) & Counter(g_toks)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(p_toks)
    recall    = overlap / len(g_toks)
    return 2 * precision * recall / (precision + recall)


def _lcs_length(a: list[str], b: list[str]) -> int:
    """Length of the longest common subsequence (token-level). O(|a|*|b|)."""
    if not a or not b:
        return 0
    n, m = len(a), len(b)
    # Single-row DP for memory.
    prev = [0] * (m + 1)
    curr = [0] * (m + 1)
    for i in range(1, n + 1):
        ai = a[i - 1]
        for j in range(1, m + 1):
            if ai == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, prev
        for j in range(m + 1):
            curr[j] = 0
    return prev[m]


def rouge_l(pred: str, gold: str) -> float:
    """ROUGE-L F1: F-score of LCS-based precision and recall."""
    p_toks = normalize_text(pred).split()
    g_toks = normalize_text(gold).split()
    if not p_toks and not g_toks:
        return 1.0
    if not p_toks or not g_toks:
        return 0.0
    lcs = _lcs_length(p_toks, g_toks)
    if lcs == 0:
        return 0.0
    precision = lcs / len(p_toks)
    recall    = lcs / len(g_toks)
    return 2 * precision * recall / (precision + recall)


# ─────────────────────────────────────────────────────────────────────────────
# LLM-as-judge
# ─────────────────────────────────────────────────────────────────────────────

JUDGE_PROMPT = """You are grading a student answer against a reference answer.

Question: {question}
Reference answer: {gold}
Student answer: {pred}

Rate the student's answer on three dimensions, each on a 1-5 integer scale:
- correctness: factual accuracy compared to the reference.
- completeness: does it cover the key points?
- clarity: is it well written and unambiguous?

Output ONLY a JSON object. No markdown, no backticks, no extra text.

Output format:
{{"correctness": <1-5>, "completeness": <1-5>, "clarity": <1-5>, "reason": "<one sentence>"}}"""


_judge_pipeline = None


def _load_judge(model_id: str):
    """Lazy-load the judge model.  Reused across calls."""
    global _judge_pipeline
    if _judge_pipeline is not None:
        return _judge_pipeline
    # Defer the import so eval harness works even without transformers
    from model.data_generation import load_generator
    _judge_pipeline = load_generator(model_id)
    return _judge_pipeline


def llm_judge(
    question: str,
    pred: str,
    gold: str,
    model_id: str = "Qwen/Qwen2.5-3B-Instruct",
    max_new_tokens: int = 200,
) -> Optional[dict]:
    """
    Run an LLM judge on a single (question, prediction, reference) triple.

    Returns a dict like {"correctness": int, "completeness": int,
    "clarity": int, "reason": str} or None on failure.
    """
    try:
        pipe = _load_judge(model_id)
    except Exception as e:
        print(f"  [llm_judge] could not load judge: {e}")
        return None

    prompt = JUDGE_PROMPT.format(question=question, gold=gold, pred=pred)
    try:
        result = pipe(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=False,           # deterministic judge
            return_full_text=False,
        )
        text = result[0]["generated_text"].strip()
    except Exception as e:
        print(f"  [llm_judge] generation error: {e}")
        return None

    return _parse_judge_json(text)


def _parse_judge_json(text: str) -> Optional[dict]:
    """Robustly extract a JSON object from a possibly-noisy generation."""
    if not text:
        return None
    # Try direct parse
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        # Strip markdown fences and retry
        for fence in ("```json", "```JSON", "```"):
            if fence in text:
                for part in text.split(fence):
                    part = part.strip().rstrip("`").strip()
                    try:
                        obj = json.loads(part)
                        break
                    except json.JSONDecodeError:
                        continue
                else:
                    continue
                break
        else:
            # Fall back: find first { ... }
            start = text.find("{")
            end   = text.rfind("}")
            if start == -1 or end <= start:
                return None
            try:
                obj = json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                return None

    if not isinstance(obj, dict):
        return None

    # Coerce score fields to ints in 1..5 if possible
    out: dict = {}
    for key in ("correctness", "completeness", "clarity"):
        v = obj.get(key)
        try:
            iv = int(v)
            if 1 <= iv <= 5:
                out[key] = iv
        except (TypeError, ValueError):
            pass
    out["reason"] = str(obj.get("reason", ""))
    if not all(k in out for k in ("correctness", "completeness", "clarity")):
        return None
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_scores(records: Iterable[dict]) -> dict:
    """
    Take an iterable of per-example records (each with a "scores" dict) and
    return mean values per metric.  Missing keys are skipped.
    """
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    n_total = 0
    for rec in records:
        n_total += 1
        for k, v in (rec.get("scores") or {}).items():
            if isinstance(v, (int, float)):
                sums[k] = sums.get(k, 0.0) + float(v)
                counts[k] = counts.get(k, 0) + 1
    means = {k: sums[k] / counts[k] for k in sums}
    return {"n": n_total, "means": means, "counts": counts}


# ─────────────────────────────────────────────────────────────────────────────
# Quick self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pred = "Dropout is a regularization technique that randomly zeroes neurons."
    gold = "Dropout randomly zeroes neurons to regularize a network."
    print(f"EM:      {exact_match(pred, gold)}")
    print(f"F1:      {token_f1(pred, gold):.3f}")
    print(f"ROUGE-L: {rouge_l(pred, gold):.3f}")
