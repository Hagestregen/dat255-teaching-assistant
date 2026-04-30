"""
rag_retriever.py  —  Minimal shared RAG helper
===============================================
Thin wrapper around the existing rag/retriever.py.
Lives in eval/ so any eval script can import it directly.

Usage:
    from rag_retriever import RAGRetriever, build_rag_prompt

    rag = RAGRetriever()                      # auto-finds rag_index/
    chunks = rag.retrieve("What is dropout?")
    prompt = build_rag_prompt("What is dropout?", chunks)
"""

import sys
from pathlib import Path
from typing import Optional

# Workspace root = two levels up from this file (eval/ → dat255-teaching-assistant/)
_WORKSPACE = Path(__file__).resolve().parent.parent
_RAG_INDEX  = _WORKSPACE / "rag" / "rag_index"

# Make rag/ importable
if str(_WORKSPACE) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE))


class RAGRetriever:
    """
    Wraps rag/retriever.py with auto-path resolution.
    Lazy-loads on first call so importing this file has no cost.
    """

    def __init__(self, index_dir: Optional[str] = None, top_k: int = 3,
                 context_token_budget: int = 300):
        self.index_dir   = Path(index_dir) if index_dir else _RAG_INDEX
        self.top_k       = top_k
        self.budget      = context_token_budget
        self._retriever  = None   # loaded on first use

    def _load(self):
        if self._retriever is None:
            from rag.retriever import Retriever
            self._retriever = Retriever(index_dir=str(self.index_dir))
            print(f"  RAG index loaded from: {self.index_dir}")

    def retrieve(self, question: str, top_k: Optional[int] = None) -> list[dict]:
        """
        Return top-k chunks most relevant to the question.
        Each chunk dict has: text, score, source, breadcrumb
        """
        self._load()
        k = top_k or self.top_k
        return self._retriever.query(question, top_k=k)

    def build_context(self, chunks: list[dict]) -> str:
        """
        Concatenate chunks into a context string, truncating to token budget.
        Highest-scoring chunks come first and are least likely to be cut.
        """
        parts, tokens_used = [], 0
        for chunk in chunks:
            text   = chunk["text"]
            # ~4 chars per token (rough GPT-2 average)
            est    = len(text) // 4
            if tokens_used + est > self.budget:
                chars_left = (self.budget - tokens_used) * 4
                if chars_left > 100:
                    parts.append(text[:chars_left] + "...")
                break
            parts.append(text)
            tokens_used += est
        return "\n---\n".join(parts)


def build_rag_prompt(question: str, chunks: list[dict],
                     retriever: Optional[RAGRetriever] = None,
                     for_custom: bool = False,
                     question_type: str = "open_ended") -> str:
    """
    Build a prompt with retrieved context prepended.

    for_custom=True  → wraps in task token format matching your training data
    for_custom=False → plain context block suitable for HF instruct models
    """
    if retriever is None:
        retriever = RAGRetriever()

    context = retriever.build_context(chunks)

    if for_custom:
        TASK_PREFIX = {
            "multiple_choice": "<|quiz|>",
            "open_ended":      "<|explain|>",
        }
        tok = TASK_PREFIX.get(question_type, "<|explain|>")
        return (
            f"{tok} Context: {context}\n\n"
            f"Question: {question}\nAnswer:"
        )

    return (
        f"Context:\n{context}\n\n"
        f"Question: {question}"
    )