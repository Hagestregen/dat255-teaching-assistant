#!/usr/bin/env python3
"""
retriever.py
============
Loads a pre-built HNSW index (from embedder.py) and provides a clean
retrieval interface.

Can be used:
  1. As a standalone CLI for quick testing
  2. Imported as a module in your RAG application

Key feature: metadata-based filtering
    - Filter by h1/h2/h3 header (e.g. restrict to "Chapter 3")
    - Filter by source file (e.g. only notebook slides)

Usage (CLI):
    python retriever.py --query "What is self-attention?" --top-k 5
    python retriever.py --query "Convolutional layers" --filter-h1 "Convolutional networks"

Usage (as module):
    from retriever import Retriever
    r = Retriever()
    results = r.query("What is self-attention?", top_k=3)
    for chunk in results:
        print(chunk["score"], chunk["breadcrumb"])
        print(chunk["text"])

Requirements:
    pip install sentence-transformers hnswlib numpy
"""

import argparse
import json
from pathlib import Path

import numpy as np

DEFAULT_INDEX_DIR = "rag_index"


def detect_device() -> str:
    """Return 'cuda' or 'cpu' depending on what's available."""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


class Retriever:
    def __init__(self, index_dir: str = DEFAULT_INDEX_DIR, device: str | None = None):
        import hnswlib
        from sentence_transformers import SentenceTransformer

        index_dir = Path(index_dir)
        self.device = device or detect_device()

        # ── Load metadata (tells us which model to use) ───────────────────────
        meta_path = index_dir / "index_meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"index_meta.json not found in {index_dir}. Run embedder.py first."
            )
        with meta_path.open(encoding="utf-8") as f:
            meta = json.load(f)

        self.model_name  = meta["model"]
        self.dim         = meta["dim"]
        self.num_chunks  = meta["num_chunks"]

        # ── Load chunk texts ──────────────────────────────────────────────────
        chunks_path = index_dir / "chunks.json"
        with chunks_path.open(encoding="utf-8") as f:
            self.chunks: list[dict] = json.load(f)

        # ── Load HNSW index ───────────────────────────────────────────────────
        self.index = hnswlib.Index(space="cosine", dim=self.dim)
        self.index.load_index(
            str(index_dir / "index.bin"),
            max_elements=self.num_chunks,
        )
        self.index.set_ef(50)   # query-time ef; raise for better accuracy

        # ── Load embedding model ──────────────────────────────────────────────
        print(f"Loading embedding model '{self.model_name}' on {self.device} …")
        self.model = SentenceTransformer(self.model_name, device=self.device)
        print("Retriever ready.\n")

    # ── Core retrieval ────────────────────────────────────────────────────────

    def query(
        self,
        query_text:    str,
        top_k:         int  = 5,
        filter_h1:     str | None = None,
        filter_h2:     str | None = None,
        filter_source: str | None = None,
        min_score:     float = 0.0,
    ) -> list[dict]:
        """
        Retrieve the top-k most relevant chunks for a query.

        Parameters
        ----------
        query_text    : the question or query string
        top_k         : number of results to return
        filter_h1     : if set, only return chunks whose h1 header contains
                        this string (case-insensitive substring match)
        filter_h2     : same for h2
        filter_source : if set, only return chunks from files whose path
                        contains this string (e.g. "slide_md" or "notebook_md")
        min_score     : discard results below this cosine similarity (0–1)

        Returns
        -------
        List of dicts with keys:
            text, source, metadata, score, breadcrumb
        Sorted by score descending.
        """
        # Over-fetch to account for filtering losses
        fetch_k = min(top_k * 10, self.num_chunks)

        query_vec = self.model.encode(
            [query_text], normalize_embeddings=True
        )
        labels, distances = self.index.knn_query(query_vec, k=fetch_k)

        results = []
        for idx, dist in zip(labels[0], distances[0]):
            score = float(1.0 - dist)   # cosine distance → similarity
            if score < min_score:
                continue

            chunk = self.chunks[int(idx)]
            meta  = chunk.get("metadata", {})

            # ── Apply filters ─────────────────────────────────────────────────
            if filter_h1 and not _icontains(meta.get("h1") or "", filter_h1):
                continue
            if filter_h2 and not _icontains(meta.get("h2") or "", filter_h2):
                continue
            if filter_source and filter_source.lower() not in chunk.get("source", "").lower():
                continue

            results.append({
                "score":      score,
                "text":       chunk["text"],
                "source":     chunk.get("source", ""),
                "metadata":   meta,
                "breadcrumb": meta.get("breadcrumb", ""),
            })

            if len(results) >= top_k:
                break

        return results

    def format_results(self, results: list[dict], show_text: bool = True) -> str:
        """Pretty-print retrieval results."""
        if not results:
            return "No results found."
        lines = []
        for i, r in enumerate(results, 1):
            lines.append(f"{'─'*60}")
            lines.append(f"#{i}  score={r['score']:.3f}  |  {r['breadcrumb'] or '(no heading)'}")
            lines.append(f"    source: {r['source']}")
            if show_text:
                # Show up to 400 chars of the chunk
                preview = r["text"][:400].replace("\n", " ")
                if len(r["text"]) > 400:
                    preview += " …"
                lines.append(f"\n{preview}\n")
        return "\n".join(lines)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _icontains(haystack: str, needle: str) -> bool:
    return needle.lower() in haystack.lower()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Query the RAG index")
    parser.add_argument("--query",         required=True,
                        help="The question to retrieve context for")
    parser.add_argument("--top-k",         type=int, default=5)
    parser.add_argument("--index-dir",     default=DEFAULT_INDEX_DIR)
    parser.add_argument("--filter-h1",     default=None,
                        help="Only return chunks from this h1 section (substring)")
    parser.add_argument("--filter-h2",     default=None)
    parser.add_argument("--filter-source", default=None,
                        help="Only return chunks whose source path contains this string")
    parser.add_argument("--min-score",     type=float, default=0.0)
    parser.add_argument("--no-text",       action="store_true",
                        help="Don't print chunk text, only scores and sources")
    parser.add_argument("--device",        default=None,
                        choices=["cpu", "cuda"],
                        help="Compute device for the embedding model (default: auto-detect)")
    args = parser.parse_args()

    r = Retriever(index_dir=args.index_dir, device=args.device)
    results = r.query(
        args.query,
        top_k=args.top_k,
        filter_h1=args.filter_h1,
        filter_h2=args.filter_h2,
        filter_source=args.filter_source,
        min_score=args.min_score,
    )

    print(f'Query: "{args.query}"')
    print(f"Top {len(results)} results:\n")
    print(r.format_results(results, show_text=not args.no_text))


if __name__ == "__main__":
    main()