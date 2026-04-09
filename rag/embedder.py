#!/usr/bin/env python3
"""
embedder.py
===========
Reads chunks.json (produced by chunker.py), generates embeddings using
SentenceTransformers, and saves:

    embeddings.npy   — raw numpy array  (num_chunks × dim)
    index.bin        — hnswlib HNSW index for fast ANN search
    chunks.json      — untouched; embedder reads but doesn't modify it

Usage:
    python embedder.py [--chunks chunks.json] [--out-dir ./rag_index] [--device cpu|cuda]

Requirements:
    pip install sentence-transformers hnswlib numpy
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

# ── Model config ──────────────────────────────────────────────────────────────
#
# CPU — smaller, faster, slightly lower quality
CPU_MODEL  = "all-MiniLM-L6-v2"          # 384-dim,  ~80 MB
#
# CUDA — larger, slower to load but better retrieval quality
CUDA_MODEL = "BAAI/bge-base-en-v1.5"     # 768-dim, ~440 MB
#                                         # Also good: "all-mpnet-base-v2"
#
# HNSW construction params (tune if your index is huge, defaults are fine)
HNSW_EF_CONSTRUCTION = 200
HNSW_M               = 16
# ──────────────────────────────────────────────────────────────────────────────


def detect_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def load_chunks(chunks_path: Path) -> list[dict]:
    with chunks_path.open(encoding="utf-8") as f:
        return json.load(f)


def embed_chunks(chunks: list[dict], model_name: str, device: str,
                 batch_size: int = 64) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    print(f"  Loading model '{model_name}' on {device} …")
    model = SentenceTransformer(model_name, device=device)

    texts = [c["text"] for c in chunks]
    print(f"  Embedding {len(texts)} chunks (batch_size={batch_size}) …")

    t0 = time.time()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,   # cosine similarity = dot product after normalisation
        show_progress_bar=True,
    )
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s  — shape: {embeddings.shape}")
    return embeddings


def build_hnsw_index(embeddings: np.ndarray) -> "hnswlib.Index":
    import hnswlib

    dim = embeddings.shape[1]
    n   = embeddings.shape[0]

    index = hnswlib.Index(space="cosine", dim=dim)
    index.init_index(
        max_elements=n,
        ef_construction=HNSW_EF_CONSTRUCTION,
        M=HNSW_M,
    )
    index.add_items(embeddings, np.arange(n))
    # ef at query time — higher = more accurate, slower; 50 is a good default
    index.set_ef(50)
    return index


def main():
    parser = argparse.ArgumentParser(description="Embed chunks and build HNSW index")
    parser.add_argument("--chunks",  default="chunks.json",
                        help="chunks.json produced by chunker.py")
    parser.add_argument("--out-dir", default="rag_index",
                        help="Directory to save index files (default: rag_index/)")
    parser.add_argument("--device",  default=None,
                        choices=["cpu", "cuda"],
                        help="Compute device (default: auto-detect)")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    device = args.device or detect_device()
    model_name = CUDA_MODEL if device == "cuda" else CPU_MODEL
    print(f"Device: {device}  →  model: {model_name}")

    chunks_path = Path(args.chunks)
    if not chunks_path.exists():
        raise FileNotFoundError(f"chunks.json not found: {chunks_path}\n"
                                "Run chunker.py first.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load ──────────────────────────────────────────────────────────────────
    print(f"\nLoading chunks from {chunks_path} …")
    chunks = load_chunks(chunks_path)
    print(f"  {len(chunks)} chunks loaded.")

    # ── Embed ─────────────────────────────────────────────────────────────────
    print("\nEmbedding …")
    embeddings = embed_chunks(chunks, model_name, device, args.batch_size)

    # ── Save embeddings ───────────────────────────────────────────────────────
    emb_path = out_dir / "embeddings.npy"
    np.save(emb_path, embeddings)
    print(f"\nEmbeddings saved → {emb_path}")

    # ── Build and save HNSW index ─────────────────────────────────────────────
    print("Building HNSW index …")
    index = build_hnsw_index(embeddings)
    index_path = out_dir / "index.bin"
    index.save_index(str(index_path))
    print(f"HNSW index saved  → {index_path}")

    # ── Save a copy of the chunk texts + metadata alongside the index ─────────
    # (retriever needs to map index IDs back to text)
    import shutil
    dest_chunks = out_dir / "chunks.json"
    shutil.copy(chunks_path, dest_chunks)
    print(f"Chunks copied     → {dest_chunks}")

    # Save which model was used (retriever needs the same dim)
    meta = {"model": model_name, "device": device, "dim": int(embeddings.shape[1]),
            "num_chunks": len(chunks)}
    (out_dir / "index_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    print(f"\nAll done! Index saved to ./{out_dir}/")


if __name__ == "__main__":
    main()