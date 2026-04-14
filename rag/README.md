# RAG Pipeline — Chunker · Embedder · Retriever

A lightweight, local RAG (Retrieval-Augmented Generation) pipeline for Markdown files.
No cloud services required — runs fully on CPU or CUDA GPU.

```text
your .md files
     │
     ▼
 chunker.py  ──→  chunks.json
     │
     ▼
 embedder.py ──→  rag_index/  (index.bin, embeddings.npy, chunks.json, index_meta.json)
     │
     ▼
 retriever.py ──→  top-k relevant chunks for any query
```

---

## Installation

```bash
pip install langchain langchain-text-splitters sentence-transformers hnswlib numpy torch
```

> `torch` is only needed for GPU/MPS auto-detection. CPU-only installs can skip it,
> but device detection will fall back to CPU automatically anyway.

---

## Step 1 — Chunk your Markdown files (`chunker.py`)

Reads a manifest listing your `.md` files or folders, splits them
structure-aware on headers (`#` / `##` / `###` / `####`), then sub-chunks
any section still larger than ~1 000 characters. Outputs `chunks.json`.

### Create a manifest

```text
# sources.txt
/path/to/docs/folder
/path/to/single_file.md
relative/paths/work/too
```

Lines starting with `#` are ignored. Folders are expanded recursively for all `.md` files.

### Run

```bash
python chunker.py --manifest sources.txt
# Output: rag_index/chunks.json  (default)

python chunker.py --manifest sources.txt --output my_chunks.json
```

### Key config (top of file)

| Variable              | Default               | Effect                           |
| --------------------- | --------------------- | -------------------------------- |
| `PASS2_CHUNK_SIZE`    | `1000`                | Max characters per chunk         |
| `PASS2_CHUNK_OVERLAP` | `100`                 | Overlap between sub-chunks       |
| `HEADERS_TO_SPLIT_ON` | `#` `##` `###` `####` | Header levels used as boundaries |

---

## Step 2 — Embed and index (`embedder.py`)

Reads `chunks.json`, generates embeddings with SentenceTransformers, and builds
an HNSW approximate-nearest-neighbour index via `hnswlib`.

### Run

```bash
# Auto-detect device (CPU / CUDA)
python embedder.py

# Explicit device
python embedder.py --device cuda
python embedder.py --device cpu

# Custom paths
python embedder.py --chunks my_chunks.json --out-dir my_index

# Override the embedding model (must match dim if reusing an existing index)
python embedder.py --model BAAI/bge-base-en-v1.5
```

### Output (inside `rag_index/` by default)

| File              | Contents                     |
| ----------------- | ---------------------------- |
| `index.bin`       | HNSW binary index            |
| `embeddings.npy`  | Raw embedding matrix         |
| `chunks.json`     | Copy of input chunks         |
| `index_meta.json` | Model name, dim, chunk count |

### Default models

| Device     | Model                   | Dim | Notes                   |
| ---------- | ----------------------- | --- | ----------------------- |
| CPU        | `all-MiniLM-L6-v2`      | 384 | Fast, ~80 MB            |
| CUDA / MPS | `BAAI/bge-base-en-v1.5` | 768 | Better quality, ~440 MB |

---

## Step 3 — Query (`retriever.py`)

Loads the index from disk and retrieves the most semantically relevant chunks
for any query string. Can be used from the CLI or imported as a module.

### CLI

```bash
# Basic query
python retriever.py --query "What is self-attention?"

# Return more results
python retriever.py --query "Convolutional layers" --top-k 10

# Filter to a specific section (substring match, case-insensitive)
python retriever.py --query "Backpropagation" --filter-h1 "Chapter 3"
python retriever.py --query "Backpropagation" --filter-h2 "Gradients"

# Filter by source file path
python retriever.py --query "Attention mechanism" --filter-source "slides"

# Drop results below a similarity threshold (0–1)
python retriever.py --query "Loss functions" --min-score 0.5

# Hide chunk text, show scores and sources only
python retriever.py --query "Dropout" --no-text

# Custom index location
python retriever.py --query "Embeddings" --index-dir my_index
```

### As a module

```python
from retriever import Retriever

r = Retriever()                          # loads from ./rag_index by default
r = Retriever(index_dir="my_index")      # custom path

results = r.query(
    "What is self-attention?",
    top_k=5,
    filter_h1="Transformers",            # optional section filter
    filter_source="lecture_notes",       # optional file filter
    min_score=0.4,                       # optional similarity floor
)

for chunk in results:
    print(f"{chunk['score']:.3f}  {chunk['breadcrumb']}")
    print(chunk["text"])
    print()
```

Each result is a dict:

```python
{
    "score":      float,   # cosine similarity 0–1, higher = more relevant
    "text":       str,     # chunk content (breadcrumb prepended)
    "source":     str,     # original file path
    "metadata":   dict,    # h1/h2/h3/h4/breadcrumb
    "breadcrumb": str,     # e.g. "Transformers > Self-Attention"
}
```

---

## GPU notes

All three scripts share a common `detect_device()` priority:

```text
CUDA  →  CPU
```

`chunker.py` is CPU-only (text splitting — no tensors involved).
`embedder.py` and `retriever.py` both auto-detect and use CUDA if available.
Pass `--device cpu` to force CPU even when a GPU is present.

---

## Typical full run

```bash
# 1. Chunk
python chunker.py --manifest sources.txt

# 2. Embed  (auto-detects GPU)
python embedder.py

# 3. Query
python retriever.py --query "Your question here" --top-k 5
```

Total time on CPU for ~500 chunks: under 2 minutes.
On a CUDA GPU the embedding step is typically 10–20× faster.
