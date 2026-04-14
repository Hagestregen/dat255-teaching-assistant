# DAT255 Teaching Assistant

A small decoder-only transformer + RAG pipeline that answers ML questions
and quizzes students on course material.

## File overview

```text
teaching_assistant/
├── model.py          # Transformer with RoPE — the core model
├── dataset.py        # Data loading, normalization, QA formatting
├── train.py          # Training loop (AdamW, cosine LR, gradient clipping)
├── rag_pipeline.py   # RAG: ties retriever → model for grounded answers
├── quiz.py           # Quiz generation, parsing, quality evaluation
├── evaluate.py       # Multi-metric evaluation (GPT-4 judge, BLEU, BERTScore)
├── app.py            # Gradio web UI
│
├── chunker.py        # (existing) markdown → overlapping chunks
├── embedder.py       # (existing) chunks → HNSW index
└── retriever.py      # (existing) query → top-k chunks
```

## Step-by-step order of operations

### Step 1: Install dependencies

```bash
pip install torch tiktoken datasets sentence-transformers hnswlib numpy openai gradio bert-score
```

### Step 2: Build the RAG index from your course material

```bash
# First chunk your PDFs/markdown (your existing chunker.py)
python chunker.py --manifest sources.txt --output chunks.json

# Then embed and build HNSW index (your existing embedder.py)
python embedder.py --chunks chunks.json --out-dir rag_index/
```

### Step 3: Generate synthetic QA pairs from course material

```python
from dataset import generate_synthetic_qa

generate_synthetic_qa(
    chunks_json_path="chunks.json",
    output_jsonl_path="data/synthetic_qa.jsonl",
    api_key="your-openai-api-key",
    max_chunks=200,
    qa_per_chunk=3,
)
# Result: ~600 DAT255-specific QA pairs saved to data/synthetic_qa.jsonl
```

### Step 4: Train the model

```bash
python train.py
# Trains for 5000 steps by default
# Checkpoints saved to checkpoints/ every 500 steps
# Monitor loss — should drop from ~10 to ~2-4
```

### Step 5: Evaluate (compare Model A, B, C)

```python
from dataset import build_datasets, Tokenizer
from rag_pipeline import RAGPipeline
from evaluate import run_evaluation

_, _, test_ds = build_datasets()
rag = RAGPipeline("checkpoints/step_005000.pt", "rag_index/")

test_examples = [{"question": item["question"], "answer": item["answer"]}
                 for item in test_ds.data[:100]]

run_evaluation(test_examples, rag, openai_api_key="...", output_path="results.json")
```

### Step 6: Run the web app

```bash
python app.py --checkpoint checkpoints/step_005000.pt --index rag_index/
# Opens at http://localhost:7860
# Add --share for a public URL (useful for demo/report)
```

## Key design decisions

**Why RoPE?**
RoPE rotates Q/K vectors in attention based on position, encoding relative
distance instead of absolute position. No extra parameters, generalizes better
to longer contexts at inference — important when RAG prepends retrieved text.

**Why train on HF datasets but RAG from course material?**
Your small model (~6-50M params) can't memorize a full curriculum. By
fine-tuning on generic ML QA data, it learns *how* to answer questions.
By using RAG at inference, it can *look up* course-specific facts each time.
This is like an open-book exam: you need to know how to reason, not memorize.

**Why loss mask on the question tokens?**
We only compute loss on the answer tokens. The model was given the question —
it shouldn't waste gradient steps re-learning to predict it. This improves
training efficiency and answer quality.

**Why weight tying?**
The token embedding matrix (lookup) and LM head (output projection) share
the same weights. Reduces ~25M parameters. The intuition: the same vector
that represents a token when reading it should work when generating it.

## Expected training curve

```text
step 0:    loss ~10.8 (random, ~ln(50257))
step 100:  loss ~6-7
step 500:  loss ~4-5
step 1000: loss ~3-4
step 5000: loss ~2-3
```

If loss doesn't decrease past step 100, check your learning rate and
data format (especially the -1 masking on question tokens).

## Scaling guide

| Setting      | n_layer | n_embd | Params | Hardware   | Time      |
|--------------|---------|--------|--------|------------|-----------|
| Tiny (test)  | 4       | 256    | ~8M    | CPU/laptop | 30 min    |
| Small        | 6       | 512    | ~50M   | GPU (4GB)  | 2-3 hours |
| Medium       | 8       | 768    | ~125M  | GPU (8GB)  | 6-8 hours |
