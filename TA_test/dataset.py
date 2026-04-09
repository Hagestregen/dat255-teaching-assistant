"""
dataset.py  —  Data pipeline: HuggingFace QA datasets → instruction-tuning format
===================================================================================
KEY CONCEPT: How do you teach a model to answer questions?
──────────────────────────────────────────────────────────
Your curriculum PDFs are PLAIN TEXT. They're not in question-answer format.
You have two separate roles for data:

  1. QA DATASETS (HuggingFace) → used for FINE-TUNING the model
     The model reads thousands of (question, answer) pairs formatted as:
       "Question: What is dropout?\nAnswer: Dropout is a regularization...<|endoftext|>"
     It learns the PATTERN: "when I see Question: ..., I should generate a good Answer:"
     This is called INSTRUCTION FINE-TUNING.

  2. CURRICULUM TEXT (your PDFs) → used for RAG RETRIEVAL ONLY
     These are chunked, embedded, and stored in a vector index (your chunker/embedder).
     At inference, relevant chunks are retrieved and prepended to the prompt as context.
     The model DOES NOT need to memorize the lectures — it reads them at inference time.

So the training only needs to happen once on generic ML QA data.
The curriculum knowledge is always available via RAG.

Why compute loss only on the Answer part?
─────────────────────────────────────────
During training we use teacher forcing: we feed in the full string and predict
the next token at every position. But we only want the model to improve at
generating *answers*, not at re-generating the question it was already given.

We achieve this by setting the target token IDs to -1 for every position that
is part of the prompt (the "Question: ..." prefix). PyTorch's cross_entropy
with ignore_index=-1 simply doesn't count those positions in the loss.

  Input:   "Question: What is X?\nAnswer: X is Y<|endoftext|>"
  Targets: [-1, -1, -1, ... , id("X"), id("is"), id("Y"), id("<|endoftext|>")]
            ^---- all -1 ----^  ^----------- loss computed here ─────────────^
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Optional

import torch
from torch.utils.data import Dataset, DataLoader


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: Tokenizer Wrapper
# ─────────────────────────────────────────────────────────────────────────────

class Tokenizer:
    """
    Thin wrapper around tiktoken's GPT-2 tokenizer.

    We use tiktoken (OpenAI's tokenizer library) because:
    - Same BPE vocabulary as GPT-2: 50,257 tokens
    - Very fast (Rust-backed)
    - <|endoftext|> token (id 50256) serves as our end-of-sequence marker

    The tokenizer is SEPARATE from the model's embedding layer. The
    model only sees integer IDs — the tokenizer handles text ↔ integers.

    NOTE on the RAG embedder mismatch:
    Your retriever.py uses sentence-transformers for EMBEDDING (meaning search).
    That model uses a completely different tokenizer internally. This is fine!
    The two tokenizers serve different purposes:
      - sentence-transformers tokenizer: chunking + semantic vectors for search
      - tiktoken (here): input to YOUR trained transformer for generation
    They never need to match. Don't confuse them.
    """

    def __init__(self):
        import tiktoken
        self.enc = tiktoken.get_encoding("gpt2")
        self.eot_id = self.enc.eot_token   # 50256 = <|endoftext|>
        # We'll use this as our padding token too (with loss masking)
        self.pad_id = self.eot_id

    def encode(self, text: str) -> List[int]:
        return self.enc.encode(text, allowed_special={"<|endoftext|>"})

    def decode(self, ids: List[int]) -> str:
        return self.enc.decode(ids)

    @property
    def vocab_size(self) -> int:
        return self.enc.n_vocab  # 50257


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: Data normalization
# ─────────────────────────────────────────────────────────────────────────────

def normalize_example(example: Dict) -> Optional[Dict]:
    """
    Convert a raw example from ANY source into our unified schema:
      {"question": str, "answer": str, "source": str}

    Different HuggingFace datasets have different field names. We handle
    the two you're using and your synthetic data here.

    Returns None if the example is unusable (empty fields, too short, etc.)
    """
    q = a = None

    # prsdm/Machine-Learning-QA-dataset schema
    if "question" in example and "answer" in example:
        q = str(example.get("question", "")).strip()
        a = str(example.get("answer", "")).strip()

    # win-wang/Machine_Learning_QA_Collection schema
    elif "Question" in example and "Answer" in example:
        q = str(example.get("Question", "")).strip()
        a = str(example.get("Answer", "")).strip()

    # SQuAD-style (has 'answers' as a dict with 'text' list)
    elif "question" in example and "answers" in example:
        q = str(example.get("question", "")).strip()
        answers = example.get("answers", {})
        texts = answers.get("text", [])
        a = str(texts[0]).strip() if texts else ""

    if not q or not a or len(q) < 10 or len(a) < 5:
        return None

    return {
        "question": q,
        "answer": a,
        "source": example.get("source", "unknown"),
    }


def format_for_training(question: str, answer: str) -> str:
    """
    Format a QA pair as an instruction-following string.

    This exact format is used at inference too — the model has been trained
    to complete "Question: ...\nAnswer:" with a good answer.

    We append <|endoftext|> so the model learns when to STOP generating.
    Without this, the model would generate forever.
    """
    return f"Question: {question}\nAnswer: {answer}<|endoftext|>"


def format_rag_prompt(context: str, question: str) -> str:
    """
    Format a prompt that includes RAG-retrieved context.

    At inference with RAG, we prepend the retrieved chunks. The model
    has NOT seen this exact format during training (unless you create synthetic
    RAG examples), but because it's trained to "Answer:" completions and the
    context clearly contains the answer, it will use it.

    For best results: also fine-tune on some context-question-answer triples
    where the answer is extractable from the context. See generate_synthetic_qa().
    """
    return f"Context: {context}\n\nQuestion: {question}\nAnswer:"


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: PyTorch Dataset
# ─────────────────────────────────────────────────────────────────────────────

class QADataset(Dataset):
    """
    PyTorch Dataset for instruction fine-tuning.

    Each item is a tokenized QA pair where:
    - input_ids: all tokens in the sequence
    - labels:    same as input_ids but with -1 for the question part (ignored in loss)

    Why is the label for the prompt -1?
    The model shouldn't waste capacity learning to predict the question tokens —
    it was already GIVEN the question. We only want it to learn to predict the
    answer. This is called "loss masking" and is standard for instruction tuning.
    """

    def __init__(
        self,
        examples: List[Dict],
        tokenizer: Tokenizer,
        max_length: int = 512,
    ):
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.data       = []

        skipped = 0
        for ex in examples:
            item = self._process(ex)
            if item is not None:
                self.data.append(item)
            else:
                skipped += 1

        print(f"Dataset: {len(self.data)} examples loaded, {skipped} skipped (too long).")

    def _process(self, example: Dict) -> Optional[Dict]:
        q = example["question"]
        a = example["answer"]

        # Tokenize the full string
        full_text = format_for_training(q, a)
        full_ids  = self.tokenizer.encode(full_text)

        if len(full_ids) > self.max_length:
            return None

        # Find where the answer starts in the token sequence.
        # We tokenize just the prompt part to find its length.
        prompt_text = f"Question: {q}\nAnswer:"
        prompt_ids  = self.tokenizer.encode(prompt_text)
        prompt_len  = len(prompt_ids)

        # Build labels: -1 for prompt tokens, actual ids for answer tokens
        labels = [-1] * prompt_len + full_ids[prompt_len:]

        # Pad both to max_length (so we can batch)
        pad_len   = self.max_length - len(full_ids)
        input_ids = full_ids + [self.tokenizer.pad_id] * pad_len
        labels    = labels   + [-1] * pad_len  # padding positions also ignored

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels":    torch.tensor(labels,    dtype=torch.long),
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: Loading HuggingFace datasets
# ─────────────────────────────────────────────────────────────────────────────

def load_huggingface_datasets(max_examples: int = 5000) -> List[Dict]:
    """
    Load and normalize the two HuggingFace ML QA datasets.

    These give the model general ML knowledge and, more importantly,
    teach it the PATTERN of answering questions in a coherent format.

    We limit to max_examples per dataset to keep training manageable.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Run: pip install datasets")

    examples = []

    print("Loading prsdm/Machine-Learning-QA-dataset...")
    try:
        ds1 = load_dataset("prsdm/Machine-Learning-QA-dataset", split="train")
        for ex in list(ds1)[:max_examples]:
            norm = normalize_example({**ex, "source": "prsdm_ml_qa"})
            if norm:
                examples.append(norm)
        print(f"  Loaded {len(examples)} from prsdm dataset")
    except Exception as e:
        print(f"  Warning: Could not load prsdm dataset: {e}")

    n_before = len(examples)
    print("Loading win-wang/Machine_Learning_QA_Collection...")
    try:
        ds2 = load_dataset("win-wang/Machine_Learning_QA_Collection", split="train")
        for ex in list(ds2)[:max_examples]:
            norm = normalize_example({**ex, "source": "winwang_ml_qa"})
            if norm:
                examples.append(norm)
        print(f"  Loaded {len(examples) - n_before} from win-wang dataset")
    except Exception as e:
        print(f"  Warning: Could not load win-wang dataset: {e}")

    print(f"Total examples from HuggingFace: {len(examples)}")
    return examples


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: Synthetic QA generation from course material
# ─────────────────────────────────────────────────────────────────────────────

def generate_synthetic_qa(
    chunks_json_path: str,
    output_jsonl_path: str,
    api_key: str,
    model: str = "gpt-4o-mini",  # cheap and fast for data generation
    max_chunks: int = 200,
    qa_per_chunk: int = 3,
):
    """
    Use GPT to generate QA pairs FROM YOUR COURSE MATERIAL.

    This is the bridge between your curriculum and the model's training data.
    For each chunk of lecture text, GPT generates realistic exam-style questions
    that a student might ask, along with answers grounded in that chunk.

    The result: your model has seen examples of answering DAT255-specific questions
    during training, not just generic ML questions.

    Run this ONCE and save the results — don't regenerate every training run.

    Output format (JSONL, one JSON per line):
      {"question": "...", "answer": "...", "source": "synthetic_dat255", "context": "..."}

    The "context" field lets you also train on context-grounded answers
    (for better RAG integration).
    """
    import openai
    import json

    client = openai.OpenAI(api_key=api_key)

    # Load the chunks produced by your chunker.py
    with open(chunks_json_path) as f:
        chunks = json.load(f)

    # Sample a subset (generating for all chunks can be expensive)
    selected = random.sample(chunks, min(max_chunks, len(chunks)))

    prompt_template = """
You are creating a study dataset for a machine learning course.
Given the following lecture text, generate {n} question-answer pairs.

Rules:
- Questions should be conceptual, not trivial (e.g., "What is X and why is it used?")
- Answers should be 2-4 sentences, grounded strictly in the provided text
- Format each pair EXACTLY as JSON: {{"question": "...", "answer": "..."}}
- Output a JSON array of {n} objects, nothing else

Lecture text:
{text}
"""

    results = []
    for i, chunk in enumerate(selected):
        text = chunk["text"][:800]  # truncate very long chunks
        prompt = prompt_template.format(n=qa_per_chunk, text=text)

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=800,
            )
            content = response.choices[0].message.content.strip()

            # Parse the JSON array
            qa_pairs = json.loads(content)
            for pair in qa_pairs:
                if "question" in pair and "answer" in pair:
                    results.append({
                        "question": pair["question"],
                        "answer":   pair["answer"],
                        "source":   "synthetic_dat255",
                        "context":  text,  # keep for RAG-style training
                    })
        except Exception as e:
            print(f"  Chunk {i}: failed ({e})")

        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(selected)} chunks, {len(results)} pairs so far")

    # Save as JSONL
    with open(output_jsonl_path, "w") as f:
        for item in results:
            f.write(json.dumps(item) + "\n")

    print(f"Saved {len(results)} synthetic QA pairs to {output_jsonl_path}")
    return results


def load_synthetic_qa(jsonl_path: str) -> List[Dict]:
    """Load previously generated synthetic QA pairs."""
    results = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    print(f"Loaded {len(results)} synthetic QA examples from {jsonl_path}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: Assemble and split the full dataset
# ─────────────────────────────────────────────────────────────────────────────

def build_datasets(
    chunks_json_path: str = None,
    synthetic_jsonl_path: str = None,
    tokenizer: Tokenizer = None,
    max_length: int = 512,
    val_frac: float = 0.05,
    test_frac: float = 0.1,
    seed: int = 42,
) -> tuple:
    """
    Assemble all data sources and split into train/val/test.

    IMPORTANT: We keep DAT255-specific (synthetic) data in TEST to measure
    how well the model actually performs on course material. This tests
    real-world utility, not just generic QA performance.

    Returns: (train_dataset, val_dataset, test_dataset)
    """
    random.seed(seed)
    if tokenizer is None:
        tokenizer = Tokenizer()

    # Load all sources
    general_examples  = load_huggingface_datasets()
    dat255_examples   = []

    if synthetic_jsonl_path and Path(synthetic_jsonl_path).exists():
        dat255_examples = load_synthetic_qa(synthetic_jsonl_path)
    elif chunks_json_path:
        print("No synthetic QA found. Run generate_synthetic_qa() first.")

    # Shuffle general examples
    random.shuffle(general_examples)

    # Split: dat255-specific → test (real-world evaluation)
    #        general → train/val
    n_val  = int(len(general_examples) * val_frac)
    n_test = int(len(general_examples) * test_frac)

    test_examples  = dat255_examples + general_examples[:n_test]
    val_examples   = general_examples[n_test: n_test + n_val]
    train_examples = general_examples[n_test + n_val:]

    print(f"\nDataset split:")
    print(f"  Train: {len(train_examples)} examples")
    print(f"  Val:   {len(val_examples)} examples")
    print(f"  Test:  {len(test_examples)} examples ({len(dat255_examples)} DAT255-specific)")

    train_ds = QADataset(train_examples, tokenizer, max_length)
    val_ds   = QADataset(val_examples,   tokenizer, max_length)
    test_ds  = QADataset(test_examples,  tokenizer, max_length)

    return train_ds, val_ds, test_ds


def make_dataloader(dataset: QADataset, batch_size: int, shuffle: bool = True) -> DataLoader:
    """Create a DataLoader from a QADataset."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,  # set to 4 if you have multiple CPU cores
        pin_memory=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tok = Tokenizer()
    print(f"Vocab size: {tok.vocab_size}")

    # Test formatting
    sample = {"question": "What is backpropagation?",
              "answer": "Backpropagation computes gradients by applying the chain rule."}
    text = format_for_training(sample["question"], sample["answer"])
    print(f"\nFormatted training example:\n{text}")

    ids = tok.encode(text)
    print(f"Token count: {len(ids)}")
    print(f"Decoded back: {tok.decode(ids)[:80]}...")

    # Test Dataset
    norm = normalize_example(sample)
    ds   = QADataset([norm], tok, max_length=128)
    item = ds[0]
    print(f"\ninput_ids shape: {item['input_ids'].shape}")
    print(f"labels shape:    {item['labels'].shape}")
    print(f"Non-masked labels: {(item['labels'] != -1).sum().item()} tokens")
