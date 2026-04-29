"""
dataset.py  —  Data pipeline for training and inference
========================================================
Provides the three symbols consumed by train.py:

    from dataset import Tokenizer, build_datasets, make_dataloader

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

  3. LOCALLY GENERATED DATA (from data_generation.py) → enriches fine-tuning
     Run data_generation.py once to produce a JSONL file with explanation, review,
     quiz, and QA examples generated from your course material.  Pass its path as
     synthetic_jsonl_path to build_datasets().

Why compute loss only on the Answer part?
─────────────────────────────────────────
During training we use teacher forcing: we feed in the full string and predict
the next token at every position.  But we only want the model to improve at
generating *answers*, not at re-generating the question it was already given.

We achieve this by setting the target token IDs to -1 for every position that
is part of the prompt (the "Question: ..." prefix).  PyTorch's cross_entropy
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



TASK_TOKENS = {
    "<|explain|>":   50257,
    "<|quiz|>":      50258,
    "<|review|>":    50259,
    "<|flashcard|>": 50260,
}

TASK_TYPE_TO_TOKEN = {
    "explanation": "<|explain|>",
    "quiz":        "<|quiz|>",
    "review":      "<|review|>",
    "flashcard":   "<|flashcard|>",
}

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
        base = tiktoken.get_encoding("gpt2")
        self.enc = tiktoken.Encoding(
            name="gpt2_teaching",
            pat_str=base._pat_str,
            mergeable_ranks=base._mergeable_ranks,
            special_tokens={
                **base._special_tokens,   # preserves <|endoftext|> = 50256
                **TASK_TOKENS,
            },
        )
        self.eot_id = 50256
        self.pad_id = self.eot_id

    def encode(self, text: str) -> List[int]:
        return self.enc.encode(
            text,
            allowed_special={"<|endoftext|>"} | set(TASK_TOKENS.keys()),
        )

    def decode(self, ids: List[int]) -> str:
        return self.enc.decode(ids)

    @property
    def vocab_size(self) -> int:
        return 50_261   # 50257 + 4 task tokens


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: Data normalization
# ─────────────────────────────────────────────────────────────────────────────

def normalize_example(example: Dict) -> Optional[Dict]:
    """
    Convert a raw example from ANY source into our unified schema:
      {"question": str, "answer": str, "source": str}

    Handles:
      - HuggingFace datasets (prsdm, win-wang, SQuAD-style)
      - Examples generated locally by data_generation.py
        (types: "explanation", "review", "flashcard", "quiz")

    Returns None if the example is unusable (empty fields, too short, etc.)
    """
    q = a = None

    ex_type = example.get("type", "")

    # ── locally-generated formats (from data_generation.py) ──────────────────

    if ex_type == "explanation":
        q = str(example.get("request", "")).strip()
        a = str(example.get("explanation", "")).strip()

    elif ex_type == "review":
        # Combine question + student answer into the "question" side so the
        # model learns to produce a review given both.
        question     = str(example.get("question",       "")).strip()
        student_ans  = str(example.get("student_answer", "")).strip()
        q = f"{question}\nStudent answer: {student_ans}"
        a = str(example.get("review", "")).strip()
    
    elif ex_type == "flashcard":
        q = str(example.get("front", "")).strip()
        a = str(example.get("back",  "")).strip()

    elif ex_type == "quiz":
        quiz = example.get("quiz", {})
        question = str(quiz.get("question", "")).strip()
        options  = quiz.get("options", [])
        # Format options as A/B/C/D so the model learns the MCQ layout
        lettered = "\n".join(f"{chr(65+i)}) {opt}" for i, opt in enumerate(options))
        q = f"{question}\n{lettered}"
        # Answer = correct letter + explanation
        correct_i   = quiz.get("correct_index", 0)
        correct_opt = options[correct_i] if options else ""
        explanation = str(quiz.get("explanation", "")).strip()
        a = f"{chr(65 + correct_i)}) {correct_opt}. {explanation}"

    # ── HuggingFace dataset formats ───────────────────────────────────────────

    elif "question" in example and "answer" in example:
        # prsdm/Machine-Learning-QA-dataset schema
        q = str(example.get("question", "")).strip()
        a = str(example.get("answer",   "")).strip()

    elif "Question" in example and "Answer" in example:
        # win-wang/Machine_Learning_QA_Collection schema
        q = str(example.get("Question", "")).strip()
        a = str(example.get("Answer",   "")).strip()

    elif "question" in example and "answers" in example:
        # SQuAD-style (has 'answers' as a dict with 'text' list)
        q = str(example.get("question", "")).strip()
        answers = example.get("answers", {})
        texts   = answers.get("text", [])
        a = str(texts[0]).strip() if texts else ""

    if not q or not a or len(q) < 10 or len(a) < 5:
        return None

    # Preserve retrieved/source context if provided.  data_generation.py emits
    # a "context" field on its results (Phase 3); HuggingFace examples don't.
    # When present, dataset training can randomly include it as a "Context:"
    # block to teach the model the RAG-augmented format too.
    ctx = str(example.get("context", "")).strip()

    out = {
        "question": q,
        "answer":   a,
        "source":   example.get("source", "unknown"),
        "context":  ctx,
        "type":     ex_type or "explanation",
    }
    # Propagate chunk_hash when the upstream emitter (data_generation.py)
    # provides one.  build_splits.py uses it to keep all examples produced
    # from the same lecture chunk in the SAME split — without this the
    # chunk-level leakage check is silently a no-op.
    if "chunk_hash" in example:
        out["chunk_hash"] = example["chunk_hash"]
    return out


def format_for_training(question: str, answer: str, context: str = "", task_type: str = "explanation") -> str:
    """
    Format a QA pair as an instruction-following string.

    The exact same format is used at inference time, so the model can
    complete "Question: ...\nAnswer:" with a good answer.  We append
    <|endoftext|> so the model learns when to STOP generating — without
    it, the model would generate forever.

    If `context` is provided (Phase 3), prepend it as a "Context:" block
    so the model also learns the RAG-augmented format.  Train-time mixing
    of the two formats teaches the model to handle both modes.
    """
    task_tok = TASK_TYPE_TO_TOKEN.get(task_type, "<|explain|>")
    if context:
        return (f"{task_tok} Context: {context}\n\nQuestion: {question}\n"
                f"Answer: {answer}<|endoftext|>")
    return f"{task_tok} Question: {question}\nAnswer: {answer}<|endoftext|>"


def format_rag_prompt(context: str, question: str) -> str:
    """
    Format a prompt that includes RAG-retrieved context.

    At inference with RAG, we prepend the retrieved chunks.  The model
    has NOT seen this exact format during training (unless you create synthetic
    RAG examples), but because it's trained to "Answer:" completions and the
    context clearly contains the answer, it will use it.

    For best results: also fine-tune on some context-question-answer triples
    where the answer is extractable from the context.  data_generation.py
    already emits examples with a "Context:" prefix — those train this
    naturally.
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
    it was already GIVEN the question.  We only want it to learn to predict the
    answer.  This is called "loss masking" and is standard for instruction tuning.
    """

    def __init__(
        self,
        examples: List[Dict],
        tokenizer: Tokenizer,
        max_length: int = 512,
        rag_context_prob: float = 0.0,    # Phase 3: P(include Context: prefix at train time)
    ):
        self.tokenizer        = tokenizer
        self.max_length       = max_length
        self.rag_context_prob = rag_context_prob
        self.data             = []
        self.raw_examples     = []  # raw normalized dicts (kept for eval/test-split persistence)

        skipped = 0
        for ex in examples:
            item = self._process(ex)
            if item is not None:
                self.data.append(item)
                self.raw_examples.append(ex)
            else:
                skipped += 1

        print(f"Dataset: {len(self.data)} examples loaded, {skipped} skipped (too long).")

    def _process(self, example: Dict) -> Optional[Dict]:
        q   = example["question"]
        a   = example["answer"]
        ctx = (example.get("context") or "").strip()
        task_type = example.get("type", "explanation")          # add
        task_tok  = TASK_TYPE_TO_TOKEN.get(task_type, "<|explain|>")  # add

        # Phase 3: with probability rag_context_prob and IF a context exists,
        # train on the RAG-augmented format ("Context: ...\nQuestion: ...").
        # Otherwise train on the bare format.  The model learns both modes.
        # Note: we use random.random() so each *epoch* re-rolls — this would
        # require a custom shuffler.  At dataset-build time we deterministically
        # decide once per example to keep training reproducible.
        include_context = bool(ctx) and (random.random() < self.rag_context_prob)

        full_text = format_for_training(q, a, context=ctx if include_context else "",
                                    task_type=task_type)

        full_ids  = self.tokenizer.encode(full_text)

        if len(full_ids) > self.max_length:
            return None

        prompt_text = (
            f"{task_tok} Context: {ctx}\n\nQuestion: {q}\nAnswer:"   # task_tok added
            if include_context
            else f"{task_tok} Question: {q}\nAnswer:"                 # task_tok added
        )
        prompt_ids = self.tokenizer.encode(prompt_text)
        prompt_len = len(prompt_ids)

        labels    = [-1] * prompt_len + full_ids[prompt_len:]
        pad_len   = self.max_length - len(full_ids)
        input_ids = full_ids + [self.tokenizer.pad_id] * pad_len
        labels    = labels   + [-1] * pad_len

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
        from datasets import load_dataset
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
# SECTION 5: Loading locally generated data (from data_generation.py)
# ─────────────────────────────────────────────────────────────────────────────

def load_local_generated_data(jsonl_path: str) -> List[Dict]:
    """
    Load training examples produced by data_generation.py.

    The JSONL file contains entries with types: "qa", "explanation",
    "review", "quiz_gen".  Each is normalized into {"question", "answer",
    "source"} so QADataset can process them uniformly.

    Run data_generation.py once to build this file — you don't need to
    regenerate it every training run.

    Example:
        python data_generation.py --chunks ../rag/rag_index/chunks.json \\
            --output data/train.jsonl --model Qwen/Qwen2.5-7B-Instruct
    """
    results = []
    path = Path(jsonl_path)
    if not path.exists():
        print(f"  [dataset] {jsonl_path} not found — skipping local data.")
        return results

    by_type: dict = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw  = json.loads(line)
            norm = normalize_example({**raw, "source": raw.get("source", "local_generated")})
            if norm:
                results.append(norm)
                by_type[raw.get("type", "unknown")] = by_type.get(raw.get("type", "unknown"), 0) + 1

    print(f"Loaded {len(results)} local generated examples from {jsonl_path}")
    for t, n in sorted(by_type.items()):
        print(f"  {t:12s}  {n:5d}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: Assemble and split the full dataset
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_SPLITS_DIR = "model/data/splits"


def _load_split_jsonl(path: Path) -> List[Dict]:
    """Read a frozen split file produced by build_splits.py."""
    examples: List[Dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            # Defensive: re-normalise so this still works if someone hand-edits
            # the split files or if the schema changes between runs.
            norm = normalize_example(ex) if "question" not in ex or "answer" not in ex else ex
            if norm:
                examples.append(norm)
    return examples


def build_datasets(
    synthetic_jsonl_path: str = None,
    chunks_json_path: str = None,   # kept for API compatibility, not used directly
    tokenizer: Tokenizer = None,
    max_length: int = 512,
    val_frac: float = 0.05,
    test_frac: float = 0.1,
    seed: int = 42,
    rag_context_prob: float = 0.0,    # Phase 3: P(include retrieved Context)
    max_train_examples: Optional[int] = None,  # for data-size ablations
    splits_dir: Optional[str] = None,  # frozen splits from build_splits.py
) -> tuple:
    """
    Assemble train/val/test datasets.

    There are two modes:

    1. **Frozen splits (recommended)** — pass `splits_dir` (or rely on the
       default `model/data/splits`).  We load `train.jsonl`, `val.jsonl`,
       `test.jsonl` from that directory and the manifest tells us they were
       split once with full stratification + chunk-level dedup.  Every
       training run reads the same examples, so report tables are clean.

    2. **Legacy on-the-fly split** — used only when `splits_dir` is missing.
       We load HuggingFace + locally-generated data and split here.  This
       path exists for backward compatibility with older scripts but is no
       longer the default; you should run `python model/build_splits.py`
       once and use the frozen files instead.

    Returns: (train_dataset, val_dataset, test_dataset)
    """
    if tokenizer is None:
        tokenizer = Tokenizer()

    # ── Mode 1: load frozen splits if available ──────────────────────────────
    # Try cwd-relative first (works when train.py runs from `model/`),
    # then fall back to project-root-relative (works when train.py runs from
    # the repo root or when splits_dir is given as an absolute path).
    project_root = Path(__file__).resolve().parent.parent
    raw_path     = Path(splits_dir or DEFAULT_SPLITS_DIR)
    candidates: list[Path] = []
    if raw_path.is_absolute():
        candidates.append(raw_path)
    else:
        candidates.append(Path.cwd() / raw_path)            # cwd-relative
        candidates.append(project_root / raw_path)          # project-root
        candidates.append(project_root / "model" / raw_path) # explicit model/
    candidate = next(
        (c for c in candidates
         if c.exists() and (c / "manifest.json").exists()),
        None,
    )
    use_frozen = candidate is not None

    if use_frozen:
        print(f"Loading frozen splits from {candidate}")
        train_examples = _load_split_jsonl(candidate / "train.jsonl")
        val_examples   = _load_split_jsonl(candidate / "val.jsonl")
        test_examples  = _load_split_jsonl(candidate / "test.jsonl")

    else:
        # ── Mode 2: legacy on-the-fly split (deprecated) ─────────────────────
        print(f"  [build_datasets] frozen splits not found at {candidate}; "
              "falling back to on-the-fly split (NOT recommended for the report)")
        random.seed(seed)
        general_examples = load_huggingface_datasets()
        course_examples  = []

        if synthetic_jsonl_path:
            course_examples = load_local_generated_data(synthetic_jsonl_path)
        elif chunks_json_path:
            print("No local generated data found. Run data_generation.py first.")

        random.shuffle(general_examples)
        random.shuffle(course_examples)

        def _split(examples: List[Dict]) -> tuple:
            n_val_  = int(len(examples) * val_frac)
            n_test_ = int(len(examples) * test_frac)
            test_   = examples[:n_test_]
            val_    = examples[n_test_: n_test_ + n_val_]
            train_  = examples[n_test_ + n_val_:]
            return train_, val_, test_

        g_train, g_val, g_test = _split(general_examples)
        c_train, c_val, c_test = _split(course_examples)

        train_examples = g_train + c_train
        val_examples   = g_val   + c_val
        test_examples  = g_test  + c_test
        random.shuffle(train_examples)

    # ── Optional data-size ablation (applies in both modes) ──────────────────
    # We deterministically take the FIRST `max_train_examples` items so two
    # runs with the same cap see exactly the same training subset.  In the
    # frozen-splits mode this means you also need to choose a deterministic
    # ordering when build_splits.py shuffles train — which it does using a
    # fixed seed, so this is reproducible.
    if max_train_examples is not None and max_train_examples > 0:
        before = len(train_examples)
        train_examples = train_examples[:max_train_examples]
        print(f"  [build_datasets] capped train from {before} → "
              f"{len(train_examples)} (max_train_examples={max_train_examples})")

    # ── Print breakdowns so the run log is self-describing ───────────────────
    def _by_type(xs: List[Dict]) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for x in xs:
            t = x.get("type", "explanation")
            out[t] = out.get(t, 0) + 1
        return out

    print("\nDataset split (after any cap):")
    for name, xs in (("Train", train_examples), ("Val", val_examples), ("Test", test_examples)):
        types = _by_type(xs)
        types_str = ", ".join(f"{k}={v}" for k, v in sorted(types.items()))
        print(f"  {name:5s}: {len(xs):5d} examples  [{types_str}]")

    train_ds = QADataset(train_examples, tokenizer, max_length,
                         rag_context_prob=rag_context_prob)
    val_ds   = QADataset(val_examples,   tokenizer, max_length,
                         rag_context_prob=0.0)   # eval is deterministic
    test_ds  = QADataset(test_examples,  tokenizer, max_length,
                         rag_context_prob=0.0)

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
# SECTION 7: Sequence packing (Phase 2)
# ─────────────────────────────────────────────────────────────────────────────

class PackedQADataset(Dataset):
    """
    Pack many short (question, answer) examples back-to-back into fixed-length
    windows so we don't waste compute on padding.

    Why does this matter?
    ─────────────────────
    QADataset right-pads every example to max_length=512 tokens.  For typical
    QA pairs the actual length is 60–120 tokens — meaning ≥75 % of every batch
    is wasted on PAD tokens (which have label=-1, so they do nothing for the
    loss).  Packing converts that wasted compute back into useful gradient
    signal: 2–3 × effective throughput on this dataset.

    The packing scheme here:
      ┌──────────── pack 1 (length max_length) ──────────────┐
      │ Q1 ... A1<eot>  Q2 ... A2<eot>  Q3 ... A3<eot>  pad   │
      │  [ -1 ][answer1][ -1 ][answer2][ -1 ][answer3][ -1 ]  │   ← labels
      └────────────────────────────────────────────────────────┘

    What ABOUT cross-example attention?
    ───────────────────────────────────
    With pure causal attention, tokens of example 2 *can* attend to tokens of
    example 1.  The model learns that "<|endoftext|>" resets context — this is
    exactly how nanoGPT-style pretraining works and it works fine in practice.
    Building a true block-diagonal attention mask is possible but disables the
    Flash-Attention fast path; we keep things simple here.

    `rag_context_prob` lets us mix bare and Context-prefixed prompts during
    packing, identical to QADataset's behavior (Phase 3).
    """

    def __init__(
        self,
        examples: List[Dict],
        tokenizer: Tokenizer,
        max_length: int = 512,
        rag_context_prob: float = 0.0,
        seed: int = 42,
    ):
        self.tokenizer        = tokenizer
        self.max_length       = max_length
        self.rag_context_prob = rag_context_prob
        self.raw_examples     = []
        self.packs: List[Dict[str, torch.Tensor]] = []

        rng = random.Random(seed)
        eot = tokenizer.eot_id
        pad = tokenizer.pad_id

        # 1. Tokenize each example into (input_ids, labels)
        tokenized: List[tuple[List[int], List[int]]] = []
        skipped = 0
        for ex in examples:
            tup = self._tokenize_example(ex, rng)
            if tup is None:
                skipped += 1
                continue
            ids, labels = tup
            if len(ids) > max_length:
                # Single example longer than the window; skip rather than truncate
                # (truncation kills the loss-mask alignment).
                skipped += 1
                continue
            tokenized.append((ids, labels))
            self.raw_examples.append(ex)

        # 2. Shuffle so packs don't reflect dataset order
        order = list(range(len(tokenized)))
        rng.shuffle(order)

        # 3. Greedily fill packs of max_length tokens.
        cur_ids:    list[int] = []
        cur_labels: list[int] = []

        def flush():
            n = len(cur_ids)
            if n == 0:
                return
            pad_n = max_length - n
            ids    = cur_ids    + [pad] * pad_n
            labels = cur_labels + [-1]  * pad_n
            self.packs.append({
                "input_ids": torch.tensor(ids,    dtype=torch.long),
                "labels":    torch.tensor(labels, dtype=torch.long),
            })

        for idx in order:
            ids, labels = tokenized[idx]
            if len(cur_ids) + len(ids) > max_length:
                flush()
                cur_ids, cur_labels = [], []
            cur_ids.extend(ids)
            cur_labels.extend(labels)
        flush()

        used_tokens = sum(int((p["labels"] != -1).sum().item()) for p in self.packs)
        total_tokens = len(self.packs) * max_length
        density = used_tokens / max(total_tokens, 1)
        print(f"PackedQADataset: {len(self.packs)} packs of length {max_length} "
              f"from {len(tokenized)} examples ({skipped} skipped). "
              f"Loss-token density: {density:.1%}")

    def _tokenize_example(
        self, ex: Dict, rng: random.Random
    ) -> Optional[tuple[List[int], List[int]]]:
        """
        Return (input_ids, labels) for one example.  labels has -1 on prompt
        and pad positions, real ids on answer-span positions.
        """
        q   = ex["question"]
        a   = ex["answer"]
        ctx = (ex.get("context") or "").strip()
        task_type = ex.get("type", "explanation")                    
        task_tok  = TASK_TYPE_TO_TOKEN.get(task_type, "<|explain|>")

        # Decide whether to include the context this iteration (Phase 3).
        include_context = bool(ctx) and rng.random() < self.rag_context_prob

        full_text   = format_for_training(q, a, context=ctx if include_context else "", task_type=task_type)
        prompt_text = (
            f"{task_tok} Context: {ctx}\n\nQuestion: {q}\nAnswer:"   # task_tok added
            if include_context
            else f"{task_tok} Question: {q}\nAnswer:"                 # task_tok added
        )
        full_ids   = self.tokenizer.encode(full_text)
        prompt_ids = self.tokenizer.encode(prompt_text)
        prompt_len = len(prompt_ids)

        if len(full_ids) <= prompt_len:
            return None

        labels = [-1] * prompt_len + full_ids[prompt_len:]
        return full_ids, labels

    def __len__(self):
        return len(self.packs)

    def __getitem__(self, idx):
        return self.packs[idx]


def make_packed_dataloader(dataset: PackedQADataset, batch_size: int, shuffle: bool = True) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
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
    text = format_for_training(sample["question"], sample["answer"], task_type="explanation")
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