"""
progress_tracker.py  —  Track what the student has covered and how well
=========================================================================
This solves two problems:
  1. "Quiz me on Chapter 3" — how do we know what's in Chapter 3?
  2. "What have I already answered correctly?" — how do we avoid repetition
     and focus on weak spots?

DESIGN DECISIONS:
──────────────────
Why a JSON file, not a database?
  A SQLite database would also work, but JSON is simpler for a student project:
  human-readable, version-controllable, zero extra dependencies, trivially
  copyable. For a production system you'd use SQLite or Redis.

Why track at the CHUNK level, not the question level?
  Your curriculum is structured as chunks with h1/h2/h3 headers (from chunker.py).
  Tracking per chunk lets you answer "have we covered this section?" without
  relying on remembering specific questions. The chapter structure comes for
  free from the existing metadata.

HOW THE CURRICULUM MAP IS BUILT:
──────────────────────────────────
chunks.json already looks like:
  [
    {"text": "...", "metadata": {"h1": "Neural Networks", "h2": "Backpropagation", "breadcrumb": "..."}, ...},
    {"text": "...", "metadata": {"h1": "Neural Networks", "h2": "Activation Functions", ...}, ...},
    ...
  ]

We read this and build a nested dict:
  curriculum_map = {
    "Neural Networks": {
      "Backpropagation": [chunk_0, chunk_1, ...],
      "Activation Functions": [chunk_2, chunk_3, ...],
    },
    "Convolutional Networks": { ... }
  }

"Quiz me on backpropagation" → filter by h2 == "Backpropagation" → pick
unseen or low-score chunks → generate questions from those specific chunks.

SPACED REPETITION (SIMPLE VERSION):
─────────────────────────────────────
The SM-2 algorithm (used by Anki) is the gold standard but complex.
We use a simpler version: sort topics by a "priority score":
  priority = (1 - accuracy) + recency_bonus
where recency_bonus increases as time-since-last-seen grows (capped at 24h).

This naturally surfaces items you got wrong and items you haven't seen
recently. Good enough for a course project; cite SM-2 as future work.
"""

import json
import time
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ChunkRecord:
    """Tracks a student's interaction history with a single chunk."""
    chunk_id:     int
    topic:        str          # h2 header (or h1 if no h2)
    chapter:      str          # h1 header
    breadcrumb:   str          # full path e.g. "Neural Networks > Backprop"
    text_preview: str          # first 80 chars for display

    # Interaction history
    times_seen:    int   = 0
    times_correct: int   = 0
    last_seen:     float = 0.0   # unix timestamp

    @property
    def accuracy(self) -> float:
        if self.times_seen == 0:
            return 0.0
        return self.times_correct / self.times_seen

    @property
    def priority(self) -> float:
        """
        Higher = show sooner.

        Unseen items get 1.0 (always prioritise new material first).
        Seen items: (1 - accuracy) + recency_bonus.
        recency_bonus grows as the item ages, capping at 0.3 after 24 hours.
        This means a perfectly-correct item still resurfaces after a day.
        """
        if self.times_seen == 0:
            return 1.0

        hours_ago     = (time.time() - self.last_seen) / 3600
        recency_bonus = min(hours_ago / 24, 1.0) * 0.3

        return (1.0 - self.accuracy) + recency_bonus

    def record_answer(self, correct: bool):
        self.times_seen    += 1
        self.times_correct += (1 if correct else 0)
        self.last_seen      = time.time()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: Progress tracker
# ─────────────────────────────────────────────────────────────────────────────

class ProgressTracker:
    """
    Manages curriculum progress, topic selection, and chapter navigation.

    Usage:
        tracker = ProgressTracker("rag_index/chunks.json")
        chunk_id, record = tracker.next_chunk("Neural Networks")
        tracker.record(chunk_id, correct=True)
        tracker.stats()
    """

    def __init__(self, chunks_json_path: str, save_path: str = "progress.json"):
        self.chunks_path = chunks_json_path
        self.save_path   = save_path

        with open(chunks_json_path, encoding="utf-8") as f:
            self._raw_chunks = json.load(f)

        # Build records dict: chunk_id → ChunkRecord
        self.records: Dict[int, ChunkRecord] = {}
        for i, chunk in enumerate(self._raw_chunks):
            meta  = chunk.get("metadata", {})
            h1    = meta.get("h1") or "General"
            h2    = meta.get("h2") or h1
            crumb = meta.get("breadcrumb") or h1
            # Use the leaf (most specific) part of the breadcrumb as the topic.
            # e.g. "Book > Key concepts > Various AI approaches"
            #      → topic = "Various AI approaches"  (not the top-level book title)
            # Fall back to h2 if the breadcrumb has no ">" separator.
            leaf  = crumb.split(">")[-1].strip() if ">" in crumb else h2
            self.records[i] = ChunkRecord(
                chunk_id     = i,
                topic        = leaf,
                chapter      = h1,
                breadcrumb   = crumb,
                text_preview = chunk.get("text", "")[:80].replace("\n", " "),
            )

        # Build curriculum map: h1 → {leaf_topic → [chunk_ids]}
        self.curriculum_map: Dict[str, Dict[str, List[int]]] = {}
        for chunk_id, rec in self.records.items():
            self.curriculum_map.setdefault(rec.chapter, {}).setdefault(rec.topic, []).append(chunk_id)

        self._load()
        print(f"Curriculum: {len(self.records)} chunks across "
              f"{len(self.curriculum_map)} chapters")

    # ── Navigation ────────────────────────────────────────────────────────────

    def chapters(self) -> List[str]:
        """List all available chapters (h1 headers), sorted."""
        return sorted(self.curriculum_map.keys())

    def topics(self, chapter: str = None) -> List[str]:
        """List topics, optionally filtered to a chapter."""
        if chapter:
            return sorted(self.curriculum_map.get(chapter, {}).keys())
        all_topics: set = set()
        for topics in self.curriculum_map.values():
            all_topics.update(topics.keys())
        return sorted(all_topics)

    def next_chunk(
        self,
        chapter: str = None,
        topic:   str = None,
    ) -> Optional[Tuple[int, ChunkRecord]]:
        """
        Return the highest-priority chunk to study.

        Selection order:
          1. Unseen chunks (times_seen == 0) — always shown first
          2. Seen chunks ranked by priority score (low accuracy + recency)

        Filtered by chapter and/or topic if given.
        Returns (chunk_id, ChunkRecord) or None.
        """
        if topic:
            candidates = []
            for ch_topics in self.curriculum_map.values():
                for t, ids in ch_topics.items():
                    if t.lower() == topic.lower():
                        candidates.extend(ids)
        elif chapter:
            candidates = [cid for ids in self.curriculum_map.get(chapter, {}).values()
                          for cid in ids]
        else:
            candidates = list(self.records.keys())

        if not candidates:
            return None

        sorted_ids = sorted(candidates,
                            key=lambda i: self.records[i].priority,
                            reverse=True)

        # Prefer completely unseen chunks
        for cid in sorted_ids:
            if self.records[cid].times_seen == 0:
                return cid, self.records[cid]

        return sorted_ids[0], self.records[sorted_ids[0]]

    def chunk_text(self, chunk_id: int) -> str:
        """Return the full text of a chunk by its id."""
        return self._raw_chunks[chunk_id].get("text", "")

    def chunks_for_topic(self, topic: str) -> List[Tuple[int, ChunkRecord]]:
        """Return all chunks for a topic, sorted by priority (highest first)."""
        result = []
        for ch_topics in self.curriculum_map.values():
            for t, ids in ch_topics.items():
                if t.lower() == topic.lower():
                    result.extend([(cid, self.records[cid]) for cid in ids])
        return sorted(result, key=lambda x: x[1].priority, reverse=True)

    # ── Recording progress ────────────────────────────────────────────────────

    def record(self, chunk_id: int, correct: bool):
        """Record the result of an interaction on a specific chunk."""
        if chunk_id in self.records:
            self.records[chunk_id].record_answer(correct)
            self._save()

    # ── Stats and reporting ───────────────────────────────────────────────────

    def stats(self) -> dict:
        """Return overall progress statistics."""
        total    = len(self.records)
        seen     = sum(1 for r in self.records.values() if r.times_seen > 0)
        correct  = sum(r.times_correct for r in self.records.values())
        answered = sum(r.times_seen    for r in self.records.values())

        chapter_stats = {}
        for chapter, topics in self.curriculum_map.items():
            all_ids    = [cid for ids in topics.values() for cid in ids]
            seen_ids   = [cid for cid in all_ids if self.records[cid].times_seen > 0]
            correct_c  = sum(self.records[cid].times_correct for cid in all_ids)
            answered_c = sum(self.records[cid].times_seen    for cid in all_ids)
            chapter_stats[chapter] = {
                "total_chunks": len(all_ids),
                "seen_chunks":  len(seen_ids),
                "accuracy":     correct_c / max(answered_c, 1),
                "pct_complete": len(seen_ids) / len(all_ids) * 100,
            }

        return {
            "total_chunks":     total,
            "seen_chunks":      seen,
            "pct_seen":         seen / total * 100,
            "overall_accuracy": correct / max(answered, 1),
            "chapters":         chapter_stats,
        }

    def weak_topics(self, n: int = 5) -> List[Tuple[str, float]]:
        """Return the n topics the student struggles with most (seen ones only)."""
        topic_scores: Dict[str, List[float]] = {}
        for rec in self.records.values():
            if rec.times_seen > 0:
                topic_scores.setdefault(rec.topic, []).append(rec.accuracy)
        avg = {t: sum(v) / len(v) for t, v in topic_scores.items()}
        return sorted(avg.items(), key=lambda x: x[1])[:n]

    def print_progress(self):
        """Print a readable progress report to stdout."""
        s = self.stats()
        print(f"\n{'='*50}")
        print(f"Progress: {s['seen_chunks']}/{s['total_chunks']} "
              f"({s['pct_seen']:.0f}% covered)")
        print(f"Overall accuracy: {s['overall_accuracy']:.0%}")
        print(f"\nChapter breakdown:")
        for ch, cs in s["chapters"].items():
            bar = "█" * int(cs["pct_complete"] / 5) + "░" * (20 - int(cs["pct_complete"] / 5))
            print(f"  {ch[:30]:<30} {bar} {cs['pct_complete']:.0f}% | acc {cs['accuracy']:.0%}")
        weak = self.weak_topics(3)
        if weak:
            print(f"\nWeakest topics:")
            for topic, acc in weak:
                print(f"  {topic[:35]:<35} {acc:.0%} accuracy")
        print("=" * 50)

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save(self):
        data = {str(k): asdict(v) for k, v in self.records.items()}
        with open(self.save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def _load(self):
        if not Path(self.save_path).exists():
            return
        try:
            with open(self.save_path, encoding="utf-8") as f:
                data = json.load(f)
            for k, v in data.items():
                cid = int(k)
                if cid in self.records:
                    self.records[cid].times_seen    = v.get("times_seen",    0)
                    self.records[cid].times_correct = v.get("times_correct", 0)
                    self.records[cid].last_seen     = v.get("last_seen",     0.0)
            print(f"Progress loaded from {self.save_path}")
        except Exception as e:
            print(f"  [tracker] Could not load progress: {e}")

    def reset(self, chapter: str = None):
        """Reset progress, optionally for a specific chapter only."""
        targets = (
            [cid for ids in self.curriculum_map.get(chapter, {}).values() for cid in ids]
            if chapter
            else list(self.records.keys())
        )
        for cid in targets:
            r = self.records[cid]
            r.times_seen = r.times_correct = 0
            r.last_seen  = 0.0
        self._save()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: HTML progress dashboard (for gr.HTML in Gradio)
# ─────────────────────────────────────────────────────────────────────────────

def progress_to_html(tracker: ProgressTracker) -> str:
    """
    Render a styled HTML progress dashboard for display in gr.HTML().

    Shows overall coverage, per-chapter progress bars, and weak topics.
    """
    s = tracker.stats()

    pct_seen = s["pct_seen"]
    acc      = s["overall_accuracy"] * 100
    seen     = s["seen_chunks"]
    total    = s["total_chunks"]

    # ── Overall bar ───────────────────────────────────────────────────────────
    overall_bar_fill  = f"{pct_seen:.1f}%"
    overall_acc_fill  = f"{acc:.1f}%"

    # ── Chapter rows ──────────────────────────────────────────────────────────
    chapter_rows = ""
    for ch, cs in sorted(s["chapters"].items(), key=lambda x: x[0]):
        pct  = cs["pct_complete"]
        a    = cs["accuracy"] * 100
        seen_ch = cs["seen_chunks"]
        tot_ch  = cs["total_chunks"]

        # Color: green if >70% covered, amber if >30%, grey otherwise
        bar_color = ("#2f855a" if pct >= 70
                     else "#b7791f" if pct >= 30
                     else "#4a5568")
        acc_color = ("#2f855a" if a >= 70 else "#c53030" if a < 40 else "#b7791f")

        chapter_rows += f"""
        <tr>
          <td style="padding:6px 12px;font-size:14px;max-width:200px;
                     white-space:nowrap;overflow:hidden;text-overflow:ellipsis;
                     color:#e2e8f0;">{ch}</td>
          <td style="padding:6px 12px;min-width:180px;">
            <div style="background:#2d3748;border-radius:6px;height:14px;overflow:hidden;">
              <div style="width:{pct:.1f}%;background:{bar_color};height:100%;
                          border-radius:6px;transition:width 0.3s;"></div>
            </div>
            <span style="font-size:11px;color:#a0aec0;">{seen_ch}/{tot_ch} chunks</span>
          </td>
          <td style="padding:6px 12px;font-size:13px;font-weight:600;
                     color:{bar_color};text-align:right;">{pct:.0f}%</td>
          <td style="padding:6px 12px;font-size:13px;font-weight:600;
                     color:{acc_color};text-align:right;">{a:.0f}% acc</td>
        </tr>"""

    # ── Weak topics ───────────────────────────────────────────────────────────
    weak = tracker.weak_topics(5)
    if weak:
        weak_items = "".join(
            f"""<li style="margin:4px 0;font-size:13px;color:#e2e8f0;">
                  <span style="color:#fc8181;font-weight:600;">{t}</span>
                  <span style="color:#718096;"> — {a*100:.0f}% accuracy</span>
                </li>"""
            for t, a in weak
        )
        weak_section = f"""
        <div style="margin-top:20px;">
          <div style="font-size:13px;font-weight:700;color:#a0aec0;
                      text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;">
            ⚠ Topics to revisit
          </div>
          <ul style="margin:0;padding-left:20px;">{weak_items}</ul>
        </div>"""
    else:
        weak_section = (
            '<p style="color:#718096;font-size:13px;margin-top:16px;">'
            'Answer some questions to see weak topic analysis.</p>'
        )

    return f"""
<div style="font-family:'Segoe UI',system-ui,sans-serif;
            background:#1a202c;border-radius:14px;padding:24px 28px;
            color:#e2e8f0;max-width:700px;margin:8px auto;">

  <!-- Header -->
  <div style="font-size:18px;font-weight:700;margin-bottom:18px;">
    📊 Curriculum Progress
  </div>

  <!-- Overall stats row -->
  <div style="display:flex;gap:16px;margin-bottom:20px;flex-wrap:wrap;">
    <div style="flex:1;background:#2d3748;border-radius:10px;padding:14px 18px;min-width:140px;">
      <div style="font-size:11px;color:#a0aec0;text-transform:uppercase;letter-spacing:1px;">
        Coverage
      </div>
      <div style="font-size:28px;font-weight:700;color:#63b3ed;margin:4px 0;">
        {pct_seen:.0f}%
      </div>
      <div style="background:#4a5568;border-radius:4px;height:8px;overflow:hidden;margin-top:6px;">
        <div style="width:{overall_bar_fill};background:#3182ce;height:100%;border-radius:4px;"></div>
      </div>
      <div style="font-size:11px;color:#718096;margin-top:4px;">{seen} / {total} chunks</div>
    </div>
    <div style="flex:1;background:#2d3748;border-radius:10px;padding:14px 18px;min-width:140px;">
      <div style="font-size:11px;color:#a0aec0;text-transform:uppercase;letter-spacing:1px;">
        Overall accuracy
      </div>
      <div style="font-size:28px;font-weight:700;
                  color:{'#68d391' if acc>=70 else '#fc8181' if acc<40 else '#f6ad55'};
                  margin:4px 0;">
        {overall_acc_fill}
      </div>
      <div style="font-size:11px;color:#718096;margin-top:4px;">
        {s['total_chunks'] - s['seen_chunks']} chunks not yet seen
      </div>
    </div>
  </div>

  <!-- Chapter table -->
  <div style="font-size:13px;font-weight:700;color:#a0aec0;
              text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;">
    Chapters
  </div>
  <table style="width:100%;border-collapse:collapse;">
    <tbody>{chapter_rows}</tbody>
  </table>

  {weak_section}
</div>
"""


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: Tracked quiz session
# ─────────────────────────────────────────────────────────────────────────────

class TrackedQuizSession:
    """
    Quiz session that integrates with the progress tracker.

    Replaces the simple random selection in quiz.py with priority-based
    chunk selection: unseen chunks first, then lowest-accuracy chunks,
    with a recency bonus so old correct answers resurface over time.

    Supports BOTH the pretrained HuggingFace pipeline AND the custom checkpoint.
    The tracker selects WHICH chunk to draw the question from; the existing
    generators handle the actual question generation.
    """

    def __init__(
        self,
        tracker: ProgressTracker,
        rag_pipeline=None,        # custom RAGPipeline (checkpoint mode)
        pretrained_pipe=None,     # HuggingFace text-generation pipeline
        retriever=None,           # standalone Retriever (pretrained mode)
    ):
        if rag_pipeline is None and pretrained_pipe is None:
            raise ValueError("Provide either rag_pipeline or pretrained_pipe.")

        self.tracker         = tracker
        self.rag             = rag_pipeline
        self.pretrained_pipe = pretrained_pipe
        self.retriever       = retriever or (rag_pipeline.retriever if rag_pipeline else None)

        self._current_chunk_id: Optional[int] = None
        self.session_history: list = []

    def _active_retriever(self):
        if self.retriever:
            return self.retriever
        if self.rag:
            return self.rag.retriever
        return None

    def next_question(self, chapter: str = None, topic: str = None):
        """
        Pick the next chunk by priority, then generate a quiz question from it.

        Returns (QuizQuestion | None, ChunkRecord | None).
        """
        result = self.tracker.next_chunk(chapter=chapter, topic=topic)
        if result is None:
            return None, None

        chunk_id, record = result
        self._current_chunk_id = chunk_id

        print(f"[Tracker] chunk {chunk_id}: {record.breadcrumb} "
              f"(seen {record.times_seen}×, acc {record.accuracy:.0%})")

        retriever = self._active_retriever()
        if retriever is None:
            return None, record

        if self.pretrained_pipe is not None:
            from quiz import generate_quiz_question_with_pretrained
            quiz = generate_quiz_question_with_pretrained(
                topic=record.topic,
                pipe=self.pretrained_pipe,
                retriever=retriever,
            )
        else:
            from quiz import generate_quiz_question_with_model
            quiz = generate_quiz_question_with_model(
                topic=record.topic,
                model=self.rag.model,
                tokenizer=self.rag.tokenizer,
                retriever=retriever,
                device=self.rag.device,
            )

        return quiz, record

    def submit_answer(self, answer_index: int, quiz) -> dict:
        """Record result and return feedback dict."""
        if quiz is None or self._current_chunk_id is None:
            return {"error": "No active question"}

        correct = (answer_index == quiz.correct_index)
        self.tracker.record(self._current_chunk_id, correct)
        self.session_history.append((quiz, answer_index, correct))

        n_correct = sum(1 for _, _, c in self.session_history if c)
        return {
            "correct":        correct,
            "correct_answer": quiz.correct_answer,
            "explanation":    quiz.explanation,
            "session_score":  f"{n_correct}/{len(self.session_history)}",
        }


# ─────────────────────────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import tempfile, os

    fake_chunks = [
        {"text": "Dropout randomly zeroes neuron outputs.", "source": "lec1.md",
         "metadata": {"h1": "Regularization", "h2": "Dropout",
                      "breadcrumb": "Regularization > Dropout"}},
        {"text": "Batch normalization normalizes layer inputs.", "source": "lec1.md",
         "metadata": {"h1": "Regularization", "h2": "Batch Norm",
                      "breadcrumb": "Regularization > Batch Norm"}},
        {"text": "Self-attention computes query/key/value.", "source": "lec2.md",
         "metadata": {"h1": "Transformers", "h2": "Self-Attention",
                      "breadcrumb": "Transformers > Self-Attention"}},
    ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(fake_chunks, f)
        tmp_path = f.name

    tracker = ProgressTracker(tmp_path, "test_progress.json")
    print("Chapters:", tracker.chapters())
    print("Topics in Regularization:", tracker.topics("Regularization"))

    cid, rec = tracker.next_chunk("Regularization")
    print(f"\nNext chunk: {rec.breadcrumb}")
    tracker.record(cid, correct=True)

    cid, rec = tracker.next_chunk("Regularization")
    print(f"Next chunk: {rec.breadcrumb}")
    tracker.record(cid, correct=False)

    tracker.print_progress()
    print("\n--- HTML preview (first 200 chars) ---")
    print(progress_to_html(tracker)[:200])

    os.unlink(tmp_path)
    Path("test_progress.json").unlink(missing_ok=True)