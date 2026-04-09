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
  priority = (1 - accuracy) + recency_penalty
where recency_penalty decreases for recently-seen items.

This naturally surfaces items you got wrong and items you haven't seen
recently. Good enough for a course project, and you can cite SM-2 as
"future work" in the report.
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
    chunk_id:    int          # index into chunks.json
    topic:       str          # h2 header (or h1 if no h2)
    chapter:     str          # h1 header
    breadcrumb:  str          # full path e.g. "Neural Networks > Backprop"
    text_preview: str         # first 80 chars for display

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
        Higher priority = should be shown sooner.

        Unseen items: priority = 1.0 (always prioritize new material)
        Seen items:   priority = (1 - accuracy) + recency_bonus
        where recency_bonus decays as time since last_seen increases.
        """
        if self.times_seen == 0:
            return 1.0  # unseen = highest priority

        # How long ago (in hours) was this last seen?
        hours_ago = (time.time() - self.last_seen) / 3600
        recency_bonus = min(hours_ago / 24, 1.0)  # caps at 1 after 24h

        return (1.0 - self.accuracy) + 0.3 * recency_bonus

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

        # Get next chunk to quiz on, filtered to a chapter
        chunk = tracker.next_chunk("Neural Networks")

        # Record the result
        tracker.record("Neural Networks", chunk_id=42, correct=True)

        # Get overall stats
        tracker.stats()
    """

    def __init__(self, chunks_json_path: str, save_path: str = "progress.json"):
        self.chunks_path = chunks_json_path
        self.save_path   = save_path

        # Load all chunks and build the curriculum map
        with open(chunks_json_path) as f:
            raw_chunks = json.load(f)

        # Build records dict: chunk_id → ChunkRecord
        self.records: Dict[int, ChunkRecord] = {}
        for i, chunk in enumerate(raw_chunks):
            meta = chunk.get("metadata", {})
            h1   = meta.get("h1") or "General"
            h2   = meta.get("h2") or h1
            crumb = meta.get("breadcrumb") or h1
            self.records[i] = ChunkRecord(
                chunk_id    = i,
                topic       = h2,
                chapter     = h1,
                breadcrumb  = crumb,
                text_preview = chunk.get("text", "")[:80].replace("\n", " "),
            )

        # Build curriculum map: h1 → {h2 → [chunk_ids]}
        self.curriculum_map: Dict[str, Dict[str, List[int]]] = {}
        for chunk_id, rec in self.records.items():
            chapter = rec.chapter
            topic   = rec.topic
            if chapter not in self.curriculum_map:
                self.curriculum_map[chapter] = {}
            if topic not in self.curriculum_map[chapter]:
                self.curriculum_map[chapter][topic] = []
            self.curriculum_map[chapter][topic].append(chunk_id)

        # Load existing progress if available
        self._load()
        print(f"Curriculum: {len(self.records)} chunks across "
              f"{len(self.curriculum_map)} chapters")

    # ── Navigation ────────────────────────────────────────────────────────────

    def chapters(self) -> List[str]:
        """List all available chapters (h1 headers)."""
        return sorted(self.curriculum_map.keys())

    def topics(self, chapter: str = None) -> List[str]:
        """List all topics, optionally filtered to a chapter."""
        if chapter:
            return sorted(self.curriculum_map.get(chapter, {}).keys())
        all_topics = set()
        for topics in self.curriculum_map.values():
            all_topics.update(topics.keys())
        return sorted(all_topics)

    def next_chunk(
        self,
        chapter: str = None,
        topic:   str = None,
        prefer_unseen: bool = True,
    ) -> Optional[Tuple[int, ChunkRecord]]:
        """
        Return the next chunk to study, prioritized by the student's progress.

        If chapter is given, filter to that chapter.
        If topic is given, filter to that topic.
        Returns (chunk_id, ChunkRecord) or None if nothing available.

        Selection strategy:
          1. If prefer_unseen: return an unseen chunk first (if any)
          2. Otherwise: return the highest-priority chunk by the priority score
        """
        # Gather candidate chunk IDs
        if topic:
            # Find which chapter this topic belongs to (search all chapters)
            candidates = []
            for ch, topics in self.curriculum_map.items():
                if topic.lower() in [t.lower() for t in topics]:
                    for t, ids in topics.items():
                        if t.lower() == topic.lower():
                            candidates.extend(ids)
        elif chapter:
            candidates = []
            for t, ids in self.curriculum_map.get(chapter, {}).items():
                candidates.extend(ids)
        else:
            candidates = list(self.records.keys())

        if not candidates:
            return None

        # Sort by priority (highest first)
        sorted_ids = sorted(candidates, key=lambda i: self.records[i].priority, reverse=True)

        if prefer_unseen:
            # Return first unseen chunk if exists
            for cid in sorted_ids:
                if self.records[cid].times_seen == 0:
                    return cid, self.records[cid]

        # Return highest priority (seen but needs review)
        return sorted_ids[0], self.records[sorted_ids[0]]

    def chunks_for_topic(self, topic: str) -> List[Tuple[int, ChunkRecord]]:
        """Return all chunks for a topic, sorted by priority."""
        result = []
        for ch, topics in self.curriculum_map.items():
            for t, ids in topics.items():
                if t.lower() == topic.lower():
                    result.extend([(cid, self.records[cid]) for cid in ids])
        return sorted(result, key=lambda x: x[1].priority, reverse=True)

    # ── Recording progress ────────────────────────────────────────────────────

    def record(self, chunk_id: int, correct: bool):
        """Record the result of a quiz question on a specific chunk."""
        if chunk_id in self.records:
            self.records[chunk_id].record_answer(correct)
            self._save()

    # ── Stats and reporting ───────────────────────────────────────────────────

    def stats(self) -> dict:
        """Return overall progress statistics."""
        total     = len(self.records)
        seen      = sum(1 for r in self.records.values() if r.times_seen > 0)
        correct   = sum(r.times_correct for r in self.records.values())
        answered  = sum(r.times_seen    for r in self.records.values())

        # Per-chapter breakdown
        chapter_stats = {}
        for chapter, topics in self.curriculum_map.items():
            all_ids   = [cid for ids in topics.values() for cid in ids]
            seen_ids  = [cid for cid in all_ids if self.records[cid].times_seen > 0]
            correct_c = sum(self.records[cid].times_correct for cid in all_ids)
            answered_c = sum(self.records[cid].times_seen   for cid in all_ids)
            chapter_stats[chapter] = {
                "total_chunks":  len(all_ids),
                "seen_chunks":   len(seen_ids),
                "accuracy":      correct_c / max(answered_c, 1),
                "pct_complete":  len(seen_ids) / len(all_ids) * 100,
            }

        return {
            "total_chunks":   total,
            "seen_chunks":    seen,
            "pct_seen":       seen / total * 100,
            "overall_accuracy": correct / max(answered, 1),
            "chapters":       chapter_stats,
        }

    def weak_topics(self, n: int = 5) -> List[Tuple[str, float]]:
        """Return the n topics the student struggles with most."""
        topic_scores: Dict[str, List[float]] = {}
        for rec in self.records.values():
            if rec.times_seen > 0:
                t = rec.topic
                if t not in topic_scores:
                    topic_scores[t] = []
                topic_scores[t].append(rec.accuracy)
        avg_scores = {t: sum(scores) / len(scores) for t, scores in topic_scores.items()}
        return sorted(avg_scores.items(), key=lambda x: x[1])[:n]

    def print_progress(self):
        """Print a readable progress report."""
        s = self.stats()
        print(f"\n{'='*50}")
        print(f"Progress: {s['seen_chunks']}/{s['total_chunks']} chunks "
              f"({s['pct_seen']:.0f}% covered)")
        print(f"Overall accuracy: {s['overall_accuracy']:.0%}")
        print(f"\nChapter breakdown:")
        for ch, cs in s["chapters"].items():
            bar_len  = int(cs["pct_complete"] / 5)   # 1 char per 5%
            bar      = "█" * bar_len + "░" * (20 - bar_len)
            print(f"  {ch[:30]:<30} {bar} {cs['pct_complete']:.0f}% | "
                  f"acc {cs['accuracy']:.0%}")

        weak = self.weak_topics(3)
        if weak:
            print(f"\nWeakest topics:")
            for topic, acc in weak:
                print(f"  {topic[:35]:<35} {acc:.0%} accuracy")
        print("="*50)

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save(self):
        """Save progress to JSON file."""
        data = {str(k): asdict(v) for k, v in self.records.items()}
        with open(self.save_path, "w") as f:
            json.dump(data, f, indent=2)

    def _load(self):
        """Load existing progress from JSON file (if it exists)."""
        if not Path(self.save_path).exists():
            return
        with open(self.save_path) as f:
            data = json.load(f)
        for k, v in data.items():
            chunk_id = int(k)
            if chunk_id in self.records:
                # Update existing record with saved progress
                self.records[chunk_id].times_seen    = v.get("times_seen", 0)
                self.records[chunk_id].times_correct = v.get("times_correct", 0)
                self.records[chunk_id].last_seen     = v.get("last_seen", 0.0)
        print(f"Progress loaded from {self.save_path}")

    def reset(self, chapter: str = None):
        """Reset progress (optionally for a specific chapter only)."""
        if chapter:
            for topics in self.curriculum_map.get(chapter, {}).values():
                for cid in topics:
                    r = self.records[cid]
                    r.times_seen = r.times_correct = 0
                    r.last_seen = 0.0
        else:
            for r in self.records.values():
                r.times_seen = r.times_correct = 0
                r.last_seen = 0.0
        self._save()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: Quiz session that uses the tracker
# ─────────────────────────────────────────────────────────────────────────────

class TrackedQuizSession:
    """
    Quiz session that integrates with the progress tracker.

    This replaces the simple QuizSession in quiz.py with one that:
    - Navigates the curriculum systematically
    - Records performance per chunk
    - Adapts: focuses on weak spots, avoids repetition of correct answers

    This is NOT agent behaviour — it's a simple priority queue based on
    the progress tracker's priority scores.
    """

    def __init__(self, rag_pipeline, chunks_json: str, progress_path: str = "progress.json"):
        from quiz import generate_quiz_question_with_model
        self.rag      = rag_pipeline
        self.tracker  = ProgressTracker(chunks_json, progress_path)
        self._gen_fn  = generate_quiz_question_with_model
        self._current_chunk_id = None
        self.session_history   = []

    def next_question(self, chapter: str = None, topic: str = None):
        """
        Generate the next quiz question, chosen by progress-aware priority.
        """
        result = self.tracker.next_chunk(chapter=chapter, topic=topic)
        if result is None:
            return None, "No chunks available for this filter."

        chunk_id, record = result
        self._current_chunk_id = chunk_id

        # Load the actual chunk text for generation
        with open(self.tracker.chunks_path) as f:
            chunks = json.load(f)
        chunk_text = chunks[chunk_id]["text"][:500]

        print(f"[Tracker] Selected chunk {chunk_id}: {record.breadcrumb} "
              f"(seen {record.times_seen}x, acc {record.accuracy:.0%})")

        quiz = self._gen_fn(
            topic=record.topic,
            model=self.rag.model,
            tokenizer=self.rag.tokenizer,
            retriever=self.rag.retriever,
            device=self.rag.device,
        )
        return quiz, record

    def submit_answer(self, answer_index: int, quiz):
        """Record the answer and update the tracker."""
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

    def progress_summary(self):
        self.tracker.print_progress()


if __name__ == "__main__":
    import tempfile, os

    # Create a fake chunks.json for testing
    fake_chunks = [
        {"text": "Dropout randomly zeroes neuron outputs.", "source": "lec1.md",
         "metadata": {"h1": "Regularization", "h2": "Dropout", "breadcrumb": "Regularization > Dropout"}},
        {"text": "Batch normalization normalizes layer inputs.", "source": "lec1.md",
         "metadata": {"h1": "Regularization", "h2": "Batch Norm", "breadcrumb": "Regularization > Batch Norm"}},
        {"text": "Self-attention computes query/key/value.", "source": "lec2.md",
         "metadata": {"h1": "Transformers", "h2": "Self-Attention", "breadcrumb": "Transformers > Self-Attention"}},
    ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(fake_chunks, f)
        tmp_path = f.name

    tracker = ProgressTracker(tmp_path, "test_progress.json")
    print("Chapters:", tracker.chapters())
    print("Topics:", tracker.topics())
    print("Topics in Regularization:", tracker.topics("Regularization"))

    # Simulate quiz interactions
    result = tracker.next_chunk("Regularization")
    if result:
        cid, rec = result
        print(f"\nNext chunk: {rec.breadcrumb}")
        tracker.record(cid, correct=True)

    result = tracker.next_chunk("Regularization")
    if result:
        cid, rec = result
        print(f"Next chunk: {rec.breadcrumb}")
        tracker.record(cid, correct=False)

    tracker.print_progress()
    os.unlink(tmp_path)
    Path("test_progress.json").unlink(missing_ok=True)
