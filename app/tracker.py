# tracker.py
"""
Tracks what the student has covered and how well.

Curriculum structure is read from chunks.json (produced by the chunker).
Each chunk carries h1/h2/breadcrumb metadata that maps cleanly onto a
chapter/section hierarchy.

Progress is stored per-chunk in a JSON file.  Priority scoring:
    priority = (1 - accuracy) + recency_bonus
where recency_bonus grows as time-since-last-seen increases, capped at 0.3
after 24 hours.  Unseen chunks always get priority = 1.0.
"""

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class ChunkRecord:
    """Tracks a student's interaction history with a single chunk."""
    chunk_id:     int
    topic:        str
    chapter:      str
    breadcrumb:   str
    text_preview: str

    times_seen:    int   = 0
    times_correct: int   = 0
    last_seen:     float = 0.0

    @property
    def accuracy(self) -> float:
        if self.times_seen == 0:
            return 0.0
        return self.times_correct / self.times_seen

    @property
    def priority(self) -> float:
        if self.times_seen == 0:
            return 1.0
        hours_ago     = (time.time() - self.last_seen) / 3600
        recency_bonus = min(hours_ago / 24, 1.0) * 0.3
        return (1.0 - self.accuracy) + recency_bonus

    def record_answer(self, correct: bool) -> None:
        self.times_seen    += 1
        self.times_correct += (1 if correct else 0)
        self.last_seen      = time.time()


class ProgressTracker:
    """
    Manages curriculum progress, chunk selection, and chapter navigation.

    Usage:
        tracker = ProgressTracker("rag_index/chunks.json")
        chunk_id, record = tracker.next_chunk(breadcrumb_prefix="Book > Chapter 2")
        tracker.record(chunk_id, correct=True)
    """

    def __init__(self, chunks_json_path: str, save_path: str = "progress.json"):
        self.chunks_path = chunks_json_path
        self.save_path   = save_path

        with open(chunks_json_path, encoding="utf-8") as f:
            self.raw_chunks: list = json.load(f)

        self.records: Dict[int, ChunkRecord] = {}
        for i, chunk in enumerate(self.raw_chunks):
            meta  = chunk.get("metadata", {})
            h1    = meta.get("h1") or "General"
            h2    = meta.get("h2") or h1
            crumb = meta.get("breadcrumb") or h1
            leaf  = crumb.split(">")[-1].strip() if ">" in crumb else h2
            self.records[i] = ChunkRecord(
                chunk_id     = i,
                topic        = leaf,
                chapter      = h1,
                breadcrumb   = crumb,
                text_preview = chunk.get("text", "")[:80].replace("\n", " "),
            )

        # h1 -> { leaf_topic -> [chunk_ids] }
        self.curriculum_map: Dict[str, Dict[str, List[int]]] = {}
        for chunk_id, rec in self.records.items():
            self.curriculum_map.setdefault(rec.chapter, {}).setdefault(rec.topic, []).append(chunk_id)

        self._load()
        print(f"Tracker: {len(self.records)} chunks, {len(self.curriculum_map)} chapters")

    # -------------------------------------------------------------------------
    # Navigation
    # -------------------------------------------------------------------------

    def chapters(self) -> List[str]:
        """Sorted list of H1 chapter names."""
        return sorted(self.curriculum_map.keys())

    def topics(self, chapter: str = None) -> List[str]:
        """Sorted list of leaf topics, optionally filtered by chapter."""
        if chapter:
            return sorted(self.curriculum_map.get(chapter, {}).keys())
        topics: set = set()
        for t in self.curriculum_map.values():
            topics.update(t.keys())
        return sorted(topics)

    def next_chunk(
        self,
        chapter:           str = None,
        topic:             str = None,
        breadcrumb_prefix: str = None,
    ) -> Optional[Tuple[int, ChunkRecord]]:
        """
        Return the highest-priority chunk to study next.

        Priority order: unseen chunks first, then lowest-accuracy + recency.
        At most one of breadcrumb_prefix / chapter / topic should be set.
        breadcrumb_prefix takes precedence when given.
        """
        if breadcrumb_prefix:
            candidates = [
                cid for cid, rec in self.records.items()
                if rec.breadcrumb.startswith(breadcrumb_prefix)
            ]
        elif topic:
            candidates = []
            for ch_topics in self.curriculum_map.values():
                for t, ids in ch_topics.items():
                    if t.lower() == topic.lower():
                        candidates.extend(ids)
        elif chapter:
            candidates = [
                cid
                for ids in self.curriculum_map.get(chapter, {}).values()
                for cid in ids
            ]
        else:
            candidates = list(self.records.keys())

        if not candidates:
            return None

        sorted_ids = sorted(candidates, key=lambda i: self.records[i].priority, reverse=True)

        for cid in sorted_ids:
            if self.records[cid].times_seen == 0:
                return cid, self.records[cid]

        return sorted_ids[0], self.records[sorted_ids[0]]

    def chunk_text(self, chunk_id: int) -> str:
        return self.raw_chunks[chunk_id].get("text", "")

    def chunks_for_topic(self, topic: str) -> List[Tuple[int, ChunkRecord]]:
        result = []
        for ch_topics in self.curriculum_map.values():
            for t, ids in ch_topics.items():
                if t.lower() == topic.lower():
                    result.extend([(cid, self.records[cid]) for cid in ids])
        return sorted(result, key=lambda x: x[1].priority, reverse=True)

    # -------------------------------------------------------------------------
    # Recording progress
    # -------------------------------------------------------------------------

    def record(self, chunk_id: int, correct: bool) -> None:
        if chunk_id in self.records:
            self.records[chunk_id].record_answer(correct)
            self._save()

    # -------------------------------------------------------------------------
    # Stats and reporting
    # -------------------------------------------------------------------------

    def stats(self) -> dict:
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
        topic_scores: Dict[str, List[float]] = {}
        for rec in self.records.values():
            if rec.times_seen > 0:
                topic_scores.setdefault(rec.topic, []).append(rec.accuracy)
        avg = {t: sum(v) / len(v) for t, v in topic_scores.items()}
        return sorted(avg.items(), key=lambda x: x[1])[:n]

    # -------------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------------

    def reset(self, breadcrumb_prefix: str = None, chapter: str = None) -> None:
        """
        Reset progress for matching chunks.

        breadcrumb_prefix targets all chunks whose breadcrumb starts with the
        given string, allowing reset of a section or subsection.
        chapter resets an entire H1 chapter.
        Passing neither resets everything.
        """
        if breadcrumb_prefix:
            targets = [
                cid for cid, rec in self.records.items()
                if rec.breadcrumb.startswith(breadcrumb_prefix)
            ]
        elif chapter:
            targets = [
                cid
                for ids in self.curriculum_map.get(chapter, {}).values()
                for cid in ids
            ]
        else:
            targets = list(self.records.keys())

        for cid in targets:
            r = self.records[cid]
            r.times_seen = r.times_correct = 0
            r.last_seen  = 0.0
        self._save()

    def _save(self) -> None:
        data = {str(k): asdict(v) for k, v in self.records.items()}
        with open(self.save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def _load(self) -> None:
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
            print(f"[tracker] Could not load progress: {e}")


# =============================================================================
# HTML progress dashboard
# =============================================================================

def progress_to_html(tracker: ProgressTracker) -> str:
    s        = tracker.stats()
    pct_seen = s["pct_seen"]
    acc      = s["overall_accuracy"] * 100
    seen     = s["seen_chunks"]
    total    = s["total_chunks"]

    chapter_rows = ""
    for ch, cs in sorted(s["chapters"].items(), key=lambda x: x[0]):
        pct     = cs["pct_complete"]
        a       = cs["accuracy"] * 100
        seen_ch = cs["seen_chunks"]
        tot_ch  = cs["total_chunks"]

        bar_color = ("#2f855a" if pct >= 70 else "#b7791f" if pct >= 30 else "#4a5568")
        acc_color = ("#2f855a" if a  >= 70 else "#c53030" if a  < 40  else "#b7791f")

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

    weak = tracker.weak_topics(5)
    if weak:
        weak_items = "".join(
            f'<li style="margin:4px 0;font-size:13px;color:#e2e8f0;">'
            f'<span style="color:#fc8181;font-weight:600;">{t}</span>'
            f'<span style="color:#718096;"> — {a*100:.0f}% accuracy</span></li>'
            for t, a in weak
        )
        weak_section = f"""
        <div style="margin-top:20px;">
          <div style="font-size:13px;font-weight:700;color:#a0aec0;
                      text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;">
            Topics to revisit
          </div>
          <ul style="margin:0;padding-left:20px;">{weak_items}</ul>
        </div>"""
    else:
        weak_section = (
            '<p style="color:#718096;font-size:13px;margin-top:16px;">'
            "Answer some questions to see weak topic analysis.</p>"
        )

    return f"""
<div style="font-family:'Segoe UI',system-ui,sans-serif;
            background:#1a202c;border-radius:14px;padding:24px 28px;
            color:#e2e8f0;max-width:700px;margin:8px auto;">
  <div style="font-size:18px;font-weight:700;margin-bottom:18px;">Curriculum Progress</div>
  <div style="display:flex;gap:16px;margin-bottom:20px;flex-wrap:wrap;">
    <div style="flex:1;background:#2d3748;border-radius:10px;padding:14px 18px;min-width:140px;">
      <div style="font-size:11px;color:#a0aec0;text-transform:uppercase;letter-spacing:1px;">Coverage</div>
      <div style="font-size:28px;font-weight:700;color:#63b3ed;margin:4px 0;">{pct_seen:.0f}%</div>
      <div style="background:#4a5568;border-radius:4px;height:8px;overflow:hidden;margin-top:6px;">
        <div style="width:{pct_seen:.1f}%;background:#3182ce;height:100%;border-radius:4px;"></div>
      </div>
      <div style="font-size:11px;color:#718096;margin-top:4px;">{seen} / {total} chunks</div>
    </div>
    <div style="flex:1;background:#2d3748;border-radius:10px;padding:14px 18px;min-width:140px;">
      <div style="font-size:11px;color:#a0aec0;text-transform:uppercase;letter-spacing:1px;">Overall accuracy</div>
      <div style="font-size:28px;font-weight:700;
                  color:{'#68d391' if acc>=70 else '#fc8181' if acc<40 else '#f6ad55'};
                  margin:4px 0;">{acc:.1f}%</div>
      <div style="font-size:11px;color:#718096;margin-top:4px;">
        {total - seen} chunks not yet seen
      </div>
    </div>
  </div>
  <div style="font-size:13px;font-weight:700;color:#a0aec0;
              text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;">Chapters</div>
  <table style="width:100%;border-collapse:collapse;">
    <tbody>{chapter_rows}</tbody>
  </table>
  {weak_section}
</div>
"""


# =============================================================================
# Self-test
# =============================================================================

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
        tmp = f.name

    tracker = ProgressTracker(tmp, "test_progress.json")
    print("Chapters:", tracker.chapters())

    cid, rec = tracker.next_chunk("Regularization")
    print("Next:", rec.breadcrumb)
    tracker.record(cid, correct=True)

    cid, rec = tracker.next_chunk(breadcrumb_prefix="Regularization")
    print("Next:", rec.breadcrumb)
    tracker.record(cid, correct=False)

    os.unlink(tmp)
    Path("test_progress.json").unlink(missing_ok=True)