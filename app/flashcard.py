"""
flashcard.py  —  Flashcard generation for the Teaching Assistant
=================================================================
Generates two-sided flashcards from course material:
  Front: a key concept or question (≤15 words)
  Back:  a concise 2-3 sentence definition / explanation

The back is pre-generated immediately but hidden until the user flips the card.
Cards are saved to a JSON deck file that persists between sessions.

Two UI modes (handled in gradio_app.py):
  Generate tab — create new cards one at a time, flip to check yourself
  My Deck tab  — browse all saved cards, navigate with Previous / Next

Export:
  Deck can be exported as a tab-separated CSV for Anki import
  (File → Import, separator = Tab).
"""

from __future__ import annotations
import csv
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, List


# ─────────────────────────────────────────────────────────────────────────────
# Data structure
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Flashcard:
    front:        str          # concept / question side
    back:         str          # definition / explanation side
    topic:        str = ""
    source_chunk: str = ""

    def display(self) -> str:
        return f"▶ {self.front}\n\n  {self.back}"


# ─────────────────────────────────────────────────────────────────────────────
# HTML rendering  — produces a visual card for gr.HTML()
# ─────────────────────────────────────────────────────────────────────────────

def flashcard_to_html(card: Flashcard, revealed: bool = False) -> str:
    """
    Render a flashcard as styled HTML for display in gr.HTML().

    When revealed=False the back panel shows a blurred placeholder, so the
    student is prompted to recall the answer before flipping.
    """
    front_text = card.front if card else "No card loaded"
    back_text  = card.back  if card else ""

    back_inner = (
        back_text
        if revealed
        else '<span style="opacity:0.35;font-style:italic;">Flip to reveal the answer…</span>'
    )
    back_blur  = "" if revealed else "filter:blur(3px);pointer-events:none;"

    topic_badge = (
        f'<span style="'
        f'background:rgba(255,255,255,0.15);border-radius:6px;'
        f'padding:2px 8px;font-size:11px;letter-spacing:0.5px;">'
        f'{card.topic}</span>'
        if (card and card.topic) else ""
    )

    return f"""
<div style="font-family:'Segoe UI',system-ui,sans-serif;max-width:640px;margin:12px auto;">

  <!-- FRONT -->
  <div style="
    background:linear-gradient(135deg,#1a365d 0%,#2b6cb0 100%);
    color:#fff;border-radius:14px;padding:28px 32px 24px;
    margin-bottom:10px;box-shadow:0 4px 18px rgba(0,0,0,0.25);
    min-height:110px;
  ">
    <div style="
      font-size:10px;font-weight:700;text-transform:uppercase;
      letter-spacing:1.5px;opacity:0.55;margin-bottom:10px;
    ">▲ QUESTION</div>
    <div style="font-size:17px;font-weight:600;line-height:1.5;">
      {front_text}
    </div>
    <div style="margin-top:14px;">{topic_badge}</div>
  </div>

  <!-- BACK -->
  <div style="
    background:linear-gradient(135deg,#1a4731 0%,#2f855a 100%);
    color:#fff;border-radius:14px;padding:28px 32px 24px;
    box-shadow:0 4px 18px rgba(0,0,0,0.25);
    min-height:110px;{back_blur}
  ">
    <div style="
      font-size:10px;font-weight:700;text-transform:uppercase;
      letter-spacing:1.5px;opacity:0.55;margin-bottom:10px;
    ">▼ ANSWER</div>
    <div style="font-size:15px;line-height:1.6;">
      {back_inner}
    </div>
  </div>

</div>
"""


def deck_card_html(card: Flashcard, index: int, total: int, revealed: bool = False) -> str:
    """Flashcard HTML for deck browsing — includes a card counter."""
    base = flashcard_to_html(card, revealed=revealed)
    counter = (
        f'<div style="text-align:center;font-size:12px;color:#888;'
        f'font-family:sans-serif;margin-top:4px;">'
        f'Card {index + 1} of {total}</div>'
    )
    return base + counter


# ─────────────────────────────────────────────────────────────────────────────
# Deck persistence
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_DECK_PATH = "flashcards_deck.json"


def save_deck(cards: List[Flashcard], path: str = DEFAULT_DECK_PATH) -> None:
    """Save all cards to a JSON file, overwriting the previous save."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump([asdict(c) for c in cards], f, ensure_ascii=False, indent=2)


def load_deck(path: str = DEFAULT_DECK_PATH) -> List[Flashcard]:
    """Load cards from the JSON deck file.  Returns an empty list if the file doesn't exist."""
    p = Path(path)
    if not p.exists():
        return []
    try:
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        return [Flashcard(**d) for d in data]
    except Exception as e:
        print(f"  [flashcard] Could not load deck from {path}: {e}")
        return []


def append_card_to_deck(card: Flashcard, path: str = DEFAULT_DECK_PATH) -> List[Flashcard]:
    """Load the existing deck, append a new card, and save.  Returns the updated deck."""
    deck = load_deck(path)
    deck.append(card)
    save_deck(deck, path)
    return deck


# ─────────────────────────────────────────────────────────────────────────────
# Generation prompts
# ─────────────────────────────────────────────────────────────────────────────

FLASHCARD_PROMPT_TEMPLATE = """You are creating study flashcards for a machine learning course.
Based on the following lecture material, create ONE flashcard about: {topic}

Lecture material:
{context}

Respond ONLY with a JSON object, nothing else:
{{
  "front": "A short question or concept name (max 15 words)",
  "back":  "A clear 2-3 sentence explanation of the answer or concept"
}}"""


def _flashcard_chat_messages(topic: str, context: str) -> list:
    """
    Chat messages for instruction-tuned models.

    Sending the raw FLASHCARD_PROMPT_TEMPLATE to an instruction-tuned model
    causes it to echo placeholder text.  Chat format tells the model it is
    the assistant and must now reply — so it generates real content.
    """
    system = (
        "You are a flashcard generator for a machine learning course. "
        "Your only output is a single valid JSON object — no prose, no markdown fences. "
        "The JSON must have exactly two keys: "
        "\"front\" (string, ≤15 words) and \"back\" (string, 2-3 sentences)."
    )
    user = (
        f"Create one flashcard about the topic: \"{topic}\"\n\n"
        f"Base it on this lecture material:\n{context}\n\n"
        "Output only the JSON object."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]


# ─────────────────────────────────────────────────────────────────────────────
# JSON helpers
# ─────────────────────────────────────────────────────────────────────────────

import json as _json
import re as _re


def _extract_json(text: str) -> Optional[dict]:
    text = text.strip()
    try:
        return _json.loads(text)
    except _json.JSONDecodeError:
        pass
    match = _re.search(r'\{.*\}', text, _re.DOTALL)
    if match:
        try:
            return _json.loads(match.group())
        except _json.JSONDecodeError:
            pass
    return None


def _validate_flashcard_json(data: dict) -> Optional[Flashcard]:
    if not isinstance(data, dict):
        return None
    front = data.get("front", "").strip()
    back  = data.get("back",  "").strip()
    if len(front) < 5 or len(back) < 10:
        return None
    return Flashcard(front=front, back=back)


# ─────────────────────────────────────────────────────────────────────────────
# Generation — pretrained HuggingFace pipeline
# ─────────────────────────────────────────────────────────────────────────────

def generate_flashcard_with_pretrained(
    topic:       str,
    pipe,
    retriever,
    max_retries: int   = 3,
    temperature: float = 0.6,
) -> Optional[Flashcard]:
    chunks   = retriever.query(topic, top_k=2)
    context  = "\n---\n".join(c["text"] for c in chunks[:2])[:600]
    messages = _flashcard_chat_messages(topic, context)

    for attempt in range(max_retries):
        temp = temperature if attempt == 0 else 0.4
        try:
            result = pipe(messages, max_new_tokens=200, do_sample=True,
                          temperature=temp, top_p=0.95, return_full_text=False)
            raw = result[0]["generated_text"]
            generated = (raw[-1].get("content", "") if isinstance(raw, list)
                         else str(raw)).strip()
        except Exception as e:
            print(f"  Flashcard pipeline error attempt {attempt+1}: {e}")
            continue

        print(f"  [flashcard] attempt {attempt+1} raw: {repr(generated[:100])}")
        data = _extract_json(generated)
        if data:
            card = _validate_flashcard_json(data)
            if card:
                card.topic        = topic
                card.source_chunk = context
                return card

        print(f"  Flashcard attempt {attempt+1} failed. Retrying...")

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Generation — custom checkpoint
# ─────────────────────────────────────────────────────────────────────────────

def generate_flashcard_with_model(
    topic:       str,
    model,
    tokenizer,
    retriever,
    device:      str   = "cpu",
    max_retries: int   = 3,
    temperature: float = 0.7,
) -> Optional[Flashcard]:
    import torch
    chunks  = retriever.query(topic, top_k=2)
    context = "\n---\n".join(c["text"] for c in chunks[:2])[:600]
    prompt  = FLASHCARD_PROMPT_TEMPLATE.format(context=context, topic=topic)

    for attempt in range(max_retries):
        temp      = temperature if attempt == 0 else 0.5
        input_ids = tokenizer.encode(prompt)
        x         = torch.tensor([input_ids], dtype=torch.long).to(device)
        out       = model.generate(x, max_new_tokens=200, temperature=temp,
                                   top_k=50, top_p=0.95,
                                   stop_token=tokenizer.eot_id)
        generated = tokenizer.decode(out[0, len(input_ids):].tolist())
        generated = generated.replace("<|endoftext|>", "").strip()

        data = _extract_json(generated)
        if data:
            card = _validate_flashcard_json(data)
            if card:
                card.topic        = topic
                card.source_chunk = context
                return card

        print(f"  Flashcard attempt {attempt+1} failed. Retrying...")

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Export
# ─────────────────────────────────────────────────────────────────────────────

def export_to_anki_csv(cards: List[Flashcard], path: str = "flashcards.csv") -> str:
    """
    Export a list of flashcards to a CSV that Anki can import.
    Format: front TAB back (no header, UTF-8).
    Import in Anki: File → Import, separator = Tab.
    """
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        for card in cards:
            writer.writerow([card.front, card.back])
    return path