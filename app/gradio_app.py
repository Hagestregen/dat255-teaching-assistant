"""
gradio_app.py  —  Gradio web interface for the Teaching Assistant
==================================================================
Five tabs:
  1. 💬 Ask a question   — RAG-assisted answer, shows retrieved sources
  2. 🧠 Quiz me          — multiple-choice questions, optional chapter filter
  3. 📝 Practice answer  — assistant generates a question; you answer; it grades you
  4. 🗂️ Flashcards       — Generate tab + My Deck tab
  5. 📊 Progress         — curriculum coverage dashboard (tracking mode only)

TWO MODES:
──────────
  Random mode (default):
    Questions are drawn from random topics across the whole index.
    No history is saved.  Useful for free-form exploration.

  Tracking mode (--track):
    Questions are drawn using a priority score: unseen material first, then
    lowest-accuracy topics, with a recency bonus so old correct answers
    resurface over time.  Performance is saved to progress.json.
    The Progress tab shows your coverage dashboard.

    Requires --chunks (path to chunks.json from your chunker).
    Defaults to {--index}/chunks.json if not set.

Run examples
────────────
  # Random mode (no tracking):
  python gradio_app.py --pretrained qwen-3b --index ../rag/rag_index

  # Tracking mode:
  python gradio_app.py --pretrained qwen-3b --index ../rag/rag_index --track

  # Tracking mode with explicit paths:
  python gradio_app.py --pretrained qwen-3b --index ../rag/rag_index \\
      --track --chunks ../rag/rag_index/chunks.json --progress my_progress.json

  # Custom checkpoint + tracking:
  python gradio_app.py --checkpoint ../model/checkpoints/step_005000.pt \\
      --index ../rag/rag_index --track
"""

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import gradio as gr
import torch


# ─────────────────────────────────────────────────────────────────────────────
# Global state  (loaded once at startup)
# ─────────────────────────────────────────────────────────────────────────────

_pipeline        = None   # custom RAGPipeline (checkpoint mode)
_pretrained_pipe = None   # HuggingFace text-generation pipeline
_retriever       = None   # standalone Retriever (pretrained mode)

_tracker         = None   # ProgressTracker — None when tracking is disabled

# Per-question state used for result recording
_current_quiz:          object = None
_current_quiz_chunk_id: int    = None   # chunk_id for this quiz question (tracking)
_current_fb_chunk_id:   int    = None   # chunk_id for current practice question (tracking)


def _active_retriever():
    if _pipeline is not None:
        return _pipeline.retriever
    return _retriever


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_pipeline(checkpoint: str, index_dir: str):
    global _pipeline
    from rag_pipeline import RAGPipeline
    _pipeline = RAGPipeline(checkpoint, index_dir, top_k=3)
    print("Custom pipeline loaded.")


def load_retriever(index_dir: str):
    global _retriever
    from rag.retriever import Retriever
    _retriever = Retriever(index_dir)
    print(f"Retriever loaded from {index_dir}")


PRETRAINED_MODELS = {
    "qwen-1.5b":  "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen-3b":    "Qwen/Qwen2.5-3B-Instruct",
    "qwen-3b-4b": "Qwen/Qwen2.5-3B-Instruct:bnb4",
    "mistral-4b": "mistralai/Mistral-7B-Instruct-v0.3:bnb4",
}


def _clear_generation_config(pipe):
    try:
        pipe.model.generation_config.max_length = None
    except Exception as e:
        print(f"  [WARN] Could not clear generation_config.max_length: {e}")


def load_pretrained(model_key_or_id: str):
    global _pretrained_pipe
    from transformers import (pipeline, AutoModelForCausalLM, AutoTokenizer,
                               BitsAndBytesConfig)

    model_id = PRETRAINED_MODELS.get(model_key_or_id, model_key_or_id)
    use_gpu  = torch.cuda.is_available()

    if model_id.endswith(":bnb4"):
        real_id = model_id[:-5]
        try:
            print(f"Loading {real_id} (4-bit) ...")
            bnb_cfg   = BitsAndBytesConfig(load_in_4bit=True,
                                           bnb_4bit_compute_dtype=torch.float16)
            tokenizer = AutoTokenizer.from_pretrained(real_id)
            model     = AutoModelForCausalLM.from_pretrained(
                            real_id, quantization_config=bnb_cfg, device_map="auto")
            _pretrained_pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
            _clear_generation_config(_pretrained_pipe)
            print("Pretrained pipeline loaded (4-bit).")
            return
        except ImportError:
            print("  [WARN] bitsandbytes not installed -- falling back to fp16")
            model_id = real_id

    device = 0 if use_gpu else -1
    dtype  = torch.float16 if use_gpu else torch.float32
    print(f"Loading {model_id} on {'GPU' if use_gpu else 'CPU'} ...")
    _pretrained_pipe = pipeline(
        "text-generation", model=model_id,
        device=device, torch_dtype=dtype, trust_remote_code=True,
    )
    _clear_generation_config(_pretrained_pipe)
    print("Pretrained pipeline loaded.")


# ─────────────────────────────────────────────────────────────────────────────
# Tab 1: Ask a question
# ─────────────────────────────────────────────────────────────────────────────

def ask_question(question: str, use_rag: bool, temperature: float):
    if not question.strip():
        return "Please enter a question.", ""

    from generation import answer_question
    result = answer_question(
        question, use_rag, temperature,
        pipeline=_pipeline,
        pretrained_pipe=_pretrained_pipe,
        retriever=_retriever,
    )

    if not result["chunks"]:
        sources_md = "_No RAG context used._"
    else:
        parts = [
            f"**Source {i+1}** (score={c['score']:.2f}): {c.get('breadcrumb','')}\n"
            f"{c['text'][:300]}..."
            for i, c in enumerate(result["chunks"])
        ]
        sources_md = "### Retrieved context\n" + "\n\n".join(parts)

    return result["answer"], sources_md


# ─────────────────────────────────────────────────────────────────────────────
# Tab 2: Quiz
# ─────────────────────────────────────────────────────────────────────────────

def new_quiz_question(topic: str, chapter: str):
    global _current_quiz, _current_quiz_chunk_id
    retriever = _active_retriever()
    if retriever is None:
        return ("No retriever loaded.", "",
                gr.update(choices=[]), gr.update(visible=False), "")

    from quiz import (generate_quiz_question_with_pretrained,
                      generate_quiz_question_with_model)
    from generation import get_random_topic_from_retriever

    # ── Tracking mode: tracker picks the chunk by priority ───────────────────
    if _tracker is not None:
        filter_chapter = chapter if chapter and chapter != "-- All chapters --" else None
        filter_topic   = topic.strip() or None
        result = _tracker.next_chunk(chapter=filter_chapter, topic=filter_topic)
        if result is None:
            return ("No chunks found for that filter.", "",
                    gr.update(choices=[]), gr.update(visible=False), "")
        chunk_id, record = result
        _current_quiz_chunk_id = chunk_id
        effective_topic = record.topic
        status = (f"**{record.breadcrumb}**  |  "
                  f"seen {record.times_seen}x  |  accuracy {record.accuracy:.0%}")
    else:
        # ── Random mode ───────────────────────────────────────────────────────
        _current_quiz_chunk_id = None
        effective_topic = topic.strip() if topic.strip() else get_random_topic_from_retriever(retriever)
        status = f"Topic: *{effective_topic}*"

    print(f"Quiz topic: {effective_topic!r}")

    if _pretrained_pipe is not None:
        _current_quiz = generate_quiz_question_with_pretrained(
            topic=effective_topic, pipe=_pretrained_pipe, retriever=retriever)
    elif _pipeline is not None:
        _current_quiz = generate_quiz_question_with_model(
            topic=effective_topic, model=_pipeline.model,
            tokenizer=_pipeline.tokenizer, retriever=retriever,
            device=_pipeline.device)
    else:
        return ("No model loaded.", "",
                gr.update(choices=[]), gr.update(visible=False), "")

    if _current_quiz is None:
        return ("Failed to generate question -- try a different topic.", "",
                gr.update(choices=[]), gr.update(visible=False), status)

    choices = [f"{l}) {o}" for l, o in zip("ABCD", _current_quiz.options)]
    return (
        _current_quiz.question,
        "",
        gr.update(choices=choices, value=None, visible=True),
        gr.update(visible=True),
        status,
    )


def submit_quiz_answer(selected: str):
    if _current_quiz is None or not selected:
        return "Please select an answer first."

    idx     = "ABCD".index(selected[0])
    correct = idx == _current_quiz.correct_index

    if _tracker is not None and _current_quiz_chunk_id is not None:
        _tracker.record(_current_quiz_chunk_id, correct)

    if correct:
        return f"Correct!\n\n{_current_quiz.explanation}"

    cl = "ABCD"[_current_quiz.correct_index]
    return (f"Wrong. The correct answer was {cl}) "
            f"{_current_quiz.correct_answer}\n\n{_current_quiz.explanation}")


# ─────────────────────────────────────────────────────────────────────────────
# Tab 3: Practice answer
# ─────────────────────────────────────────────────────────────────────────────

def generate_feedback_question(topic: str, chapter: str):
    global _current_fb_chunk_id
    retriever = _active_retriever()
    if retriever is None:
        return (gr.update(value="No retriever loaded.", interactive=False),
                gr.update(visible=False), gr.update(visible=False), "")

    from feedback_mode import generate_question_for_feedback

    if _tracker is not None:
        filter_chapter = chapter if chapter and chapter != "-- All chapters --" else None
        filter_topic   = topic.strip() or None
        result = _tracker.next_chunk(chapter=filter_chapter, topic=filter_topic)
        if result:
            chunk_id, record = result
            _current_fb_chunk_id = chunk_id
            effective_topic = record.topic
            status = (f"**{record.breadcrumb}**  |  "
                      f"seen {record.times_seen}x  |  accuracy {record.accuracy:.0%}")
        else:
            _current_fb_chunk_id = None
            effective_topic = topic.strip()
            status = "No chunks found for that filter."
    else:
        _current_fb_chunk_id = None
        effective_topic = topic.strip()
        status = ""

    question = generate_question_for_feedback(
        retriever       = retriever,
        topic           = effective_topic,
        pretrained_pipe = _pretrained_pipe,
        model           = _pipeline.model     if _pipeline else None,
        tokenizer       = _pipeline.tokenizer if _pipeline else None,
        device          = _pipeline.device    if _pipeline else "cpu",
    )

    if not question:
        return (gr.update(value="Failed to generate a question -- try entering a topic manually.",
                          interactive=True),
                gr.update(visible=False), gr.update(visible=False), status)

    return (
        gr.update(value=question, interactive=True),
        gr.update(visible=True),
        gr.update(visible=True),
        status,
    )


def review_student_answer(question: str, student_answer: str):
    if not question.strip():
        return "Please generate or enter a question first."
    if not student_answer.strip():
        return "Please write your answer before submitting."

    retriever = _active_retriever()

    # ── Pretrained path ───────────────────────────────────────────────────────
    if _pretrained_pipe is not None:
        context = ""
        if retriever:
            try:
                chunks  = retriever.query(question, top_k=2)
                context = "\n---\n".join(c["text"][:200] for c in chunks[:2])
            except Exception:
                pass

        system = (
            "You are a strict but fair teaching assistant for a machine learning course. "
            "Review the student answer. Score it 1-5 and give one specific piece of "
            "feedback: what they got right and what they missed. "
            "Format: Score: X/5. <feedback>"
        )
        user_parts = []
        if context:
            user_parts.append(f"Context from course material:\n{context}")
        user_parts.append(f"Question: {question}")
        user_parts.append(f"Student answer: {student_answer}")
        user_parts.append("Review:")

        messages = [
            {"role": "system", "content": system},
            {"role": "user",   "content": "\n\n".join(user_parts)},
        ]
        try:
            result = _pretrained_pipe(messages, max_new_tokens=200, do_sample=True,
                                      temperature=0.4, top_p=0.9,
                                      return_full_text=False)
            raw = result[0]["generated_text"]
            if isinstance(raw, list):
                raw = raw[-1].get("content", "")
            raw = str(raw).strip()
        except Exception as e:
            return f"Generation error: {e}"

        formatted = _format_review(raw)
        _maybe_record_fb(formatted)
        return formatted

    # ── Custom checkpoint path ────────────────────────────────────────────────
    if _pipeline is not None:
        from feedback_mode import review_answer_with_model, format_review_for_display
        review    = review_answer_with_model(question, student_answer, _pipeline)
        formatted = format_review_for_display(review, question, student_answer)
        if _tracker is not None and _current_fb_chunk_id is not None:
            _tracker.record(_current_fb_chunk_id, correct=(review.get("score", 0) >= 3))
        return formatted

    return "No model loaded."


def _format_review(raw: str) -> str:
    import re
    m = re.search(r'[Ss]core[:\s]+(\d)[/\s]*5', raw)
    if m:
        score    = int(m.group(1))
        feedback = raw[m.end():].strip().lstrip('.').strip()
    elif raw and raw[0].isdigit():
        score    = int(raw[0])
        feedback = raw[1:].strip().lstrip('/5').lstrip('.').strip()
    else:
        score, feedback = 0, raw
    stars = "★" * score + "☆" * (5 - score)
    return f"**Score: {score}/5** {stars}\n\n{feedback}"


def _maybe_record_fb(formatted_md: str):
    import re
    if _tracker is None or _current_fb_chunk_id is None:
        return
    m = re.search(r'Score:\s*(\d)', formatted_md)
    score = int(m.group(1)) if m else 0
    _tracker.record(_current_fb_chunk_id, correct=(score >= 3))


# ─────────────────────────────────────────────────────────────────────────────
# Tab 4: Flashcards
# ─────────────────────────────────────────────────────────────────────────────

_flashcard_deck: list = []
_gen_current_card     = None
_gen_revealed         = False
_deck_index:  int     = 0
_deck_revealed        = False


def _sync_deck_from_file():
    global _flashcard_deck
    from flashcard import load_deck
    _flashcard_deck = load_deck()


def new_flashcard(topic: str):
    global _gen_current_card, _gen_revealed
    retriever = _active_retriever()
    if retriever is None:
        from flashcard import flashcard_to_html, Flashcard
        return flashcard_to_html(Flashcard("No retriever loaded.", ""), True), gr.update(visible=False), ""

    from flashcard import (generate_flashcard_with_pretrained,
                            generate_flashcard_with_model,
                            flashcard_to_html, append_card_to_deck, Flashcard)
    from generation import get_random_topic_from_retriever

    effective_topic = topic.strip() if topic.strip() else get_random_topic_from_retriever(retriever)

    if _pretrained_pipe is not None:
        card = generate_flashcard_with_pretrained(effective_topic, _pretrained_pipe, retriever)
    elif _pipeline is not None:
        card = generate_flashcard_with_model(effective_topic, _pipeline.model,
                                             _pipeline.tokenizer, retriever, _pipeline.device)
    else:
        return flashcard_to_html(Flashcard("No model loaded.", ""), True), gr.update(visible=False), ""

    if card is None:
        return (flashcard_to_html(Flashcard("Failed to generate -- try a different topic.", ""), True),
                gr.update(visible=False), "")

    append_card_to_deck(card)
    _sync_deck_from_file()
    _gen_current_card = card
    _gen_revealed     = False

    return (flashcard_to_html(card, revealed=False),
            gr.update(visible=True, value="Flip"),
            f"Saved to deck ({len(_flashcard_deck)} cards total)")


def flip_gen_card():
    global _gen_revealed
    if _gen_current_card is None:
        return "", gr.update()
    from flashcard import flashcard_to_html
    _gen_revealed = not _gen_revealed
    return (flashcard_to_html(_gen_current_card, revealed=_gen_revealed),
            gr.update(value="Hide" if _gen_revealed else "Flip"))


def load_deck_tab():
    global _deck_index
    _sync_deck_from_file()
    _deck_index = 0
    return _deck_card_display(False)


def _deck_card_display(revealed: bool):
    from flashcard import deck_card_html, Flashcard, flashcard_to_html
    if not _flashcard_deck:
        empty = Flashcard("Your deck is empty.",
                          "Generate some flashcards in the Generate tab first.")
        return (flashcard_to_html(empty, True), "0 cards",
                gr.update(interactive=False), gr.update(interactive=True),
                gr.update(interactive=False))
    card  = _flashcard_deck[_deck_index]
    html  = deck_card_html(card, _deck_index, len(_flashcard_deck), revealed)
    count = f"{len(_flashcard_deck)} card{'s' if len(_flashcard_deck) != 1 else ''}"
    return (html, count,
            gr.update(interactive=_deck_index > 0),
            gr.update(interactive=True),
            gr.update(interactive=_deck_index < len(_flashcard_deck) - 1))


def deck_flip():
    global _deck_revealed
    _deck_revealed = not _deck_revealed
    return _deck_card_display(_deck_revealed)


def deck_prev():
    global _deck_index, _deck_revealed
    _deck_index    = max(0, _deck_index - 1)
    _deck_revealed = False
    return _deck_card_display(False)


def deck_next():
    global _deck_index, _deck_revealed
    _deck_index    = min(len(_flashcard_deck) - 1, _deck_index + 1)
    _deck_revealed = False
    return _deck_card_display(False)


def export_deck():
    if not _flashcard_deck:
        return None
    from flashcard import export_to_anki_csv
    return export_to_anki_csv(_flashcard_deck, "flashcards_export.csv")


# ─────────────────────────────────────────────────────────────────────────────
# Tab 5: Progress
# ─────────────────────────────────────────────────────────────────────────────

_TRACKING_OFF_HTML = """
<div style="font-family:system-ui,sans-serif;background:#1a202c;border-radius:14px;
            padding:32px;color:#718096;text-align:center;max-width:600px;margin:12px auto;">
  <div style="font-size:32px;margin-bottom:12px;">&#128202;</div>
  <div style="font-size:16px;font-weight:600;color:#a0aec0;margin-bottom:8px;">
    Progress tracking is off
  </div>
  <div style="font-size:13px;line-height:1.7;">
    Start the app with
    <code style="background:#2d3748;padding:2px 6px;border-radius:4px;">--track</code>
    to enable curriculum tracking.<br><br>
    <code style="background:#2d3748;padding:6px 10px;border-radius:6px;display:inline-block;">
    python gradio_app.py --pretrained qwen-3b --index ../rag/rag_index --track
    </code>
  </div>
</div>
"""


def refresh_progress():
    if _tracker is None:
        return _TRACKING_OFF_HTML
    from progress_tracker import progress_to_html
    return progress_to_html(_tracker)


def reset_chapter_progress(chapter: str):
    if _tracker is None:
        return _TRACKING_OFF_HTML, "Tracking not enabled."
    ch = chapter if chapter and chapter != "-- All chapters --" else None
    _tracker.reset(chapter=ch)
    label = f"chapter '{ch}'" if ch else "all chapters"
    from progress_tracker import progress_to_html
    return progress_to_html(_tracker), f"Reset {label}."


# ─────────────────────────────────────────────────────────────────────────────
# UI helper
# ─────────────────────────────────────────────────────────────────────────────

def _chapter_choices() -> list:
    if _tracker is None:
        return ["-- All chapters --"]
    return ["-- All chapters --"] + _tracker.chapters()


# ─────────────────────────────────────────────────────────────────────────────
# Gradio UI
# ─────────────────────────────────────────────────────────────────────────────

def build_ui():
    tracking_on     = _tracker is not None
    chapter_choices = _chapter_choices()

    mode_note = (
        "**Tracking mode** — questions are chosen by curriculum priority, results saved to `progress.json`."
        if tracking_on else
        "**Random mode** — questions are drawn randomly. Start with `--track` to enable progress tracking."
    )

    with gr.Blocks(title="DAT255 Teaching Assistant") as demo:
        gr.Markdown(
            "# DAT255 Teaching Assistant\n"
            "Powered by a custom decoder-only transformer + RAG\n\n"
            + mode_note
        )

        with gr.Tabs():

            # ── Tab 1: Ask ────────────────────────────────────────────────────
            with gr.Tab("Ask a question"):
                question_box = gr.Textbox(label="Your question",
                                          placeholder="e.g. What is dropout and why does it help?",
                                          lines=2)
                with gr.Row():
                    use_rag = gr.Checkbox(label="Use RAG (recommended)", value=True)
                    temp    = gr.Slider(0.1, 1.5, value=0.7, step=0.1, label="Temperature")
                ask_btn     = gr.Button("Ask", variant="primary")
                answer_out  = gr.Textbox(label="Answer", lines=6)
                sources_out = gr.Markdown()

                ask_btn.click(fn=ask_question,
                              inputs=[question_box, use_rag, temp],
                              outputs=[answer_out, sources_out])

            # ── Tab 2: Quiz ───────────────────────────────────────────────────
            with gr.Tab("Quiz me"):
                gr.Markdown(
                    "Questions chosen by curriculum priority (unseen first, then weak topics). "
                    "Filter by chapter to stay focused on one area."
                    if tracking_on else
                    "Leave topic and chapter blank for a random question from the index."
                )
                with gr.Row():
                    quiz_chapter = gr.Dropdown(
                        choices=chapter_choices, value="-- All chapters --",
                        label="Chapter" + (" (tracking)" if tracking_on else ""),
                        scale=2,
                    )
                    quiz_topic = gr.Textbox(
                        label="Topic override (optional)",
                        placeholder="e.g. attention mechanism", scale=3,
                    )
                quiz_gen_btn  = gr.Button("Generate question", variant="primary")
                quiz_status   = gr.Markdown()
                q_display     = gr.Textbox(label="Question", lines=3, interactive=False)
                options_radio = gr.Radio(choices=[], label="Options", visible=False)
                submit_btn    = gr.Button("Submit answer", visible=False)
                feedback_md   = gr.Markdown()

                quiz_gen_btn.click(fn=new_quiz_question,
                                   inputs=[quiz_topic, quiz_chapter],
                                   outputs=[q_display, feedback_md, options_radio,
                                            submit_btn, quiz_status])
                submit_btn.click(fn=submit_quiz_answer,
                                 inputs=[options_radio], outputs=[feedback_md])

            # ── Tab 3: Practice answer ────────────────────────────────────────
            with gr.Tab("Practice answer"):
                gr.Markdown(
                    "The assistant picks the next highest-priority chunk and generates an "
                    "exam question from it. Write your answer and get a score + feedback. "
                    "Score >= 3/5 counts as correct in the tracker."
                    if tracking_on else
                    "The assistant generates an exam question from your course material. "
                    "Write your answer, then click **Get feedback**."
                )
                with gr.Row():
                    fb_chapter = gr.Dropdown(
                        choices=chapter_choices, value="-- All chapters --",
                        label="Chapter" + (" (tracking)" if tracking_on else ""),
                        scale=2,
                    )
                    fb_topic  = gr.Textbox(label="Topic override (optional)",
                                           placeholder="e.g. batch normalisation", scale=3)
                fb_gen_btn    = gr.Button("Generate question", variant="primary")
                fb_status     = gr.Markdown()
                fb_question   = gr.Textbox(
                    label="Question",
                    placeholder="Click 'Generate question' or type your own...",
                    lines=3, interactive=True)
                fb_answer     = gr.Textbox(label="Your answer",
                                           placeholder="Write your answer here...",
                                           lines=6, visible=False)
                fb_submit_btn = gr.Button("Get feedback", variant="primary", visible=False)
                fb_result     = gr.Markdown()

                fb_gen_btn.click(fn=generate_feedback_question,
                                 inputs=[fb_topic, fb_chapter],
                                 outputs=[fb_question, fb_answer, fb_submit_btn, fb_status])
                fb_question.change(
                    fn=lambda q: (gr.update(visible=bool(q.strip())),
                                  gr.update(visible=bool(q.strip()))),
                    inputs=[fb_question], outputs=[fb_answer, fb_submit_btn])
                fb_submit_btn.click(fn=review_student_answer,
                                    inputs=[fb_question, fb_answer],
                                    outputs=[fb_result])

            # ── Tab 4: Flashcards ─────────────────────────────────────────────
            with gr.Tab("Flashcards"):
                with gr.Tabs():
                    with gr.Tab("Generate"):
                        gr.Markdown("Generate a flashcard. The answer is pre-loaded but hidden.")
                        with gr.Row():
                            fc_topic   = gr.Textbox(label="Topic (optional)",
                                                    placeholder="e.g. transformer self-attention",
                                                    scale=4)
                            fc_gen_btn = gr.Button("Generate", variant="primary", scale=1)
                        fc_card_html = gr.HTML()
                        fc_flip_btn  = gr.Button("Flip", visible=False)
                        fc_status    = gr.Markdown()

                        fc_gen_btn.click(fn=new_flashcard, inputs=[fc_topic],
                                         outputs=[fc_card_html, fc_flip_btn, fc_status])
                        fc_flip_btn.click(fn=flip_gen_card,
                                          outputs=[fc_card_html, fc_flip_btn])

                    with gr.Tab("My Deck"):
                        gr.Markdown("Browse saved flashcards. Cards are auto-saved when generated.")
                        deck_load_btn = gr.Button("Load / Refresh deck")
                        deck_html     = gr.HTML()
                        deck_count_md = gr.Markdown()
                        with gr.Row():
                            deck_prev_btn = gr.Button("Previous", interactive=False)
                            deck_flip_btn = gr.Button("Flip")
                            deck_next_btn = gr.Button("Next", interactive=False)
                        with gr.Row():
                            deck_export_btn  = gr.Button("Export to Anki CSV")
                            deck_export_file = gr.File(label="Download CSV", visible=False)

                        _deck_out = [deck_html, deck_count_md,
                                     deck_prev_btn, deck_flip_btn, deck_next_btn]
                        deck_load_btn.click(fn=load_deck_tab,  outputs=_deck_out)
                        deck_flip_btn.click(fn=deck_flip,      outputs=_deck_out)
                        deck_prev_btn.click(fn=deck_prev,      outputs=_deck_out)
                        deck_next_btn.click(fn=deck_next,      outputs=_deck_out)
                        deck_export_btn.click(fn=export_deck,  outputs=[deck_export_file])

            # ── Tab 5: Progress ───────────────────────────────────────────────
            with gr.Tab("Progress"):
                gr.Markdown(
                    "Your curriculum coverage and accuracy, updated as you answer questions."
                    if tracking_on else
                    "Start the app with `--track` to enable the progress dashboard."
                )
                progress_html = gr.HTML(value=refresh_progress())
                refresh_btn   = gr.Button("Refresh", variant="secondary")
                refresh_btn.click(fn=refresh_progress, outputs=[progress_html])

                if tracking_on:
                    gr.Markdown("---")
                    with gr.Row():
                        reset_chapter_dd = gr.Dropdown(
                            choices=chapter_choices, value="-- All chapters --",
                            label="Reset progress for chapter", scale=3)
                        reset_btn = gr.Button("Reset progress", variant="stop", scale=1)
                    reset_status = gr.Markdown()
                    reset_btn.click(fn=reset_chapter_progress,
                                    inputs=[reset_chapter_dd],
                                    outputs=[progress_html, reset_status])

    return demo


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    parser.add_argument("--checkpoint", default="../model/checkpoints/step_005000.pt")
    parser.add_argument("--index",      default="../rag/rag_index")
    parser.add_argument("--port",       type=int, default=7860)
    parser.add_argument("--share",      action="store_true")
    parser.add_argument("--pretrained", default=None, metavar="MODEL",
                        help="HuggingFace model key (qwen-1.5b/qwen-3b/qwen-3b-4b/mistral-4b) "
                             "or full model ID. Append :bnb4 for 4-bit.")
    parser.add_argument("--track",      action="store_true",
                        help="Enable progress tracking. Saves results to --progress file.")
    parser.add_argument("--chunks",     default=None,
                        help="Path to chunks.json. Defaults to {index}/chunks.json.")
    parser.add_argument("--progress",   default="progress.json",
                        help="Path to progress save file (default: progress.json).")
    args = parser.parse_args()

    # Load model / retriever
    if Path(args.checkpoint).exists():
        load_pipeline(args.checkpoint, args.index)
    else:
        print(f"Checkpoint not found: {args.checkpoint}")
        print("Loading retriever only (pretrained generation mode).")
        load_retriever(args.index)

    if args.pretrained:
        load_pretrained(args.pretrained)

    # Load progress tracker
    if args.track:
        chunks_path = args.chunks or str(Path(args.index) / "chunks.json")
        if not Path(chunks_path).exists():
            print(f"[WARN] --track enabled but chunks.json not found at: {chunks_path}")
            print("       Pass --chunks <path> to specify its location.")
        else:
            from progress_tracker import ProgressTracker
            _tracker = ProgressTracker(chunks_path, args.progress)
            print(f"Progress tracking enabled. Saving to: {args.progress}")
    else:
        print("Random mode (no tracking). Pass --track to enable progress tracking.")

    ui = build_ui()
    ui.launch(server_port=args.port, share=args.share)