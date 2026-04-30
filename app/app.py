# app.py
"""
Gradio teaching assistant.

Tabs:
  1. Ask          — RAG-assisted free-form Q&A
  2. Quiz         — multiple-choice questions
  3. Practice     — long-answer with scoring feedback
  4. Flashcards   — generate + deck browser

Run:
  # Random mode (no tracking):
  python app.py --pretrained qwen-3b

  # LoRA-finetuned Qwen 3B:
  python app.py --pretrained qwen-3b --lora-path ../model/checkpoints/qwen_lora \
      --index ../rag/rag_index

  # Tracking mode:
  python app.py --pretrained qwen-3b --index ../rag/rag_index --track

  # Explicit chunks path:
  python app.py --pretrained qwen-3b --index ../rag/rag_index \\
      --track --chunks ../rag/rag_index/chunks.json --progress my_progress.json
"""

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import gradio as gr
import torch

from topic_tree import TopicTree, natural_sort
from rag.retriever import Retriever
from generation import get_random_chunk_in_scope, get_random_topic_from_retriever


# =============================================================================
# Global state
# =============================================================================

_pipeline        = None   # custom RAGPipeline (checkpoint mode)
_pretrained_pipe = None   # HuggingFace text-generation pipeline
_retriever       = None   # standalone Retriever (pretrained mode)
_tracker         = None   # ProgressTracker — None when tracking is off
_topic_tree      = None   # TopicTree — built from chunks.json when available

_current_quiz:          object = None
_current_quiz_chunk_id: int    = None
_current_fb_chunk_id:   int    = None


def _active_retriever() -> Retriever | None:
    if _pipeline is not None:
        return _pipeline.retriever
    if _retriever is not None:
        return _retriever
    return None


def _pipe_kwargs() -> dict:
    """Convenience: return the pipe/model/tokenizer/device kwargs for generation calls."""
    return dict(
        pipe      = _pretrained_pipe,
        model     = _pipeline.model     if _pipeline else None,
        tokenizer = _pipeline.tokenizer if _pipeline else None,
        device    = _pipeline.device    if _pipeline else "cpu",
    )


# =============================================================================
# Model / retriever loading
# =============================================================================

def load_pipeline(checkpoint: str, index_dir: str) -> None:
    global _pipeline
    from rag_pipeline import RAGPipeline
    _pipeline = RAGPipeline(checkpoint, index_dir, top_k=3)
    print("Custom pipeline loaded.")


def load_retriever(index_dir: str) -> None:
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


def load_pretrained(model_key_or_id: str, lora_path: str | None = None) -> None:
    global _pretrained_pipe
    from transformers import (pipeline, AutoModelForCausalLM, AutoTokenizer,
                               BitsAndBytesConfig)

    model_id = PRETRAINED_MODELS.get(model_key_or_id, model_key_or_id)
    use_gpu  = torch.cuda.is_available()

    if model_id.endswith(":bnb4"):
        real_id = model_id[:-5]
        try:
            print(f"Loading {real_id} (4-bit)...")
            bnb_cfg   = BitsAndBytesConfig(load_in_4bit=True,
                                           bnb_4bit_compute_dtype=torch.float16)
            tokenizer = AutoTokenizer.from_pretrained(real_id)
            model     = AutoModelForCausalLM.from_pretrained(
                            real_id, quantization_config=bnb_cfg, device_map="auto")
            if lora_path:
                from peft import PeftModel
                print(f"Applying LoRA adapter: {lora_path}")
                model = PeftModel.from_pretrained(model, lora_path)
                model = model.merge_and_unload()
            _pretrained_pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
            _clear_generation_config(_pretrained_pipe)
            print("Pretrained pipeline loaded (4-bit).")
            return
        except ImportError:
            print("[WARN] bitsandbytes not installed, falling back to fp16")
            model_id = real_id

    if lora_path:
        # Load model manually so we can apply the LoRA adapter before wrapping in pipeline
        dtype = torch.bfloat16 if use_gpu else torch.float32
        print(f"Loading {model_id} (bfloat16) for LoRA merge...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model     = AutoModelForCausalLM.from_pretrained(
                        model_id, torch_dtype=dtype, device_map="auto")
        from peft import PeftModel
        print(f"Applying LoRA adapter: {lora_path}")
        model = PeftModel.from_pretrained(model, lora_path)
        model = model.merge_and_unload()
        model.eval()
        _pretrained_pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
        _clear_generation_config(_pretrained_pipe)
        print("Pretrained pipeline loaded (LoRA merged).")
        return

    device = 0 if use_gpu else -1
    dtype  = torch.float16 if use_gpu else torch.float32
    print(f"Loading {model_id} on {'GPU' if use_gpu else 'CPU'}...")
    _pretrained_pipe = pipeline(
        "text-generation", model=model_id,
        device=device, torch_dtype=dtype, trust_remote_code=True,
    )
    _clear_generation_config(_pretrained_pipe)
    print("Pretrained pipeline loaded.")


def _clear_generation_config(pipe) -> None:
    try:
        pipe.model.generation_config.max_length = None
    except Exception as e:
        print(f"[WARN] Could not clear generation_config.max_length: {e}")


# =============================================================================
# Topic picker helpers
# =============================================================================

def _resolve_filter(source: str, chapter: str, section: str, topic_text: str):
    """
    Resolve the three-level picker + optional text override into a filter.
    Text override takes precedence over the picker.
    Returns (breadcrumb_prefix_or_None, topic_text_or_None).
    """
    if topic_text and topic_text.strip():
        return None, topic_text.strip()
    if _topic_tree is None:
        return None, None
    prefix = _topic_tree.breadcrumb_prefix(source, chapter, section)
    return (prefix or None), None


def _on_source_change(source_val: str):
    if not source_val or _topic_tree is None:
        return (gr.update(choices=[], value=None, visible=False),
                gr.update(choices=[], value=None, visible=False))
    chapters = _topic_tree.child_choices(source_val)
    return (gr.update(choices=chapters, value=None, visible=bool(chapters)),
            gr.update(choices=[], value=None, visible=False))


def _on_chapter_change(source_val: str, chapter_val: str):
    if not chapter_val or _topic_tree is None:
        return gr.update(choices=[], value=None, visible=False)
    sections = _topic_tree.child_choices(source_val, chapter_val)
    return gr.update(choices=sections, value=None, visible=bool(sections))


def _picker_initial_state() -> tuple[list, str | None, list]:
    if _topic_tree is None or _topic_tree.is_empty():
        return [], None, []
    sources        = _topic_tree.root_choices()
    default_source = _topic_tree.single_root()
    initial_chapters = (
        _topic_tree.child_choices(default_source) if default_source else []
    )
    return sources, default_source, initial_chapters


def _pick_random_chunk_and_topic(retriever, breadcrumb_prefix: str | None):
    """
    Pick a random chunk within the given breadcrumb scope and derive a topic label.
    Returns (prefetched_chunk_or_None, effective_topic_str).
    """
    chunk = get_random_chunk_in_scope(retriever, breadcrumb_prefix)
    if chunk:
        breadcrumb = chunk.get("metadata", {}).get("breadcrumb", "")
        topic = breadcrumb.split(" > ")[-1].strip() if breadcrumb else \
                get_random_topic_from_retriever(retriever)
    else:
        topic = get_random_topic_from_retriever(retriever)
    return chunk, topic


# =============================================================================
# Tab 1: Ask
# =============================================================================

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


# =============================================================================
# Tab 2: Quiz
# =============================================================================

def new_quiz_question(topic_text: str, source: str, chapter: str, section: str):
    print(f"new_quiz_question: {topic_text!r}, {source}, {chapter}, {section}")
    global _current_quiz, _current_quiz_chunk_id

    retriever = _active_retriever()
    if retriever is None:
        return ("No retriever loaded.", "",
                gr.update(choices=[]), gr.update(visible=False), "")
    if _pretrained_pipe is None and _pipeline is None:
        return ("No model loaded.", "",
                gr.update(choices=[]), gr.update(visible=False), "")

    from quiz import generate_quiz_question

    breadcrumb_prefix, topic_override = _resolve_filter(source, chapter, section, topic_text)
    prefetched_chunk = None

    if _tracker is not None:
        result = _tracker.next_chunk(breadcrumb_prefix=breadcrumb_prefix, topic=topic_override)
        if result is None:
            return ("No chunks found for that filter.", "",
                    gr.update(choices=[]), gr.update(visible=False), "")
        chunk_id, record       = result
        _current_quiz_chunk_id = chunk_id
        effective_topic        = record.topic
        status = (
            f"**{record.breadcrumb}**  |  "
            f"seen {record.times_seen}x  |  accuracy {record.accuracy:.0%}"
        )
    else:
        _current_quiz_chunk_id = None
        if topic_override:
            effective_topic = topic_override
        else:
            prefetched_chunk, effective_topic = _pick_random_chunk_and_topic(
                retriever, breadcrumb_prefix
            )
        status = f"Topic: *{effective_topic}*"

    print(f"effective_topic: {effective_topic!r}")

    _current_quiz = generate_quiz_question(
        topic            = effective_topic,
        retriever        = retriever,
        prefetched_chunk = prefetched_chunk,
        **_pipe_kwargs(),
    )

    if _current_quiz is None:
        return ("Failed to generate question — try a different topic.", "",
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
    return (
        f"Incorrect. The correct answer was {cl}) "
        f"{_current_quiz.correct_answer}\n\n{_current_quiz.explanation}"
    )


# =============================================================================
# Tab 3: Practice answer
# =============================================================================

def generate_feedback_question(topic_text: str, source: str, chapter: str, section: str):
    global _current_fb_chunk_id

    retriever = _active_retriever()
    if retriever is None:
        return (gr.update(value="No retriever loaded.", interactive=False),
                gr.update(visible=False), gr.update(), "")
    if _pretrained_pipe is None and _pipeline is None:
        return (gr.update(value="No model loaded.", interactive=False),
                gr.update(visible=False), gr.update(), "")

    from feedback import generate_question

    breadcrumb_prefix, topic_override = _resolve_filter(source, chapter, section, topic_text)
    prefetched_chunk = None

    if _tracker is not None:
        result = _tracker.next_chunk(breadcrumb_prefix=breadcrumb_prefix, topic=topic_override)
        if result:
            chunk_id, record     = result
            _current_fb_chunk_id = chunk_id
            effective_topic      = record.topic
            status = (
                f"**{record.breadcrumb}**  |  "
                f"seen {record.times_seen}x  |  accuracy {record.accuracy:.0%}"
            )
        else:
            _current_fb_chunk_id = None
            effective_topic      = topic_override or ""
            status               = "No chunks found for that filter."
    else:
        _current_fb_chunk_id = None
        if topic_override:
            effective_topic = topic_override
        else:
            prefetched_chunk, effective_topic = _pick_random_chunk_and_topic(
                retriever, breadcrumb_prefix
            )
        status = f"Topic: *{effective_topic}*" if effective_topic else ""

    question = generate_question(
        retriever        = retriever,
        topic            = effective_topic,
        prefetched_chunk = prefetched_chunk,
        **_pipe_kwargs(),
    )

    if not question:
        return (
            gr.update(value="Failed to generate a question — try entering a topic manually.",
                      interactive=True),
            gr.update(visible=False), gr.update(), status,
        )

    return (
        gr.update(value=question, interactive=True),
        gr.update(visible=True),
        gr.update(),   # submit button is always-on, no-op update
        status,
    )


def review_student_answer(question: str, student_answer: str):
    if not question.strip():
        return "Please generate or enter a question first."
    if not student_answer.strip():
        return "Please write your answer before submitting."

    retriever = _active_retriever()
    if _pretrained_pipe is None and _pipeline is None:
        return "No model loaded."

    from feedback import review_answer, parse_score_from_markdown
    formatted = review_answer(
        question       = question,
        student_answer = student_answer,
        retriever      = retriever,
        **_pipe_kwargs(),
    )

    if _tracker is not None and _current_fb_chunk_id is not None:
        from config import FEEDBACK_PASS_THRESHOLD
        _tracker.record(_current_fb_chunk_id,
                        correct=(parse_score_from_markdown(formatted) >= FEEDBACK_PASS_THRESHOLD))

    return formatted


# =============================================================================
# Tab 4: Flashcards
# =============================================================================

_flashcard_deck: list = []
_gen_current_card     = None
_gen_revealed         = False
_deck_index:  int     = 0
_deck_revealed        = False


def _sync_deck_from_file() -> None:
    global _flashcard_deck
    from flashcard import load_deck
    _flashcard_deck = load_deck()


def new_flashcard(topic_text: str, source: str, chapter: str, section: str):
    global _gen_current_card, _gen_revealed

    retriever = _active_retriever()
    if retriever is None:
        from flashcard import flashcard_to_html, Flashcard
        return flashcard_to_html(Flashcard("No retriever loaded.", ""), True), gr.update(visible=False), ""
    if _pretrained_pipe is None and _pipeline is None:
        from flashcard import flashcard_to_html, Flashcard
        return flashcard_to_html(Flashcard("No model loaded.", ""), True), gr.update(visible=False), ""

    from flashcard import generate_flashcard, flashcard_to_html, append_card_to_deck, Flashcard

    breadcrumb_prefix, topic_override = _resolve_filter(source, chapter, section, topic_text)
    prefetched_chunk = None

    if topic_override:
        effective_topic = topic_override
    else:
        prefetched_chunk, effective_topic = _pick_random_chunk_and_topic(
            retriever, breadcrumb_prefix
        )

    card = generate_flashcard(
        topic            = effective_topic,
        retriever        = retriever,
        prefetched_chunk = prefetched_chunk,
        **_pipe_kwargs(),
    )

    if card is None:
        return (
            flashcard_to_html(Flashcard("Failed to generate — try a different topic.", ""), True),
            gr.update(visible=False), "",
        )

    append_card_to_deck(card)
    _sync_deck_from_file()
    _gen_current_card = card
    _gen_revealed     = False

    return (
        flashcard_to_html(card, revealed=False),
        gr.update(visible=True, value="Flip"),
        f"Saved to deck ({len(_flashcard_deck)} cards total)",
    )


def flip_gen_card():
    global _gen_revealed
    if _gen_current_card is None:
        return "", gr.update()
    from flashcard import flashcard_to_html
    _gen_revealed = not _gen_revealed
    return (
        flashcard_to_html(_gen_current_card, revealed=_gen_revealed),
        gr.update(value="Hide" if _gen_revealed else "Flip"),
    )


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


# =============================================================================
# Tab 5: Progress
# =============================================================================

_TRACKING_OFF_HTML = """
<div style="font-family:system-ui,sans-serif;background:#1a202c;border-radius:14px;
            padding:32px;color:#718096;text-align:center;max-width:600px;margin:12px auto;">
  <div style="font-size:16px;font-weight:600;color:#a0aec0;margin-bottom:8px;">
    Progress tracking is off
  </div>
  <div style="font-size:13px;line-height:1.7;">
    Start the app with
    <code style="background:#2d3748;padding:2px 6px;border-radius:4px;">--track</code>
    to enable curriculum tracking.<br><br>
    <code style="background:#2d3748;padding:6px 10px;border-radius:6px;display:inline-block;">
    python app.py --pretrained qwen-3b --index ../rag/rag_index --track
    </code>
  </div>
</div>
"""


def refresh_progress():
    if _tracker is None:
        return _TRACKING_OFF_HTML
    from tracker import progress_to_html
    return progress_to_html(_tracker)


def reset_progress(source: str, chapter: str, section: str):
    if _tracker is None:
        return _TRACKING_OFF_HTML, "Tracking not enabled."
    prefix = _topic_tree.breadcrumb_prefix(source, chapter, section) if _topic_tree else None
    _tracker.reset(breadcrumb_prefix=prefix or None)
    label = f"'{prefix}'" if prefix else "everything"
    from tracker import progress_to_html
    return progress_to_html(_tracker), f"Reset {label}."


# =============================================================================
# UI construction
# =============================================================================

def _make_topic_picker(show_topic_override: bool = True):
    """
    Build the three-level cascading source/chapter/section dropdowns.
    Returns (source_dd, chapter_dd, section_dd, topic_txt).
    """
    sources, default_source, initial_chapters = _picker_initial_state()
    # print(f"sources: {sources}  default: {default_source!r}  chapters: {len(initial_chapters)}")

    with gr.Row():
        source_dd = gr.Dropdown(
            choices=sources, value=default_source,
            label="Source", scale=2, visible=True,
        )
        chapter_dd = gr.Dropdown(
            choices=initial_chapters, value=None,
            label="Chapter", scale=2, visible=bool(initial_chapters),
        )
        section_dd = gr.Dropdown(
            choices=[], value=None,
            label="Section", scale=2, visible=False,
        )
        topic_txt = gr.Textbox(
            label="Topic override (optional)",
            placeholder="e.g. self-attention mechanism",
            scale=3, visible=show_topic_override,
        )

    return source_dd, chapter_dd, section_dd, topic_txt


def _wire_picker(source_dd, chapter_dd, section_dd) -> None:
    source_dd.change(
        fn=_on_source_change, inputs=[source_dd],
        outputs=[chapter_dd, section_dd], show_progress=False,
    )
    chapter_dd.change(
        fn=_on_chapter_change, inputs=[source_dd, chapter_dd],
        outputs=[section_dd], trigger_mode="always_last", show_progress=False,
    )


def build_ui():
    tracking_on = _tracker is not None
    mode_note = (
        "**Tracking mode** — questions chosen by curriculum priority, results saved."
        if tracking_on else
        "**Random mode** — questions drawn randomly. Pass `--track` to enable progress tracking."
    )

    with gr.Blocks(title="Teaching Assistant") as demo:
        gr.Markdown("# Teaching Assistant\n" + mode_note)

        with gr.Tabs():

            # ── Tab 1: Ask ────────────────────────────────────────────────────
            with gr.Tab("Ask"):
                question_box = gr.Textbox(
                    label="Your question",
                    placeholder="e.g. What is dropout and why does it help?",
                    lines=2,
                )
                with gr.Row():
                    use_rag = gr.Checkbox(label="Use RAG (recommended)", value=True)
                    temp    = gr.Slider(0.1, 1.5, value=0.7, step=0.1, label="Temperature")
                ask_btn    = gr.Button("Ask", variant="primary")
                answer_out = gr.Textbox(label="Answer", lines=6)
                sources_md = gr.Markdown()

                ask_btn.click(fn=ask_question,
                              inputs=[question_box, use_rag, temp],
                              outputs=[answer_out, sources_md])

            # ── Tab 2: Quiz ───────────────────────────────────────────────────
            with gr.Tab("Quiz"):
                gr.Markdown(
                    "Questions chosen by curriculum priority. "
                    "Filter by chapter or section to focus on one area."
                    if tracking_on else
                    "Leave all filters blank for a random question from the index."
                )

                gr.Markdown("### 📚 Generate from source")
                q_source, q_chapter, q_section, _ = _make_topic_picker(show_topic_override=False)
                _wire_picker(q_source, q_chapter, q_section)
                section_btn = gr.Button("Generate from section", variant="primary")

                gr.Markdown("---")

                gr.Markdown("### ✏️ Generate from topic")
                q_topic = gr.Textbox(
                    label="Topic", placeholder="e.g. self-attention mechanism", lines=1,
                )
                topic_btn = gr.Button("Generate from topic", variant="secondary")

                gr.Markdown("---")
                quiz_status   = gr.Markdown()
                q_display     = gr.Textbox(label="Question", lines=3, interactive=False)
                options_radio = gr.Radio(choices=[], label="Options", visible=False)
                submit_btn    = gr.Button("Submit answer", visible=False)
                feedback_md   = gr.Markdown()

                section_btn.click(
                    fn      = new_quiz_question,
                    inputs  = [gr.State(""), q_source, q_chapter, q_section],
                    outputs = [q_display, feedback_md, options_radio, submit_btn, quiz_status],
                )
                topic_btn.click(
                    fn      = new_quiz_question,
                    inputs  = [q_topic, gr.State(None), gr.State(None), gr.State(None)],
                    outputs = [q_display, feedback_md, options_radio, submit_btn, quiz_status],
                )
                submit_btn.click(
                    fn=submit_quiz_answer, inputs=[options_radio], outputs=[feedback_md],
                )

            # ── Tab 3: Long answer ────────────────────────────────────────────
            with gr.Tab("Long answer"):
                gr.Markdown(
                    "The assistant picks the next high-priority chunk and generates a "
                    "long answer question. Write your answer and get a score + feedback. "
                    "Score >= 3/5 counts as correct in the tracker."
                    if tracking_on else
                    "The assistant generates a long answer question from your course material. "
                    "Write your answer, then click **Get feedback**."
                )

                gr.Markdown("### 📚 Generate from source")
                fb_source, fb_chapter, fb_section, _ = _make_topic_picker(show_topic_override=False)
                _wire_picker(fb_source, fb_chapter, fb_section)
                fb_section_btn = gr.Button("Generate from section", variant="primary")

                gr.Markdown("---")

                gr.Markdown("### ✏️ Generate from topic")
                fb_topic = gr.Textbox(
                    label="Topic", placeholder="e.g. backpropagation through time", lines=1,
                )
                fb_topic_btn = gr.Button("Generate from topic", variant="secondary")

                gr.Markdown("---")
                fb_status     = gr.Markdown()
                fb_question   = gr.Textbox(
                    label="Question",
                    placeholder="Click a generate button or type your own question...",
                    lines=3, interactive=True,
                )
                fb_answer     = gr.Textbox(
                    label="Your answer", placeholder="Write your answer here...",
                    lines=6, visible=False,
                )
                fb_submit_btn = gr.Button("Get feedback", variant="primary",
                                          visible=True, interactive=True)
                fb_result     = gr.Markdown()

                fb_section_btn.click(
                    fn      = generate_feedback_question,
                    inputs  = [gr.State(""), fb_source, fb_chapter, fb_section],
                    outputs = [fb_question, fb_answer, fb_submit_btn, fb_status],
                )
                fb_topic_btn.click(
                    fn      = generate_feedback_question,
                    inputs  = [fb_topic, gr.State(None), gr.State(None), gr.State(None)],
                    outputs = [fb_question, fb_answer, fb_submit_btn, fb_status],
                )
                fb_question.change(
                    fn      = lambda q: (gr.update(visible=bool(q.strip())),
                                         gr.update(interactive=bool(q.strip()))),
                    inputs  = [fb_question],
                    outputs = [fb_answer, fb_submit_btn],
                )
                fb_submit_btn.click(
                    fn=review_student_answer, inputs=[fb_question, fb_answer], outputs=[fb_result],
                )

            # ── Tab 4: Flashcards ─────────────────────────────────────────────
            with gr.Tab("Flashcards"):
                with gr.Tabs():
                    with gr.Tab("Generate"):
                        gr.Markdown("Generate a flashcard. The answer is pre-loaded but hidden.")

                        gr.Markdown("### 📚 Generate from source")
                        fc_source, fc_chapter, fc_section, _ = _make_topic_picker(
                            show_topic_override=False
                        )
                        _wire_picker(fc_source, fc_chapter, fc_section)
                        fc_section_btn = gr.Button("Generate from section", variant="primary")

                        gr.Markdown("---")

                        gr.Markdown("### ✏️ Generate from topic")
                        fc_topic = gr.Textbox(
                            label="Topic", placeholder="e.g. gradient vanishing problem", lines=1,
                        )
                        fc_topic_btn = gr.Button("Generate from topic", variant="secondary")

                        gr.Markdown("---")
                        fc_card_html = gr.HTML()
                        fc_flip_btn  = gr.Button("Flip", visible=False)
                        fc_status    = gr.Markdown()

                        fc_section_btn.click(
                            fn      = new_flashcard,
                            inputs  = [gr.State(""), fc_source, fc_chapter, fc_section],
                            outputs = [fc_card_html, fc_flip_btn, fc_status],
                        )
                        fc_topic_btn.click(
                            fn      = new_flashcard,
                            inputs  = [fc_topic, gr.State(None), gr.State(None), gr.State(None)],
                            outputs = [fc_card_html, fc_flip_btn, fc_status],
                        )
                        fc_flip_btn.click(fn=flip_gen_card, outputs=[fc_card_html, fc_flip_btn])

                    with gr.Tab("My Deck"):
                        gr.Markdown("Browse saved flashcards.")
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
                    "Curriculum coverage and accuracy, updated as you answer questions."
                    if tracking_on else
                    "Start the app with `--track` to enable the progress dashboard."
                )
                progress_html = gr.HTML(value=refresh_progress())
                refresh_btn   = gr.Button("Refresh", variant="secondary")
                refresh_btn.click(fn=refresh_progress, outputs=[progress_html])

                if tracking_on:
                    gr.Markdown("---\n**Reset progress**")
                    p_source, p_chapter, p_section, _ = _make_topic_picker(
                        show_topic_override=False
                    )
                    _wire_picker(p_source, p_chapter, p_section)
                    reset_btn    = gr.Button("Reset selected", variant="stop")
                    reset_status = gr.Markdown()
                    reset_btn.click(
                        fn      = reset_progress,
                        inputs  = [p_source, p_chapter, p_section],
                        outputs = [progress_html, reset_status],
                    )

    return demo


# =============================================================================
# Entry point
# =============================================================================

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
                        help="Model key or full HuggingFace model ID. Append :bnb4 for 4-bit.")
    parser.add_argument("--lora-path", default=None, metavar="PATH",
                        help="Path to a LoRA adapter directory (with adapter_config.json). "
                             "Only used together with --pretrained.")
    parser.add_argument("--track",      action="store_true",
                        help="Enable progress tracking.")
    parser.add_argument("--chunks",     default=None,
                        help="Path to chunks.json. Defaults to {index}/chunks.json.")
    parser.add_argument("--progress",   default="progress.json",
                        help="Progress save file (default: progress.json).")
    args = parser.parse_args()

    # Load retriever / model
    if Path(args.checkpoint).exists():
        load_pipeline(args.checkpoint, args.index)
    else:
        print(f"Checkpoint not found: {args.checkpoint}")
        print("Loading retriever only (pretrained generation mode).")
        load_retriever(args.index)

    if args.pretrained:
        load_pretrained(args.pretrained, lora_path=args.lora_path)

    # Load chunks for topic tree (independent of --track)
    chunks_path = args.chunks or str(Path(args.index) / "chunks.json")
    if Path(chunks_path).exists():
        import json
        with open(chunks_path, encoding="utf-8") as f:
            _raw_chunks = json.load(f)
        _topic_tree = TopicTree(_raw_chunks)
        roots       = _topic_tree.root_choices()
        print(f"Topic tree: {len(roots)} source(s) from {chunks_path}")
        if _topic_tree.single_root():
            chapters = _topic_tree.child_choices(roots[0])
            print(f"  Auto-selected source: {roots[0]!r} ({len(chapters)} chapters)")
    else:
        print(f"[INFO] chunks.json not found at {chunks_path} — chapter picker disabled.")
        print("       Pass --chunks <path> to enable it.")

    # Load progress tracker
    if args.track:
        if not Path(chunks_path).exists():
            print(f"[WARN] --track enabled but chunks.json not found at: {chunks_path}")
        else:
            from tracker import ProgressTracker
            _tracker = ProgressTracker(chunks_path, args.progress)
            print(f"Progress tracking enabled. Saving to: {args.progress}")
    else:
        print("Random mode. Pass --track to enable progress tracking.")

    ui = build_ui()
    ui.launch(server_port=args.port, share=args.share)