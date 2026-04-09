"""
app.py  —  Gradio web interface for the Teaching Assistant
===========================================================
Two tabs:
  1. "Ask a question"  — RAG-assisted answer, shows retrieved sources
  2. "Quiz me"         — generates and checks multiple-choice questions

Run with:
  python app.py --checkpoint checkpoints/step_005000.pt --index rag_index/

Deploy to HuggingFace Spaces:
  1. Push this repo to GitHub
  2. Create a new Space on huggingface.co (Gradio SDK)
  3. Set the checkpoint + index in the Space's Files or as secrets
"""

import argparse
from pathlib import Path
import gradio as gr


# ─────────────────────────────────────────────────────────────────────────────
# Global state (loaded once at startup)
# ─────────────────────────────────────────────────────────────────────────────

_pipeline = None
_session  = None


def load_pipeline(checkpoint: str, index_dir: str):
    global _pipeline, _session
    from rag_pipeline import RAGPipeline
    from quiz import QuizSession
    _pipeline = RAGPipeline(checkpoint, index_dir, top_k=3)
    _session  = QuizSession(_pipeline)
    print("Pipeline loaded.")


# ─────────────────────────────────────────────────────────────────────────────
# Tab 1: Ask a question
# ─────────────────────────────────────────────────────────────────────────────

def ask_question(question: str, use_rag: bool, temperature: float):
    if not question.strip():
        return "Please enter a question.", ""

    if _pipeline is None:
        return "Model not loaded yet.", ""

    if use_rag:
        result = _pipeline.answer(question, temperature=temperature, max_tokens=250)
        answer = result["answer"]
        sources = "\n\n".join(
            f"**Source {i+1}** (score={c['score']:.2f}): {c['breadcrumb']}\n{c['text'][:300]}..."
            for i, c in enumerate(result["chunks"])
        )
        sources_md = f"### Retrieved context\n{sources}"
    else:
        from rag_pipeline import answer_question
        answer = answer_question(question, _pipeline.model, _pipeline.tokenizer,
                                  _pipeline.device, temperature=temperature)
        sources_md = "_No RAG used — model answered from training data only._"

    return answer, sources_md


# ─────────────────────────────────────────────────────────────────────────────
# Tab 2: Quiz
# ─────────────────────────────────────────────────────────────────────────────

_current_quiz = None


def new_quiz_question(topic: str):
    global _current_quiz
    if _pipeline is None:
        return "Model not loaded.", "", gr.update(choices=[]), gr.update(visible=False)

    from quiz import generate_quiz_question_with_model
    _current_quiz = generate_quiz_question_with_model(
        topic=topic or "machine learning",
        model=_pipeline.model,
        tokenizer=_pipeline.tokenizer,
        retriever=_pipeline.retriever,
        device=_pipeline.device,
    )

    if _current_quiz is None:
        return "Failed to generate question. Try a different topic.", "", \
               gr.update(choices=[]), gr.update(visible=False)

    choices = [f"{l}) {o}" for l, o in zip(["A","B","C","D"], _current_quiz.options)]
    return (
        _current_quiz.question,
        "",   # clear feedback
        gr.update(choices=choices, value=None, visible=True),
        gr.update(visible=True),  # submit button
    )


def submit_quiz_answer(selected: str):
    global _current_quiz
    if _current_quiz is None or selected is None:
        return "Please select an answer first."

    letter = selected[0]  # "A", "B", "C", or "D"
    idx    = "ABCD".index(letter)
    correct = idx == _current_quiz.correct_index

    if correct:
        feedback = f"✓ **Correct!**\n\n{_current_quiz.explanation}"
    else:
        correct_letter = "ABCD"[_current_quiz.correct_index]
        feedback = (f"✗ **Wrong.** The correct answer was **{correct_letter})**"
                    f" {_current_quiz.correct_answer}\n\n{_current_quiz.explanation}")

    return feedback


# ─────────────────────────────────────────────────────────────────────────────
# Gradio UI
# ─────────────────────────────────────────────────────────────────────────────

def build_ui():
    with gr.Blocks(title="DAT255 Teaching Assistant") as demo:
        gr.Markdown("# DAT255 Teaching Assistant\nPowered by a custom decoder-only transformer + RAG")

        with gr.Tabs():
            # ── Tab 1 ─────────────────────────────────────────────────────────
            with gr.Tab("Ask a question"):
                question_box = gr.Textbox(
                    label="Your question",
                    placeholder="e.g. What is dropout and why does it help?",
                    lines=2,
                )
                with gr.Row():
                    use_rag = gr.Checkbox(label="Use RAG (recommended)", value=True)
                    temp    = gr.Slider(0.1, 1.5, value=0.7, step=0.1, label="Temperature")
                ask_btn  = gr.Button("Ask", variant="primary")
                answer_out  = gr.Textbox(label="Answer", lines=6)
                sources_out = gr.Markdown()

                ask_btn.click(
                    fn=ask_question,
                    inputs=[question_box, use_rag, temp],
                    outputs=[answer_out, sources_out],
                )

            # ── Tab 2 ─────────────────────────────────────────────────────────
            with gr.Tab("Quiz me"):
                topic_box   = gr.Textbox(label="Topic (optional)", placeholder="e.g. attention mechanism")
                gen_btn     = gr.Button("Generate question", variant="primary")
                q_display   = gr.Textbox(label="Question", lines=3, interactive=False)
                options_radio = gr.Radio(choices=[], label="Options", visible=False)
                submit_btn  = gr.Button("Submit answer", visible=False)
                feedback_md = gr.Markdown()

                gen_btn.click(
                    fn=new_quiz_question,
                    inputs=[topic_box],
                    outputs=[q_display, feedback_md, options_radio, submit_btn],
                )
                submit_btn.click(
                    fn=submit_quiz_answer,
                    inputs=[options_radio],
                    outputs=[feedback_md],
                )

    return demo


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/step_005000.pt")
    parser.add_argument("--index",      default="rag_index")
    parser.add_argument("--port",       type=int, default=7860)
    parser.add_argument("--share",      action="store_true")
    args = parser.parse_args()

    if not Path(args.checkpoint).exists():
        print(f"Checkpoint not found: {args.checkpoint}")
        print("Run train.py first, then come back.")
    else:
        load_pipeline(args.checkpoint, args.index)

    ui = build_ui()
    ui.launch(server_port=args.port, share=args.share)
