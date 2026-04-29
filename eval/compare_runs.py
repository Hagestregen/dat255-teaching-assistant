"""
compare_runs.py  —  Aggregate every training run + eval result into one table
=============================================================================
Reads the JSON files this project already writes (no new dependencies):

  runs/curves_<NAME>.json             ← train.py --log-curves
  eval/results/<NAME>/summary.json    ← eval/run_eval.py

and prints a Markdown table you can paste straight into the project report.

Usage
-----
    python eval/compare_runs.py                       # all runs
    python eval/compare_runs.py --out docs/runs.md    # write to a file
    python eval/compare_runs.py --filter gpt2         # only matching names

Why not MLflow / W&B?
---------------------
This project already persists everything we need (loss curves + eval scores)
as JSON.  For a single-author / short-timeline project a markdown registry
hits the sweet spot of "enough to write the report" without running an
external server.  For richer dashboards, train.py supports `--use-wandb`
(W&B Cloud) out of the box — flip that flag and you get the same curves
streamed live.  See the README of this script for guidance.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Optional

ROOT       = Path(__file__).resolve().parent.parent
RUNS_DIR   = ROOT / "runs"
EVAL_DIR   = ROOT / "eval" / "results"


def _slope(records: list, key: str) -> Optional[float]:
    if len(records) < 4:
        return None
    tail = records[-max(4, len(records) // 4):]
    xs = [r["step"] for r in tail]
    ys = [r[key]    for r in tail]
    n  = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    den = sum((xs[i] - mx) ** 2 for i in range(n))
    return num / den if den else None


def _diagnose(curves: dict) -> str:
    train = curves.get("train", [])
    val   = curves.get("val",   [])
    if len(train) < 2 or len(val) < 2:
        return "n/a"
    best_val   = min(r["val_loss"] for r in val)
    final_val  = val[-1]["val_loss"]
    train_slp  = _slope(train, "loss")
    val_slp    = _slope(val,   "val_loss")
    val_rose   = final_val > best_val + 0.02
    train_down = train_slp is not None and train_slp < -1e-5
    val_flat   = val_slp   is None or val_slp >= -1e-6
    if val_rose:                       return "overfitting"
    if train_down and val_flat:        return "underfitting"
    if train_down and not val_flat:    return "still-learning"
    return "sweet-spot"


def _curve_summary(curves_path: Path) -> dict:
    with open(curves_path, encoding="utf-8") as f:
        c = json.load(f)
    train = c.get("train", [])
    val   = c.get("val",   [])
    summary = {
        "first_train_loss": train[0]["loss"] if train else None,
        "final_train_loss": train[-1]["loss"] if train else None,
        "best_val_loss":    min((r["val_loss"] for r in val), default=None),
        "best_val_step":    min(val, key=lambda r: r["val_loss"])["step"] if val else None,
        "final_val_loss":   val[-1]["val_loss"] if val else None,
        "best_ema_val_loss": min(
            (r.get("ema_val_loss") for r in val if r.get("ema_val_loss") is not None),
            default=None,
        ),
        "n_train_records":  len(train),
        "n_val_records":    len(val),
        "verdict":          _diagnose(c),
    }
    return summary


def _match_run(eval_dirname: str, run_names: list[str]) -> Optional[tuple[str, str]]:
    """
    Pick the *longest* run name that owns this eval result directory.

    With runs named e.g. "gpt2_warm" and "gpt2_warm_v2", a directory called
    "gpt2_warm_v2_live" must be assigned to "gpt2_warm_v2" (variant "live"),
    not to "gpt2_warm" (variant "v2_live").
    """
    candidates: list[tuple[str, str]] = []  # (matched_run, variant)
    for name in run_names:
        if eval_dirname == name or eval_dirname == f"run_{name}":
            candidates.append((name, "main"))
        elif eval_dirname.startswith(f"{name}_"):
            candidates.append((name, eval_dirname[len(name) + 1:]))
        elif eval_dirname.startswith(f"run_{name}_"):
            candidates.append((name, eval_dirname[len(f"run_{name}_"):]))
    if not candidates:
        return None
    # Prefer the candidate whose run name is the longest (most specific).
    candidates.sort(key=lambda t: len(t[0]), reverse=True)
    return candidates[0]


def _all_eval_summaries(run_names: list[str]) -> dict[str, list[dict]]:
    """
    Walk eval/results once and return {run_name: [summary, ...]}.
    Each summary dict carries a "variant" key.
    """
    out: dict[str, list[dict]] = {n: [] for n in run_names}
    if not EVAL_DIR.exists():
        return out
    for d in sorted(EVAL_DIR.iterdir()):
        if not d.is_dir():
            continue
        match = _match_run(d.name, run_names)
        if match is None:
            continue
        run_name, variant = match
        summary_path = d / "summary.json"
        if not summary_path.exists():
            continue
        with open(summary_path, encoding="utf-8") as f:
            s = json.load(f)
        m = s.get("metrics_mean", {})
        out[run_name].append({
            "variant":      variant,
            "checkpoint":   s.get("checkpoint"),
            "n_examples":   s.get("n_examples"),
            "with_context": s.get("with_context"),
            "exact_match":  m.get("exact_match"),
            "token_f1":     m.get("token_f1"),
            "rouge_l":      m.get("rouge_l"),
            "meteor":       m.get("meteor"),
            "bertscore_f1": m.get("bertscore_f1"),
            "semantic_sim": m.get("semantic_sim_mean"),
        })
    return out


def _fmt(x, fmt: str = ".3f") -> str:
    if x is None:
        return "—"
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return "—"
    return format(x, fmt)


def collect(filter_substr: Optional[str] = None) -> list[dict]:
    if not RUNS_DIR.exists():
        return []
    all_curves = [
        (p, p.stem.replace("curves_", ""))
        for p in sorted(RUNS_DIR.glob("curves_*.json"))
    ]
    if filter_substr:
        all_curves = [(p, n) for p, n in all_curves if filter_substr in n]
    run_names  = [n for _, n in all_curves]
    eval_index = _all_eval_summaries(run_names)
    rows: list[dict] = []
    for curves_path, name in all_curves:
        rows.append({
            "name": name,
            **_curve_summary(curves_path),
            "evals": eval_index.get(name, []),
        })
    return rows


def render_markdown(rows: list[dict]) -> str:
    if not rows:
        return "_No runs found in runs/curves_*.json yet._\n"

    out = []
    out.append("# Experiment registry")
    out.append("")
    out.append("Auto-generated by `python eval/compare_runs.py`. "
               "Lower val loss = better; lower train loss with rising val loss = overfitting.")
    out.append("")
    out.append("## Training curves")
    out.append("")
    out.append("| run | first train | final train | best val (step) | best EMA val | final val | verdict |")
    out.append("| --- | --- | --- | --- | --- | --- | --- |")
    for r in rows:
        best_val_cell = f"{_fmt(r['best_val_loss'])} ({r['best_val_step']})" if r['best_val_loss'] else "—"
        out.append(
            f"| `{r['name']}` "
            f"| {_fmt(r['first_train_loss'])} "
            f"| {_fmt(r['final_train_loss'])} "
            f"| {best_val_cell} "
            f"| {_fmt(r['best_ema_val_loss'])} "
            f"| {_fmt(r['final_val_loss'])} "
            f"| {r['verdict']} |"
        )
    out.append("")
    out.append("## Held-out eval (run_eval.py)")
    out.append("")
    out.append("Metrics on the frozen 381-example test split.  EM = exact match, "
               "F1 = token-level F1, ROUGE-L = longest-common-subsequence F1, "
               "METEOR = synonym-aware n-gram, SemSim = mean cosine similarity "
               "from `sentence-transformers/all-MiniLM-L6-v2`, BERTScore-F1 = "
               "token-aligned cosine similarity from `roberta-large`.")
    out.append("")
    out.append("| run | variant | n | with_ctx | EM | F1 | ROUGE-L | METEOR | SemSim | BERTScore-F1 |")
    out.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in rows:
        evs = r["evals"]
        if not evs:
            out.append(f"| `{r['name']}` | _none_ | | | | | | | | |")
            continue
        for e in evs:
            out.append(
                f"| `{r['name']}` "
                f"| {e['variant']} "
                f"| {e.get('n_examples', '—')} "
                f"| {e.get('with_context', '—')} "
                f"| {_fmt(e.get('exact_match'))} "
                f"| {_fmt(e.get('token_f1'))} "
                f"| {_fmt(e.get('rouge_l'))} "
                f"| {_fmt(e.get('meteor'))} "
                f"| {_fmt(e.get('semantic_sim'))} "
                f"| {_fmt(e.get('bertscore_f1'))} |"
            )
    out.append("")
    return "\n".join(out)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--filter", help="Only include runs whose name contains this string")
    p.add_argument("--out",    help="Write Markdown to this path instead of stdout")
    args = p.parse_args()

    rows = collect(args.filter)
    md   = render_markdown(rows)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md, encoding="utf-8")
        print(f"Wrote {out}")
    else:
        print(md)


if __name__ == "__main__":
    main()
