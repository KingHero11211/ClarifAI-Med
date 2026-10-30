"""
evaluation/run_evaluation.py

Batch evaluation pipeline for ClarifAI-Med.

Runs a held-out set of questions through:
  (a) Plain RAG (no NLI verification)
  (b) Full verified RAG (with NLI)

Computes hallucination rate before and after verification, outputs a results
CSV and comparison chart to evaluation/results/.

Usage:
    python evaluation/run_evaluation.py [--questions data/test_questions.json]
                                        [--n 50]
                                        [--out evaluation/results/]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger
from tqdm import tqdm

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_settings
from src.pipeline.rag_pipeline import answer_question
from evaluation.metrics import hallucination_rate, aggregate_hallucination_rates

settings = get_settings()

RESULTS_DIR = Path("evaluation/results")


def load_test_questions(path: str, n: int | None = None) -> list[dict]:
    """Load test questions from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    if n:
        questions = questions[:n]

    logger.info(f"Loaded {len(questions)} test questions from {path}")
    return questions


def run_single(question: str, skip_verification: bool) -> dict:
    """Run the pipeline on one question and return metrics."""
    try:
        result = answer_question(question, skip_verification=skip_verification)
        labels = [s["support_label"] for s in result.sentences]
        h_rate = hallucination_rate(labels)
        return {
            "question": question,
            "answer": result.answer_text,
            "sentence_labels": labels,
            "hallucination_rate": h_rate,
            "groundedness_score": result.overall_groundedness_score,
            "n_sentences": len(labels),
            "latency_seconds": result.latency_seconds,
            "error": None,
        }
    except Exception as e:
        logger.error(f"Failed on question '{question[:60]}': {e}")
        return {
            "question": question,
            "answer": "",
            "sentence_labels": [],
            "hallucination_rate": None,
            "groundedness_score": None,
            "n_sentences": 0,
            "latency_seconds": 0.0,
            "error": str(e),
        }


def run_evaluation(questions: list[dict], out_dir: Path) -> None:
    """
    Evaluate plain RAG vs. verified RAG on all test questions.

    Args:
        questions: List of question dicts (must have a "question" key).
        out_dir: Directory to save results CSV and charts.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    plain_results = []
    verified_results = []

    for item in tqdm(questions, desc="Evaluating"):
        q = item["question"] if isinstance(item, dict) else item

        logger.info(f"Question: {q[:80]}")

        # (a) Plain RAG
        plain = run_single(q, skip_verification=True)
        plain["mode"] = "plain_rag"
        plain_results.append(plain)

        # (b) Verified RAG
        verified = run_single(q, skip_verification=False)
        verified["mode"] = "verified_rag"
        verified_results.append(verified)

    # ── Save raw results ──────────────────────────────────────────────────────
    all_results = plain_results + verified_results
    df = pd.DataFrame(all_results)
    csv_path = out_dir / "evaluation_results.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Results saved to {csv_path}")

    # ── Compute headline numbers ───────────────────────────────────────────────
    plain_valid = [r for r in plain_results if r["hallucination_rate"] is not None]
    verified_valid = [r for r in verified_results if r["hallucination_rate"] is not None]

    plain_agg = aggregate_hallucination_rates(plain_valid)
    verified_agg = aggregate_hallucination_rates(verified_valid)

    summary = {
        "plain_rag": plain_agg,
        "verified_rag": verified_agg,
        "improvement_pp": round(
            (plain_agg["mean"] - verified_agg["mean"]) * 100, 2
        ),
    }

    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("=== EVALUATION RESULTS ===")
    logger.info(f"Plain RAG   hallucination rate: {plain_agg['mean']:.1%}")
    logger.info(f"Verified RAG hallucination rate: {verified_agg['mean']:.1%}")
    logger.info(f"Improvement: {summary['improvement_pp']:+.1f} percentage points")

    # ── Generate comparison chart ─────────────────────────────────────────────
    _plot_comparison(plain_agg, verified_agg, out_dir)
    _plot_distribution(plain_valid, verified_valid, out_dir)

    logger.info(f"Charts saved to {out_dir}")


def _plot_comparison(plain: dict, verified: dict, out_dir: Path) -> None:
    """Bar chart: hallucination rate before vs. after."""
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.set_style("whitegrid")

    labels = ["Plain RAG\n(no verification)", "Verified RAG\n(with NLI)"]
    values = [plain["mean"] * 100, verified["mean"] * 100]
    colors = ["#e74c3c", "#27ae60"]

    bars = ax.bar(labels, values, color=colors, width=0.45, edgecolor="white", linewidth=1.5)

    # Annotate bars
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=14,
            fontweight="bold",
        )

    ax.set_ylabel("Hallucination Rate (%)", fontsize=12)
    ax.set_title("ClarifAI-Med: Hallucination Rate Before vs. After Verification", fontsize=13, pad=15)
    ax.set_ylim(0, max(values) * 1.3 + 5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    improvement = plain["mean"] - verified["mean"]
    ax.annotate(
        f"▼ {improvement:.1%} reduction",
        xy=(0.5, 0.92),
        xycoords="axes fraction",
        ha="center",
        fontsize=11,
        color="#27ae60",
        fontweight="bold",
    )

    plt.tight_layout()
    chart_path = out_dir / "hallucination_rate_comparison.png"
    fig.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_distribution(plain_results: list, verified_results: list, out_dir: Path) -> None:
    """Histogram of per-question hallucination rates for both modes."""
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.set_style("whitegrid")

    plain_rates = [r["hallucination_rate"] * 100 for r in plain_results]
    verified_rates = [r["hallucination_rate"] * 100 for r in verified_results]

    ax.hist(plain_rates, bins=15, alpha=0.6, color="#e74c3c", label="Plain RAG", edgecolor="white")
    ax.hist(verified_rates, bins=15, alpha=0.6, color="#27ae60", label="Verified RAG", edgecolor="white")

    ax.set_xlabel("Hallucination Rate per Question (%)", fontsize=12)
    ax.set_ylabel("Number of Questions", fontsize=12)
    ax.set_title("Distribution of Per-Question Hallucination Rates", fontsize=13)
    ax.legend(fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    dist_path = out_dir / "hallucination_rate_distribution.png"
    fig.savefig(dist_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ClarifAI-Med evaluation.")
    parser.add_argument(
        "--questions",
        default=settings.test_questions_path,
        help="Path to test questions JSON file.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="Limit evaluation to first N questions.",
    )
    parser.add_argument(
        "--out",
        default=str(RESULTS_DIR),
        help="Output directory for results and charts.",
    )
    args = parser.parse_args()

    questions = load_test_questions(args.questions, n=args.n)
    run_evaluation(questions, out_dir=Path(args.out))


if __name__ == "__main__":
    main()
