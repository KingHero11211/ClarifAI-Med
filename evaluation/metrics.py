"""
evaluation/metrics.py

Core metric computation functions for the ClarifAI-Med evaluation pipeline.

Metrics:
  - hallucination_rate: % of verifiable sentences that are NOT supported by context
  - precision_at_k: % of top-k retrieved chunks relevant to the query (requires labels)
  - groundedness_score: complement of hallucination_rate (for display)
"""

from __future__ import annotations

from typing import Any


def hallucination_rate(sentence_labels: list[str]) -> float:
    """
    Compute hallucination rate = fraction of verifiable sentences that are unsupported.

    Args:
        sentence_labels: List of label strings per sentence
                         ("supported", "unsupported", "no_info", "unverified").

    Returns:
        Float in [0, 1]. Higher = more hallucination.
    """
    verifiable = [
        label for label in sentence_labels if label not in ("no_info", "unverified")
    ]
    if not verifiable:
        return 0.0
    unsupported = sum(1 for label in verifiable if label == "unsupported")
    return round(unsupported / len(verifiable), 4)


def groundedness_score(sentence_labels: list[str]) -> float:
    """Complement of hallucination_rate."""
    return round(1.0 - hallucination_rate(sentence_labels), 4)


def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """
    Precision@k for retrieval evaluation.

    Args:
        retrieved_ids: Ordered list of retrieved chunk IDs.
        relevant_ids: Set of known-relevant chunk IDs (ground truth).

    Returns:
        Precision@k (0-1).
    """
    if not retrieved_ids:
        return 0.0
    hits = sum(1 for cid in retrieved_ids if cid in relevant_ids)
    return round(hits / len(retrieved_ids), 4)


def aggregate_hallucination_rates(results: list[dict[str, Any]]) -> dict[str, float]:
    """
    Aggregate per-question results into overall statistics.

    Args:
        results: List of result dicts, each with "sentence_labels" key.

    Returns:
        Dict with mean, min, max hallucination rate and sample count.
    """
    rates = [hallucination_rate(r["sentence_labels"]) for r in results]
    if not rates:
        return {"mean": 0.0, "min": 0.0, "max": 0.0, "n": 0}
    return {
        "mean": round(sum(rates) / len(rates), 4),
        "min": round(min(rates), 4),
        "max": round(max(rates), 4),
        "n": len(rates),
    }
