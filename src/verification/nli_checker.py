"""
src/verification/nli_checker.py

NLI-based claim verification using a pretrained RoBERTa-large-mnli model.

For each (claim, context_chunks) pair, we:
  1. Concatenate each chunk with the claim as (premise, hypothesis)
  2. Run the NLI model to get [contradiction, neutral, entailment] scores
  3. Take the maximum entailment score across all chunks
  4. Apply the entailment threshold to produce a SUPPORTED / UNSUPPORTED label

The model is loaded once and cached for the lifetime of the process.

Usage:
    from src.verification.nli_checker import NLIChecker, NLIResult
    checker = NLIChecker()
    result = checker.check_claim("Aspirin reduces fever.", chunks)
    print(result.label, result.entailment_score)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import torch
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import get_settings

settings = get_settings()

# Maximum number of characters to use from each chunk (to stay within model limits)
MAX_CHUNK_CHARS = 512


class SupportLabel(str, Enum):
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    NO_INFO = "no_info"  # for "I don't have enough information" responses


@dataclass
class NLIResult:
    """Result of checking one claim against the retrieved context."""

    claim: str
    label: SupportLabel
    entailment_score: float  # Max entailment score across all chunks (0-1)
    contradiction_score: float  # Max contradiction score (for debugging)
    best_chunk_id: str  # chunk_id of the chunk that gave the highest entailment
    best_chunk_text: str  # text of the best-supporting chunk

    @property
    def is_supported(self) -> bool:
        return self.label == SupportLabel.SUPPORTED

    @property
    def confidence(self) -> float:
        """Alias for entailment_score, for display purposes."""
        return self.entailment_score


class NLIChecker:
    """
    Verifies claims against retrieved context using an NLI model.

    The pipeline and tokenizer are loaded lazily on first use.
    Thread-safety: the HuggingFace pipeline is not thread-safe; in production
    use a process pool or model serving if concurrency is needed.
    """

    def __init__(
        self,
        model_name: str | None = None,
        threshold: float | None = None,
    ) -> None:
        self._model_name = model_name or settings.nli_model
        self._threshold = threshold if threshold is not None else settings.entailment_threshold
        self._pipeline = None  # lazy init
        self._label2idx: dict[str, int] = {}

    def _load_pipeline(self) -> None:
        """Load the NLI model pipeline (called once)."""
        from transformers import pipeline as hf_pipeline

        logger.info(f"Loading NLI model: {self._model_name} (this may take a moment...)")
        device = 0 if torch.cuda.is_available() else -1
        self._pipeline = hf_pipeline(
            task="text-classification",
            model=self._model_name,
            top_k=None,  # return all label scores
            device=device,
            truncation=True,
            max_length=512,
        )
        logger.info(
            f"NLI model loaded on {'GPU' if device == 0 else 'CPU'}: {self._model_name}"
        )

    def _run_nli(self, premise: str, hypothesis: str) -> dict[str, float]:
        """
        Run NLI on a single (premise, hypothesis) pair.

        Returns a dict: {"ENTAILMENT": float, "NEUTRAL": float, "CONTRADICTION": float}
        """
        if self._pipeline is None:
            self._load_pipeline()

        # NLI models expect premise + separator + hypothesis
        input_text = f"{premise} </s></s> {hypothesis}"

        try:
            results = self._pipeline(input_text)
        except Exception as e:
            logger.warning(f"NLI inference failed: {e}")
            return {"ENTAILMENT": 0.0, "NEUTRAL": 1.0, "CONTRADICTION": 0.0}

        # results is a list of [{"label": "ENTAILMENT", "score": 0.9}, ...]
        scores: dict[str, float] = {}
        for item in results:
            label = item["label"].upper()
            scores[label] = item["score"]

        return scores

    def check_claim(
        self,
        claim: str,
        chunks: list,
    ) -> NLIResult:
        """
        Check whether a single claim is entailed by the retrieved context.

        We test each chunk independently as a premise and take the maximum
        entailment score. This is a conservative strategy: if ANY chunk
        supports the claim, we consider it supported.

        Args:
            claim: A single sentence/claim to verify.
            chunks: List of RetrievedChunk objects from the retriever.

        Returns:
            NLIResult with the label and supporting evidence.
        """
        import re

        # Handle the "no information" special case
        no_info_pattern = re.compile(r"i don.t have enough information", re.IGNORECASE)
        if no_info_pattern.search(claim):
            return NLIResult(
                claim=claim,
                label=SupportLabel.NO_INFO,
                entailment_score=1.0,  # The system chose not to answer — that's correct
                contradiction_score=0.0,
                best_chunk_id="",
                best_chunk_text="",
            )

        if not chunks:
            return NLIResult(
                claim=claim,
                label=SupportLabel.UNSUPPORTED,
                entailment_score=0.0,
                contradiction_score=0.0,
                best_chunk_id="",
                best_chunk_text="(no context retrieved)",
            )

        best_entailment = 0.0
        best_contradiction = 0.0
        best_chunk_id = ""
        best_chunk_text = ""

        for chunk in chunks:
            premise = chunk.text[:MAX_CHUNK_CHARS]
            scores = self._run_nli(premise=premise, hypothesis=claim)

            entailment = scores.get("ENTAILMENT", 0.0)
            contradiction = scores.get("CONTRADICTION", 0.0)

            if entailment > best_entailment:
                best_entailment = entailment
                best_chunk_id = chunk.chunk_id
                best_chunk_text = chunk.text[:200]

            if contradiction > best_contradiction:
                best_contradiction = contradiction

        label = (
            SupportLabel.SUPPORTED
            if best_entailment >= self._threshold
            else SupportLabel.UNSUPPORTED
        )

        logger.debug(
            f"Claim: '{claim[:60]}...' → {label.value} "
            f"(entailment={best_entailment:.3f}, threshold={self._threshold})"
        )

        return NLIResult(
            claim=claim,
            label=label,
            entailment_score=best_entailment,
            contradiction_score=best_contradiction,
            best_chunk_id=best_chunk_id,
            best_chunk_text=best_chunk_text,
        )

    def check_claims(
        self,
        claims: list[str],
        chunks: list,
    ) -> list[NLIResult]:
        """
        Check multiple claims against the retrieved context.

        Args:
            claims: List of claim strings.
            chunks: List of RetrievedChunk objects.

        Returns:
            List of NLIResult objects, one per claim.
        """
        results = []
        for claim in claims:
            result = self.check_claim(claim, chunks)
            results.append(result)
        return results


# Module-level singleton
_default_checker: NLIChecker | None = None


def get_nli_checker() -> NLIChecker:
    """Return the default NLIChecker instance (created once)."""
    global _default_checker
    if _default_checker is None:
        _default_checker = NLIChecker()
    return _default_checker
