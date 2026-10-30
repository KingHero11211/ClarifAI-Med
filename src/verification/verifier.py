"""
src/verification/verifier.py

Orchestrates the full claim verification workflow:
  1. Split generated answer into individual claims (sentence-level)
  2. Run NLI checker on each claim vs. retrieved context
  3. Return a VerificationResult with per-claim labels and an overall score

Usage:
    from src.verification.verifier import Verifier
    verifier = Verifier()
    result = verifier.verify(answer_text, chunks)
    print(result.overall_groundedness_score)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.verification.claim_splitter import split_claims
from src.verification.nli_checker import NLIChecker, NLIResult, SupportLabel, get_nli_checker


@dataclass
class VerifiedSentence:
    """A single sentence from the answer with its verification result."""

    text: str
    support_label: str  # "supported" | "unsupported" | "no_info"
    confidence: float  # entailment score (0-1)
    contradiction_score: float
    best_evidence_chunk: str  # Short excerpt from the most supporting chunk

    @property
    def is_supported(self) -> bool:
        return self.support_label == SupportLabel.SUPPORTED.value

    @property
    def is_no_info(self) -> bool:
        return self.support_label == SupportLabel.NO_INFO.value


@dataclass
class VerificationResult:
    """Complete verification result for one answer."""

    answer_text: str  # Original generated answer
    sentences: list[VerifiedSentence] = field(default_factory=list)

    @property
    def overall_groundedness_score(self) -> float:
        """
        Fraction of verifiable sentences that are supported.

        'no_info' sentences are excluded from the denominator since the system
        correctly acknowledged it couldn't answer.
        """
        verifiable = [s for s in self.sentences if not s.is_no_info]
        if not verifiable:
            return 1.0  # Nothing to verify = perfectly grounded by convention
        supported = sum(1 for s in verifiable if s.is_supported)
        return round(supported / len(verifiable), 3)

    @property
    def unsupported_sentences(self) -> list[VerifiedSentence]:
        """Return only unsupported sentences."""
        return [s for s in self.sentences if not s.is_supported and not s.is_no_info]

    @property
    def supported_count(self) -> int:
        return sum(1 for s in self.sentences if s.is_supported)

    @property
    def unsupported_count(self) -> int:
        return sum(1 for s in self.sentences if not s.is_supported and not s.is_no_info)

    def to_dict(self) -> dict:
        """Serialize to a JSON-serialisable dict (for the API response)."""
        return {
            "answer_text": self.answer_text,
            "overall_groundedness_score": self.overall_groundedness_score,
            "supported_count": self.supported_count,
            "unsupported_count": self.unsupported_count,
            "sentences": [
                {
                    "text": s.text,
                    "support_label": s.support_label,
                    "confidence": round(s.confidence, 4),
                    "contradiction_score": round(s.contradiction_score, 4),
                    "best_evidence_chunk": s.best_evidence_chunk,
                }
                for s in self.sentences
            ],
        }


class Verifier:
    """
    Orchestrates claim splitting + NLI checking into a single verify() call.

    Attributes:
        checker: NLIChecker instance (lazy-loaded by default).
    """

    def __init__(self, checker: NLIChecker | None = None) -> None:
        self._checker = checker  # If None, use module-level singleton lazily

    @property
    def checker(self) -> NLIChecker:
        if self._checker is None:
            self._checker = get_nli_checker()
        return self._checker

    def verify(self, answer_text: str, chunks: list) -> VerificationResult:
        """
        Verify every sentence in the answer against the retrieved context.

        Args:
            answer_text: The generated answer string.
            chunks: List of RetrievedChunk objects from the retriever.

        Returns:
            VerificationResult with per-sentence labels and overall score.
        """
        if not answer_text or not answer_text.strip():
            logger.warning("Empty answer passed to verifier.")
            return VerificationResult(answer_text="", sentences=[])

        # Step 1: Split into claims
        claims = split_claims(answer_text)
        logger.info(f"Verifying {len(claims)} claim(s) from answer...")

        # Step 2: NLI check each claim
        nli_results: list[NLIResult] = self.checker.check_claims(claims, chunks)

        # Step 3: Map to VerifiedSentence objects
        verified_sentences: list[VerifiedSentence] = []
        for nli_result in nli_results:
            verified_sentences.append(
                VerifiedSentence(
                    text=nli_result.claim,
                    support_label=nli_result.label.value,
                    confidence=nli_result.entailment_score,
                    contradiction_score=nli_result.contradiction_score,
                    best_evidence_chunk=nli_result.best_chunk_text,
                )
            )

        result = VerificationResult(
            answer_text=answer_text,
            sentences=verified_sentences,
        )

        logger.info(
            f"Verification complete: {result.supported_count} supported, "
            f"{result.unsupported_count} unsupported, "
            f"groundedness={result.overall_groundedness_score:.2%}"
        )

        return result


# Module-level singleton
_default_verifier: Verifier | None = None


def get_verifier() -> Verifier:
    """Return the default Verifier instance (created once)."""
    global _default_verifier
    if _default_verifier is None:
        _default_verifier = Verifier()
    return _default_verifier


def verify(answer_text: str, chunks: list) -> VerificationResult:
    """
    Convenience function: verify an answer using the default verifier.

    Args:
        answer_text: Generated answer to verify.
        chunks: Retrieved context chunks.

    Returns:
        VerificationResult.
    """
    return get_verifier().verify(answer_text, chunks)
