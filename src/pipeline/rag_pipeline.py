"""
src/pipeline/rag_pipeline.py

End-to-end RAG pipeline: Retrieval → Generation → Verification.

Exposes a single function `answer_question(query) -> VerifiedAnswer` that
takes a raw question string and returns a fully verified, structured answer.

Usage:
    from src.pipeline.rag_pipeline import answer_question
    result = answer_question("What are the symptoms of multiple sclerosis?")
    print(result.overall_groundedness_score)
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import get_settings
from src.generation.generator import Generator, get_generator
from src.retrieval.retriever import RetrievedChunk, Retriever, get_retriever
from src.verification.verifier import VerificationResult, Verifier, get_verifier

settings = get_settings()


@dataclass
class VerifiedAnswer:
    """
    The complete structured output of the RAG pipeline for one query.

    This is the contract between the pipeline and the API / frontend.
    """

    query: str
    answer_text: str
    overall_groundedness_score: float
    sentences: list[dict]  # list of VerifiedSentence dicts
    retrieved_chunks: list[dict]  # list of RetrievedChunk dicts (for transparency)
    latency_seconds: float = 0.0

    def to_dict(self) -> dict:
        """Serialize to a JSON-serialisable dict."""
        return {
            "query": self.query,
            "answer_text": self.answer_text,
            "overall_groundedness_score": self.overall_groundedness_score,
            "sentences": self.sentences,
            "retrieved_chunks": self.retrieved_chunks,
            "latency_seconds": round(self.latency_seconds, 3),
        }


class RAGPipeline:
    """
    Wires together Retriever, Generator, and Verifier into a single pipeline.

    Each component is initialised lazily so the pipeline can be imported
    without loading any ML models.
    """

    def __init__(
        self,
        retriever: Retriever | None = None,
        generator: Generator | None = None,
        verifier: Verifier | None = None,
        top_k: int | None = None,
    ) -> None:
        self._retriever = retriever
        self._generator = generator
        self._verifier = verifier
        self._top_k = top_k or settings.retrieval_top_k

    @property
    def retriever(self) -> Retriever:
        if self._retriever is None:
            self._retriever = get_retriever()
        return self._retriever

    @property
    def generator(self) -> Generator:
        if self._generator is None:
            self._generator = get_generator()
        return self._generator

    @property
    def verifier(self) -> Verifier:
        if self._verifier is None:
            self._verifier = get_verifier()
        return self._verifier

    def answer_question(
        self,
        query: str,
        skip_verification: bool = False,
    ) -> VerifiedAnswer:
        """
        Full pipeline: retrieve → generate → verify.

        Args:
            query: The user's medical question.
            skip_verification: If True, skip the NLI verification step
                (useful for comparing plain RAG vs. verified RAG in evaluation).

        Returns:
            VerifiedAnswer with answer, per-sentence labels, and groundedness score.

        Raises:
            ValueError: If the query is empty.
            RuntimeError: If a critical pipeline step fails.
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        start = time.perf_counter()

        # ── Step 1: Retrieve ──────────────────────────────────────────────────
        logger.info(f"[Pipeline] Retrieving top-{self._top_k} chunks for: '{query[:80]}'")
        try:
            chunks: list[RetrievedChunk] = self.retriever.retrieve(
                query, k=self._top_k
            )
        except RuntimeError as e:
            logger.error(f"[Pipeline] Retrieval failed: {e}")
            raise

        logger.info(f"[Pipeline] Retrieved {len(chunks)} chunks")

        # ── Step 2: Generate ──────────────────────────────────────────────────
        logger.info("[Pipeline] Generating answer...")
        try:
            answer_text: str = self.generator.generate(query, chunks)
        except RuntimeError as e:
            logger.error(f"[Pipeline] Generation failed: {e}")
            raise

        logger.info(f"[Pipeline] Generated answer ({len(answer_text)} chars)")

        # ── Step 3: Verify ────────────────────────────────────────────────────
        if skip_verification:
            # For plain-RAG baseline comparison
            verification_result = _make_unverified_result(answer_text)
        else:
            logger.info("[Pipeline] Verifying claims...")
            try:
                verification_result: VerificationResult = self.verifier.verify(
                    answer_text, chunks
                )
            except Exception as e:
                logger.error(f"[Pipeline] Verification failed: {e}")
                raise RuntimeError(f"Verification failed: {e}") from e

        latency = time.perf_counter() - start

        # ── Assemble output ───────────────────────────────────────────────────
        return VerifiedAnswer(
            query=query,
            answer_text=answer_text,
            overall_groundedness_score=verification_result.overall_groundedness_score,
            sentences=verification_result.to_dict()["sentences"],
            retrieved_chunks=[
                {
                    "chunk_id": c.chunk_id,
                    "text": c.text[:300],  # Trim for API response size
                    "similarity": round(c.similarity, 4),
                    "source_url": c.source_url,
                    "focus_area": c.focus_area,
                }
                for c in chunks
            ],
            latency_seconds=latency,
        )


def _make_unverified_result(answer_text: str) -> "VerificationResult":
    """
    Create a dummy VerificationResult for plain-RAG mode (no NLI verification).
    All sentences are marked as 'unverified' with neutral scores.
    """
    from src.verification.verifier import VerificationResult, VerifiedSentence
    from src.verification.claim_splitter import split_claims

    claims = split_claims(answer_text)
    sentences = [
        VerifiedSentence(
            text=c,
            support_label="unverified",
            confidence=0.5,
            contradiction_score=0.0,
            best_evidence_chunk="",
        )
        for c in claims
    ]
    result = VerificationResult(answer_text=answer_text, sentences=sentences)
    return result


# Module-level singleton pipeline
_default_pipeline: RAGPipeline | None = None


def get_pipeline() -> RAGPipeline:
    """Return the default RAGPipeline instance (created once)."""
    global _default_pipeline
    if _default_pipeline is None:
        _default_pipeline = RAGPipeline()
    return _default_pipeline


def answer_question(query: str, skip_verification: bool = False) -> VerifiedAnswer:
    """
    Convenience function: run the full pipeline on a query.

    Args:
        query: Medical question string.
        skip_verification: Skip NLI step (plain-RAG mode).

    Returns:
        VerifiedAnswer object.
    """
    return get_pipeline().answer_question(query, skip_verification=skip_verification)
