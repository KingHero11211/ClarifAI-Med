"""
tests/test_pipeline.py

End-to-end pipeline integration tests with mocked components.

Tests the RAGPipeline.answer_question() flow without loading any ML models.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.pipeline.rag_pipeline import RAGPipeline, VerifiedAnswer
from src.retrieval.retriever import RetrievedChunk
from src.verification.verifier import VerificationResult, VerifiedSentence


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


def _make_retriever(texts: list[str]) -> MagicMock:
    """Mock retriever that returns a fixed list of chunks."""
    chunks = [
        RetrievedChunk(
            chunk_id=f"c{i}",
            text=text,
            score=0.1 + i * 0.05,
            question="Mock question",
            source_url="https://mock.example.com",
            focus_area="Mock",
            qtype="information",
        )
        for i, text in enumerate(texts)
    ]
    mock = MagicMock()
    mock.retrieve.return_value = chunks
    return mock


def _make_generator(answer: str) -> MagicMock:
    """Mock generator that returns a fixed answer."""
    mock = MagicMock()
    mock.generate.return_value = answer
    return mock


def _make_verifier(sentences: list[dict]) -> MagicMock:
    """Mock verifier that returns a fixed verification result."""
    verified_sentences = [
        VerifiedSentence(
            text=s["text"],
            support_label=s.get("support_label", "supported"),
            confidence=s.get("confidence", 0.9),
            contradiction_score=0.0,
            best_evidence_chunk="Mock chunk text",
        )
        for s in sentences
    ]
    mock_result = VerificationResult(
        answer_text=" ".join(s["text"] for s in sentences),
        sentences=verified_sentences,
    )
    mock = MagicMock()
    mock.verify.return_value = mock_result
    return mock


# ---------------------------------------------------------------------------
# Pipeline tests
# ---------------------------------------------------------------------------


def test_pipeline_returns_verified_answer():
    """Full pipeline should return a VerifiedAnswer with expected fields."""
    retriever = _make_retriever(["Aspirin reduces fever and inflammation."])
    generator = _make_generator("Aspirin is an analgesic. It reduces fever effectively.")
    verifier = _make_verifier([
        {"text": "Aspirin is an analgesic.", "support_label": "supported", "confidence": 0.88},
        {"text": "It reduces fever effectively.", "support_label": "supported", "confidence": 0.92},
    ])

    pipeline = RAGPipeline(retriever=retriever, generator=generator, verifier=verifier)
    result = pipeline.answer_question("What does aspirin do?")

    assert isinstance(result, VerifiedAnswer)
    assert result.query == "What does aspirin do?"
    assert result.answer_text != ""
    assert 0.0 <= result.overall_groundedness_score <= 1.0
    assert len(result.sentences) == 2
    assert len(result.retrieved_chunks) == 1


def test_pipeline_empty_query_raises():
    """Empty query should raise ValueError immediately."""
    pipeline = RAGPipeline(
        retriever=_make_retriever([]),
        generator=_make_generator(""),
        verifier=_make_verifier([]),
    )
    with pytest.raises(ValueError, match="empty"):
        pipeline.answer_question("")


def test_pipeline_skip_verification():
    """skip_verification=True should mark sentences as 'unverified'."""
    retriever = _make_retriever(["Some context text."])
    generator = _make_generator("This answer has two sentences. Here is the second one.")
    # Verifier should NOT be called when skip_verification=True
    verifier = MagicMock()

    pipeline = RAGPipeline(retriever=retriever, generator=generator, verifier=verifier)
    result = pipeline.answer_question("Test question?", skip_verification=True)

    verifier.verify.assert_not_called()
    assert isinstance(result, VerifiedAnswer)
    # In plain-RAG mode, sentences are "unverified"
    for s in result.sentences:
        assert s["support_label"] == "unverified"


def test_pipeline_to_dict_serializable():
    """VerifiedAnswer.to_dict() should produce a JSON-serialisable structure."""
    import json

    retriever = _make_retriever(["Context text."])
    generator = _make_generator("Test answer sentence.")
    verifier = _make_verifier([{"text": "Test answer sentence.", "support_label": "supported"}])

    pipeline = RAGPipeline(retriever=retriever, generator=generator, verifier=verifier)
    result = pipeline.answer_question("Test?")

    d = result.to_dict()
    json_str = json.dumps(d)  # Should not raise
    assert "query" in d
    assert "overall_groundedness_score" in d
    assert "sentences" in d
    assert "retrieved_chunks" in d
