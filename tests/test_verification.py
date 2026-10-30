"""
tests/test_verification.py

Unit tests for the core verification layer — claim splitting and NLI checking.

NLI model calls are mocked to keep tests fast and offline.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import nltk
import pytest

nltk.download("punkt_tab", quiet=True)  # ensure tokenizer is available

from src.verification.claim_splitter import split_claims
from src.verification.nli_checker import NLIChecker, NLIResult, SupportLabel
from src.verification.verifier import Verifier, VerificationResult, VerifiedSentence

from src.retrieval.retriever import RetrievedChunk


# ---------------------------------------------------------------------------
# Claim splitter tests
# ---------------------------------------------------------------------------


def test_split_basic_sentences():
    """A simple two-sentence answer should split into at least one claim containing 'Aspirin'."""
    text = "Aspirin reduces fever and inflammation. It also relieves mild to moderate pain effectively."
    claims = split_claims(text)
    # Should produce 1 or 2 claims (depends on tokenizer availability)
    assert len(claims) >= 1
    assert any("Aspirin" in c for c in claims)


def test_split_empty_string():
    """Empty string should return empty list."""
    assert split_claims("") == []
    assert split_claims("   ") == []


def test_split_no_info_response():
    """'I don't have enough information' responses should not be split."""
    text = "I don't have enough information in the retrieved context to answer this question reliably."
    claims = split_claims(text)
    assert len(claims) == 1
    assert "don" in claims[0].lower()


def test_split_single_sentence():
    """A single sentence should return a list with one element."""
    text = "Parkinson's disease is a progressive neurological disorder affecting movement."
    claims = split_claims(text)
    assert len(claims) >= 1


def test_split_filters_fragments():
    """Very short fragments should be merged, not returned as separate claims."""
    text = "Diabetes is a metabolic disease. OK. This affects blood sugar regulation significantly."
    claims = split_claims(text)
    # "OK." is too short (<5 words) so it should be merged
    for claim in claims:
        assert len(claim.split()) >= 3  # fragments merged away


# ---------------------------------------------------------------------------
# NLI Checker tests (mocked pipeline)
# ---------------------------------------------------------------------------


def _make_chunk(text: str, chunk_id: str = "c1") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=text,
        score=0.1,
        question="",
        source_url="",
        focus_area="",
        qtype="",
    )


def _mock_nli_pipeline(entailment_score: float = 0.9):
    """Returns a mock NLI pipeline that always returns the given entailment score."""
    mock = MagicMock()
    mock.return_value = [
        {"label": "ENTAILMENT", "score": entailment_score},
        {"label": "NEUTRAL", "score": 1.0 - entailment_score - 0.05},
        {"label": "CONTRADICTION", "score": 0.05},
    ]
    return mock


def test_nli_supported_claim():
    """A claim strongly entailed by the chunk should be marked SUPPORTED."""
    checker = NLIChecker(threshold=0.7)
    checker._pipeline = _mock_nli_pipeline(entailment_score=0.92)

    chunk = _make_chunk("Aspirin is used to reduce fever and relieve pain.")
    result = checker.check_claim("Aspirin reduces fever.", [chunk])

    assert result.label == SupportLabel.SUPPORTED
    assert result.entailment_score > 0.7
    assert result.is_supported


def test_nli_unsupported_claim():
    """A claim with low entailment should be marked UNSUPPORTED."""
    checker = NLIChecker(threshold=0.7)
    checker._pipeline = _mock_nli_pipeline(entailment_score=0.2)

    chunk = _make_chunk("Aspirin is used to reduce fever and relieve pain.")
    result = checker.check_claim("Aspirin cures cancer completely.", [chunk])

    assert result.label == SupportLabel.UNSUPPORTED
    assert not result.is_supported


def test_nli_no_chunks():
    """Without any chunks, claims should be UNSUPPORTED."""
    checker = NLIChecker(threshold=0.7)
    result = checker.check_claim("Aspirin reduces fever.", chunks=[])

    assert result.label == SupportLabel.UNSUPPORTED
    assert result.entailment_score == 0.0


def test_nli_no_info_claim():
    """'I don't have enough information' should be labelled NO_INFO."""
    checker = NLIChecker(threshold=0.7)
    chunk = _make_chunk("Some irrelevant text.")
    result = checker.check_claim(
        "I don't have enough information in the retrieved context.",
        [chunk],
    )
    assert result.label == SupportLabel.NO_INFO


def test_nli_max_entailment_across_chunks():
    """The checker should pick the best chunk — if any chunk entails the claim, mark SUPPORTED."""
    checker = NLIChecker(threshold=0.7)

    # First call: low entailment; second call: high entailment
    mock_pipeline = MagicMock()
    call_count = [0]

    def side_effect(text):
        call_count[0] += 1
        if call_count[0] == 1:
            return [
                {"label": "ENTAILMENT", "score": 0.1},
                {"label": "NEUTRAL", "score": 0.8},
                {"label": "CONTRADICTION", "score": 0.1},
            ]
        else:
            return [
                {"label": "ENTAILMENT", "score": 0.95},
                {"label": "NEUTRAL", "score": 0.03},
                {"label": "CONTRADICTION", "score": 0.02},
            ]

    mock_pipeline.side_effect = side_effect
    checker._pipeline = mock_pipeline

    chunks = [_make_chunk("Irrelevant context.", "c1"), _make_chunk("Aspirin reduces fever.", "c2")]
    result = checker.check_claim("Aspirin reduces fever.", chunks)

    assert result.label == SupportLabel.SUPPORTED
    assert result.best_chunk_id == "c2"


# ---------------------------------------------------------------------------
# Verifier orchestration tests
# ---------------------------------------------------------------------------


def test_verifier_overall_groundedness_score():
    """Groundedness score should be fraction of supported sentences."""
    checker = NLIChecker(threshold=0.7)

    call_count = [0]
    def side_effect(text):
        call_count[0] += 1
        score = 0.9 if call_count[0] % 2 == 1 else 0.1  # alternating
        return [
            {"label": "ENTAILMENT", "score": score},
            {"label": "NEUTRAL", "score": 0.05},
            {"label": "CONTRADICTION", "score": 0.05},
        ]

    checker._pipeline = MagicMock(side_effect=side_effect)

    verifier = Verifier(checker=checker)
    chunks = [_make_chunk("Some context about aspirin and pain management.")]
    answer = "Aspirin reduces fever. This sentence will be unsupported."

    result = verifier.verify(answer, chunks)

    assert isinstance(result, VerificationResult)
    assert 0.0 <= result.overall_groundedness_score <= 1.0
    assert len(result.sentences) >= 1


def test_verifier_empty_answer():
    """Empty answer should return empty VerificationResult."""
    verifier = Verifier()
    result = verifier.verify("", chunks=[])
    assert result.answer_text == ""
    assert result.sentences == []


def test_verifier_to_dict():
    """VerificationResult.to_dict() should produce correct JSON structure."""
    checker = NLIChecker(threshold=0.7)
    checker._pipeline = _mock_nli_pipeline(entailment_score=0.85)

    verifier = Verifier(checker=checker)
    chunks = [_make_chunk("Aspirin is an analgesic.")]
    result = verifier.verify("Aspirin reduces pain.", chunks)

    d = result.to_dict()
    assert "answer_text" in d
    assert "overall_groundedness_score" in d
    assert "sentences" in d
    assert isinstance(d["sentences"], list)
