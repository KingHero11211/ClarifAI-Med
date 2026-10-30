"""
src/verification/claim_splitter.py

Splits a generated answer into individual atomic claims (sentences) using
NLTK's sentence tokenizer. Each sentence becomes a claim to be verified
independently by the NLI checker.

Usage:
    from src.verification.claim_splitter import split_claims
    claims = split_claims("Aspirin reduces fever. It also treats pain.")
    # -> ["Aspirin reduces fever.", "It also treats pain."]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from loguru import logger

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# NLTK punkt tokenizer
try:
    import nltk
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)
    _NLTK_AVAILABLE = True
except ImportError:
    _NLTK_AVAILABLE = False
    logger.warning("NLTK not available; falling back to regex sentence splitter.")


# Sentences shorter than this (in words) are considered fragments and merged
# into the next sentence to avoid noise in the verification step.
MIN_WORDS_PER_CLAIM = 5

# The "no information" response pattern — don't verify this, just pass through
NO_INFO_PATTERN = re.compile(
    r"i don.t have enough information",
    re.IGNORECASE,
)


def _regex_sentence_split(text: str) -> list[str]:
    """Simple regex fallback for sentence splitting."""
    raw = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text.strip())
    return [s.strip() for s in raw if s.strip()]


def split_claims(answer: str) -> list[str]:
    """
    Split an answer into a list of individual sentence-level claims.

    Empty answers, and answers matching the "no information" pattern, are
    returned as a single item (so the pipeline can handle them gracefully).

    Args:
        answer: The generated answer string.

    Returns:
        List of sentence strings, each representing one claim to verify.
    """
    if not answer or not answer.strip():
        return []

    # Short-circuit: don't split "I don't have enough information" responses
    if NO_INFO_PATTERN.search(answer):
        return [answer.strip()]

    # Sentence tokenize
    if _NLTK_AVAILABLE:
        try:
            sentences = nltk.sent_tokenize(answer.strip())
        except Exception as e:
            logger.warning(f"NLTK sent_tokenize failed: {e}; using regex fallback")
            sentences = _regex_sentence_split(answer)
    else:
        sentences = _regex_sentence_split(answer)

    # Clean and filter
    claims: list[str] = []
    pending_fragment: str = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        # Merge very short fragments into the next sentence
        if pending_fragment:
            sentence = pending_fragment + " " + sentence
            pending_fragment = ""

        word_count = len(sentence.split())
        if word_count < MIN_WORDS_PER_CLAIM:
            # Hold as pending — might merge with next sentence
            pending_fragment = sentence
        else:
            claims.append(sentence)

    # Flush remaining fragment as its own claim if it exists
    if pending_fragment:
        if claims:
            # Append to last claim
            claims[-1] = claims[-1] + " " + pending_fragment
        else:
            claims.append(pending_fragment)

    if not claims:
        # Edge case: everything was filtered; return the original answer
        return [answer.strip()]

    logger.debug(f"Split answer into {len(claims)} claims")
    return claims
