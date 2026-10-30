"""
tests/test_retrieval.py

Unit tests for the retrieval module.

These tests mock ChromaDB so they don't require a live vector store.
Sanity checks on the Retriever interface and RetrievedChunk properties.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.retrieval.retriever import RetrievedChunk, Retriever


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_chunk(chunk_id="c1", text="Aspirin reduces pain and fever.", score=0.1) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=text,
        score=score,
        question="What does aspirin do?",
        source_url="https://example.com",
        focus_area="Pain management",
        qtype="treatment",
        chunk_index=0,
        chunk_total=1,
    )


# ---------------------------------------------------------------------------
# RetrievedChunk tests
# ---------------------------------------------------------------------------


def test_retrieved_chunk_similarity_conversion():
    """Cosine distance of 0.1 should give similarity of 0.9."""
    chunk = _make_chunk(score=0.1)
    assert abs(chunk.similarity - 0.9) < 1e-6


def test_retrieved_chunk_similarity_clamped():
    """Similarity should never go below 0."""
    chunk = _make_chunk(score=2.5)  # > 1.0 distance
    assert chunk.similarity == 0.0


def test_retrieved_chunk_similarity_perfect():
    """Score of 0.0 means identical — similarity = 1.0."""
    chunk = _make_chunk(score=0.0)
    assert chunk.similarity == 1.0


# ---------------------------------------------------------------------------
# Retriever interface tests (mocked ChromaDB)
# ---------------------------------------------------------------------------


def _mock_collection(chunks: list[RetrievedChunk]):
    """Create a mock ChromaDB collection that returns the given chunks."""
    collection = MagicMock()
    collection.count.return_value = len(chunks)
    collection.query.return_value = {
        "ids": [[c.chunk_id for c in chunks]],
        "documents": [[c.text for c in chunks]],
        "distances": [[c.score for c in chunks]],
        "metadatas": [
            [
                {
                    "question": c.question,
                    "source_url": c.source_url,
                    "focus_area": c.focus_area,
                    "qtype": c.qtype,
                    "chunk_index": c.chunk_index,
                    "chunk_total": c.chunk_total,
                    "source_file": "",
                }
                for c in chunks
            ]
        ],
    }
    return collection


def test_retriever_returns_chunks():
    """Retriever.retrieve() should return the mock chunks."""
    mock_chunks = [_make_chunk(f"c{i}") for i in range(3)]
    retriever = Retriever()
    retriever._collection = _mock_collection(mock_chunks)

    results = retriever.retrieve("What does aspirin do?", k=3)

    assert len(results) == 3
    assert results[0].chunk_id == "c0"


def test_retriever_empty_query():
    """Empty query should return empty list without calling ChromaDB."""
    retriever = Retriever()
    retriever._collection = MagicMock()

    results = retriever.retrieve("", k=5)

    assert results == []
    retriever._collection.query.assert_not_called()


def test_retriever_no_vector_store():
    """Calling retrieve() before the vector store is built should raise RuntimeError."""
    retriever = Retriever(chroma_persist_dir="./nonexistent_dir_xyz")

    # Mock chromadb so PersistentClient raises (simulates missing collection)
    with patch("src.retrieval.retriever.chromadb") as mock_chroma:
        mock_chroma.PersistentClient.return_value.get_collection.side_effect = ValueError(
            "Collection 'medquad_chunks' does not exist."
        )
        with pytest.raises(RuntimeError, match="Could not load vector store"):
            retriever.retrieve("test query", k=3)
