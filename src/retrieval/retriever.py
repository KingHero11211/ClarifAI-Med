"""
src/retrieval/retriever.py

Provides a `retrieve()` function that queries the ChromaDB vector store
for the top-k most semantically similar chunks to a given query string.

Usage:
    from src.retrieval.retriever import Retriever
    r = Retriever()
    chunks = r.retrieve("What causes type 2 diabetes?", k=5)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import chromadb
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import get_settings

settings = get_settings()


@dataclass
class RetrievedChunk:
    """A single retrieved passage with its metadata and relevance score."""

    chunk_id: str
    text: str
    score: float  # cosine distance (lower = more similar in Chroma's default)
    question: str = ""
    source_url: str = ""
    focus_area: str = ""
    qtype: str = ""
    chunk_index: int = 0
    chunk_total: int = 1

    @property
    def similarity(self) -> float:
        """Convert Chroma's L2/cosine distance to a 0-1 similarity score."""
        # ChromaDB with cosine space returns distances in [0, 2]; convert to similarity
        return max(0.0, 1.0 - self.score)


class Retriever:
    """
    Semantic retriever backed by ChromaDB.

    Lazy-initialises the ChromaDB client and embedding function on first use
    so the module can be imported cheaply.
    """

    def __init__(
        self,
        embedding_model: str | None = None,
        chroma_persist_dir: str | None = None,
        collection_name: str | None = None,
    ) -> None:
        self._embedding_model = embedding_model or settings.embedding_model
        self._chroma_persist_dir = chroma_persist_dir or settings.chroma_persist_dir
        self._collection_name = collection_name or settings.chroma_collection_name
        self._collection = None  # lazy init

    def _init_collection(self) -> None:
        """Initialise ChromaDB client and collection (called lazily)."""
        from chromadb.utils import embedding_functions

        logger.debug(f"Initialising ChromaDB from: {self._chroma_persist_dir}")
        embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self._embedding_model
        )
        client = chromadb.PersistentClient(path=self._chroma_persist_dir)
        self._collection = client.get_collection(
            name=self._collection_name,
            embedding_function=embedding_fn,
        )
        logger.debug(
            f"Loaded collection '{self._collection_name}' "
            f"({self._collection.count()} vectors)"
        )

    def retrieve(self, query: str, k: int | None = None) -> list[RetrievedChunk]:
        """
        Retrieve the top-k semantically similar chunks for a query.

        Args:
            query: Natural language question or query string.
            k: Number of results to return (default: from config).

        Returns:
            List of RetrievedChunk objects sorted by relevance (best first).

        Raises:
            RuntimeError: If the vector store has not been built yet.
        """
        if not query or not query.strip():
            logger.warning("Empty query passed to retriever; returning []")
            return []

        if self._collection is None:
            try:
                self._init_collection()
            except Exception as e:
                raise RuntimeError(
                    f"Could not load vector store from '{self._chroma_persist_dir}'. "
                    "Have you run `python -m src.ingestion.build_vector_store`? "
                    f"Original error: {e}"
                ) from e

        top_k = k or settings.retrieval_top_k

        try:
            results = self._collection.query(
                query_texts=[query],
                n_results=min(top_k, self._collection.count()),
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            logger.error(f"ChromaDB query failed: {e}")
            raise

        chunks: list[RetrievedChunk] = []
        if not results or not results["ids"]:
            return chunks

        ids = results["ids"][0]
        documents = results["documents"][0]
        distances = results["distances"][0]
        metadatas = results["metadatas"][0]

        for chunk_id, doc, dist, meta in zip(ids, documents, distances, metadatas):
            chunks.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    text=doc,
                    score=dist,
                    question=meta.get("question", ""),
                    source_url=meta.get("source_url", ""),
                    focus_area=meta.get("focus_area", ""),
                    qtype=meta.get("qtype", ""),
                    chunk_index=meta.get("chunk_index", 0),
                    chunk_total=meta.get("chunk_total", 1),
                )
            )

        logger.debug(f"Retrieved {len(chunks)} chunks for query: '{query[:80]}...'")
        return chunks


# Module-level singleton for convenience
_default_retriever: Retriever | None = None


def get_retriever() -> Retriever:
    """Return the module-level default Retriever (created once)."""
    global _default_retriever
    if _default_retriever is None:
        _default_retriever = Retriever()
    return _default_retriever


def retrieve(query: str, k: int | None = None) -> list[RetrievedChunk]:
    """
    Convenience function: retrieve top-k chunks using the default retriever.

    Args:
        query: Natural language query.
        k: Number of results (default from config).

    Returns:
        List of RetrievedChunk objects.
    """
    return get_retriever().retrieve(query, k=k)
