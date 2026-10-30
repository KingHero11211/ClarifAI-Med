"""
src/ingestion/build_vector_store.py

Embeds all chunked MedQuAD passages using sentence-transformers and persists
them in a ChromaDB collection for fast semantic retrieval.

Usage:
    python -m src.ingestion.build_vector_store
"""

import json
import sys
from pathlib import Path
from typing import Any

from loguru import logger
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import get_settings

settings = get_settings()

BATCH_SIZE = 256  # Number of chunks to embed at a time


def load_chunks(chunks_path: Path) -> list[dict[str, Any]]:
    """Load chunked records from a JSONL file."""
    chunks: list[dict[str, Any]] = []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def build_vector_store(
    chunks: list[dict[str, Any]],
    embedding_model_name: str,
    chroma_persist_dir: str,
    collection_name: str,
    batch_size: int = BATCH_SIZE,
) -> None:
    """
    Embed chunks and upsert them into a persistent ChromaDB collection.

    Args:
        chunks: List of chunk dicts (must contain 'chunk_text' and 'chunk_id').
        embedding_model_name: HuggingFace sentence-transformer model name.
        chroma_persist_dir: Directory for ChromaDB persistence.
        collection_name: Name of the ChromaDB collection.
        batch_size: Number of chunks processed per embedding batch.
    """
    # Lazy imports to keep startup fast when the module is imported
    import chromadb
    from chromadb.utils import embedding_functions

    logger.info(f"Loading embedding model: {embedding_model_name}")
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=embedding_model_name
    )

    logger.info(f"Connecting to ChromaDB at: {chroma_persist_dir}")
    client = chromadb.PersistentClient(path=chroma_persist_dir)

    # Get or create collection (re-running is idempotent thanks to upsert)
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )

    logger.info(f"Starting to embed and upsert {len(chunks)} chunks...")

    total_upserted = 0
    for i in tqdm(range(0, len(chunks), batch_size), desc="Embedding batches"):
        batch = chunks[i : i + batch_size]

        ids = [c["chunk_id"] for c in batch]
        documents = [c["chunk_text"] for c in batch]
        metadatas = [
            {
                "question": c.get("question", "")[:500],  # Chroma metadata length limit
                "source_url": c.get("source_url", ""),
                "focus_area": c.get("focus_area", ""),
                "qtype": c.get("qtype", ""),
                "chunk_index": c.get("chunk_index", 0),
                "chunk_total": c.get("chunk_total", 1),
                "source_file": c.get("source_file", ""),
            }
            for c in batch
        ]

        try:
            collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
            total_upserted += len(batch)
        except Exception as e:
            logger.error(f"Failed to upsert batch {i // batch_size}: {e}")
            raise

    logger.info(
        f"Vector store built: {total_upserted} chunks upserted into "
        f"collection '{collection_name}'"
    )
    logger.info(f"Total vectors in collection: {collection.count()}")


def main() -> None:
    """Entry point: load chunks, embed, persist."""
    logger.info("=== Building Vector Store ===")

    chunks_path = Path(settings.data_processed_dir) / "medquad_chunks.jsonl"
    if not chunks_path.exists():
        logger.error(
            f"Chunks file not found: {chunks_path}. "
            "Run `python -m src.ingestion.chunk_documents` first."
        )
        sys.exit(1)

    chunks = load_chunks(chunks_path)
    logger.info(f"Loaded {len(chunks)} chunks from {chunks_path}")

    build_vector_store(
        chunks=chunks,
        embedding_model_name=settings.embedding_model,
        chroma_persist_dir=settings.chroma_persist_dir,
        collection_name=settings.chroma_collection_name,
    )


if __name__ == "__main__":
    main()
