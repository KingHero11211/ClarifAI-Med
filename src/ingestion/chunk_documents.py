"""
src/ingestion/chunk_documents.py

Reads parsed MedQuAD Q&A records from data/processed/medquad_parsed.jsonl,
splits long answers into overlapping passages using sentence-transformers
token-based chunking (~256 tokens with ~50 overlap), and writes the chunked
output to data/processed/medquad_chunks.jsonl.

Usage:
    python -m src.ingestion.chunk_documents
"""

import json
import os
import sys
from pathlib import Path
from typing import Any

# Ensure HF mirror endpoint is set to avoid network timeouts
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import nltk
from loguru import logger
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import get_settings

settings = get_settings()

# Download NLTK sentence tokenizer if needed
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)


def get_tokenizer(model_name: str) -> Any:
    """Load pretrained tokenizer for the embedding model."""
    logger.info(f"Loading tokenizer for model: {model_name}")
    return AutoTokenizer.from_pretrained(model_name)


def sent_tokenize(text: str) -> list[str]:
    """Split text into sentences using NLTK with period-split fallback."""
    try:
        return nltk.sent_tokenize(text)
    except Exception:
        return [s.strip() for s in text.split(".") if s.strip()]


def count_tokens(text: str, tokenizer: Any) -> int:
    """Count tokens without adding special tokens (CLS/SEP)."""
    return len(tokenizer.encode(text, add_special_tokens=False, truncation=False))


def chunk_text(
    text: str,
    tokenizer: Any,
    chunk_size: int = 256,
    overlap: int = 50,
) -> list[tuple[str, int]]:
    """
    Split text into chunks of at most `chunk_size` tokens with `overlap`.
    Preserves sentence boundaries wherever possible.

    Args:
        text: Input answer text to chunk.
        tokenizer: Transformers AutoTokenizer instance.
        chunk_size: Maximum tokens per chunk.
        overlap: Target token overlap between consecutive chunks.

    Returns:
        List of (chunk_text, token_count) tuples.
    """
    text = text.strip()
    if not text:
        return []

    raw_sentences = sent_tokenize(text)
    if not raw_sentences:
        return []

    # Handle oversized sentences by token slicing if any exceed chunk_size
    sentences: list[str] = []
    for sent in raw_sentences:
        sent = sent.strip()
        if not sent:
            continue
        sent_tokens = tokenizer.encode(sent, add_special_tokens=False, truncation=False)
        if len(sent_tokens) <= chunk_size:
            sentences.append(sent)
        else:
            # Hard-split long sentence by tokens
            step = max(1, chunk_size - overlap)
            for i in range(0, len(sent_tokens), step):
                sub_slice = sent_tokens[i : i + chunk_size]
                sub_text = tokenizer.decode(sub_slice, skip_special_tokens=True).strip()
                if sub_text:
                    sentences.append(sub_text)

    if not sentences:
        return []

    # Quick check: if all sentences combined fit in a single chunk
    total_tokens_approx = sum(
        len(tokenizer.encode(s, add_special_tokens=False, truncation=False))
        for s in sentences
    )
    if total_tokens_approx <= chunk_size:
        full_text = " ".join(sentences).strip()
        actual_tok_count = count_tokens(full_text, tokenizer)
        if actual_tok_count <= chunk_size:
            return [(full_text, actual_tok_count)]

    # Sliding window chunking with sentence boundary preservation
    chunks: list[tuple[str, int]] = []
    n = len(sentences)
    start_idx = 0

    while start_idx < n:
        current_sentences: list[str] = []
        current_tokens = 0
        end_idx = start_idx

        while end_idx < n:
            sent = sentences[end_idx]
            sent_tok = count_tokens(sent, tokenizer)
            # Stop if adding this sentence exceeds chunk_size (unless current chunk is empty)
            if current_sentences and (current_tokens + sent_tok > chunk_size):
                break
            current_sentences.append(sent)
            current_tokens += sent_tok
            end_idx += 1

        chunk_str = " ".join(current_sentences).strip()
        actual_tok_count = count_tokens(chunk_str, tokenizer)
        chunks.append((chunk_str, actual_tok_count))

        if end_idx >= n:
            break

        # Calculate next start_idx for overlap
        # Look backwards from end_idx to find sentences fitting within the overlap budget
        overlap_tokens = 0
        next_start = end_idx
        for j in range(end_idx - 1, start_idx, -1):
            tok_j = count_tokens(sentences[j], tokenizer)
            if overlap_tokens + tok_j <= overlap:
                overlap_tokens += tok_j
                next_start = j
            else:
                break

        # Ensure strict forward progress
        if next_start <= start_idx:
            next_start = start_idx + 1

        start_idx = next_start

    return chunks


def chunk_records(
    records: list[dict[str, Any]],
    tokenizer: Any,
    chunk_size: int = 256,
    overlap: int = 50,
) -> tuple[list[dict[str, Any]], list[str]]:
    """
    Chunk each Q&A record's answer field into passages.

    Args:
        records: List of parsed Q&A dicts from load_medquad.
        tokenizer: Transformers AutoTokenizer.
        chunk_size: Target token size per chunk.
        overlap: Token overlap between consecutive chunks.

    Returns:
        Tuple of (chunked_records, zero_chunk_record_ids).
    """
    chunked: list[dict[str, Any]] = []
    zero_chunk_ids: list[str] = []

    for record in records:
        source_id = record.get("id", "").strip()
        question = record.get("question", "").strip()
        source = record.get("source", "").strip()
        answer = record.get("answer", "").strip()

        if not answer:
            zero_chunk_ids.append(source_id)
            continue

        answer_chunks = chunk_text(
            answer, tokenizer=tokenizer, chunk_size=chunk_size, overlap=overlap
        )

        if not answer_chunks:
            zero_chunk_ids.append(source_id)
            continue

        for idx, (chunk_passage, tok_count) in enumerate(answer_chunks):
            chunked.append(
                {
                    "chunk_id": f"{source_id}_chunk_{idx}",
                    "source_id": source_id,
                    "question": question,
                    "source": source,
                    "chunk_text": chunk_passage,
                    "chunk_index": idx,
                    "token_count": tok_count,
                }
            )

    return chunked, zero_chunk_ids


def write_chunks(chunks: list[dict[str, Any]], output_path: Path) -> None:
    """Write chunked records line-by-line to a JSONL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")


def main() -> None:
    """Entry point: read parsed JSONL, chunk with tokenizer, save."""
    logger.info("=== Chunking MedQuAD Documents (Token-Based) ===")

    input_path = Path(settings.data_processed_dir) / "medquad_parsed.jsonl"
    output_path = Path(settings.data_processed_dir) / "medquad_chunks.jsonl"

    if not input_path.exists():
        logger.error(
            f"Input file not found: {input_path}. "
            "Run `python -m src.ingestion.load_medquad` first."
        )
        sys.exit(1)

    # Load parsed records
    records: list[dict[str, Any]] = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    logger.info(f"Loaded {len(records)} Q&A records from {input_path}")

    tokenizer = get_tokenizer(settings.embedding_model)
    chunk_size = settings.chunk_size
    chunk_overlap = settings.chunk_overlap

    logger.info(
        f"Starting chunking (model='{settings.embedding_model}', "
        f"chunk_size={chunk_size}, overlap={chunk_overlap})..."
    )

    chunked, zero_chunk_ids = chunk_records(
        records=records,
        tokenizer=tokenizer,
        chunk_size=chunk_size,
        overlap=chunk_overlap,
    )

    # Write output
    write_chunks(chunked, output_path)

    # Statistics
    total_records = len(records)
    total_chunks = len(chunked)
    avg_chunks_per_record = total_chunks / total_records if total_records else 0
    token_counts = [c["token_count"] for c in chunked]
    min_tokens = min(token_counts) if token_counts else 0
    max_tokens = max(token_counts) if token_counts else 0
    avg_tokens = sum(token_counts) / total_chunks if total_chunks else 0

    logger.info(f"Successfully saved {total_chunks} chunks to {output_path}")
    logger.info(f"Total records read: {total_records}")
    logger.info(f"Total chunks produced: {total_chunks}")
    logger.info(f"Average chunks per record: {avg_chunks_per_record:.2f}")
    logger.info(
        f"Token count stats -> min: {min_tokens}, max: {max_tokens}, avg: {avg_tokens:.1f}"
    )

    if zero_chunk_ids:
        logger.warning(
            f"Records producing zero chunks ({len(zero_chunk_ids)}): {zero_chunk_ids[:10]}"
        )
    else:
        logger.info("Zero records produced zero chunks.")


if __name__ == "__main__":
    main()
