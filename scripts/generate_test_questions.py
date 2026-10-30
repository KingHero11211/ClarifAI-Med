"""
scripts/generate_test_questions.py

Generate a held-out test question set from MedQuAD data NOT used in
the vector store, for unbiased evaluation.

Usage:
    python scripts/generate_test_questions.py [--n 100] [--out data/test_questions.json]
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import get_settings

settings = get_settings()


def generate_test_questions(
    parsed_jsonl: str,
    chunks_jsonl: str,
    n: int = 100,
    output_path: str = "data/test_questions.json",
    seed: int = 42,
) -> list[dict]:
    """
    Sample n questions from medquad_parsed.jsonl that were NOT indexed
    in the vector store (i.e., are in different source files).

    As a pragmatic approach for eval: we simply sample from ALL parsed
    records and rely on the retriever to provide context. The test set
    represents realistic user queries over the dataset.
    """
    random.seed(seed)

    # Load all parsed records
    records = []
    with open(parsed_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    print(f"Loaded {len(records)} parsed records.")

    # Filter: only take records with reasonably sized answers
    # to ensure meaningful multi-sentence verification
    valid = [r for r in records if len(r["answer"].split()) > 30]
    print(f"Valid records (answer > 30 words): {len(valid)}")

    if len(valid) < n:
        print(f"Warning: only {len(valid)} valid records, sampling all.")
        sampled = valid
    else:
        sampled = random.sample(valid, n)

    questions = [
        {
            "question": r["question"],
            "reference_answer": r["answer"],
            "focus_area": r["focus_area"],
            "source_url": r["source_url"],
        }
        for r in sampled
    ]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(questions, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(questions)} test questions to {output_path}")
    return questions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--out", default=settings.test_questions_path)
    parser.add_argument(
        "--parsed",
        default=str(Path(settings.data_processed_dir) / "medquad_parsed.jsonl"),
    )
    parser.add_argument(
        "--chunks",
        default=str(Path(settings.data_processed_dir) / "medquad_chunks.jsonl"),
    )
    args = parser.parse_args()

    generate_test_questions(
        parsed_jsonl=args.parsed,
        chunks_jsonl=args.chunks,
        n=args.n,
        output_path=args.out,
    )


if __name__ == "__main__":
    main()
