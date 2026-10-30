"""
src/ingestion/load_medquad.py

Ingests the MedQuAD dataset and writes a clean JSONL file to
data/processed/medquad_parsed.jsonl, ready for chunk_documents.py.

Default source: HuggingFace dataset (mukulb/clustered_MEDQUAD_dataset_with_groups).
Fallback source: raw XML files downloaded from GitHub (original behaviour).

Usage:
    python -m src.ingestion.load_medquad              # HF (default)
    python -m src.ingestion.load_medquad --source hf
    python -m src.ingestion.load_medquad --source github

HF column mapping
    query            -> question
    answers          -> answer
    group_name       -> source
    text             -> raw_text  (kept for debugging)
    topic_embeddings -> DROPPED   (integer cluster IDs, not useful downstream)

Output record schema
    {
        "id":       "medquad_000001",   # stable sequential counter
        "question": "...",
        "answer":   "...",
        "source":   "1_CancerGov_QA",
        "raw_text": "<HUMAN>: ... <ASSISTANT>: ..."
    }
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ── HF mirror must be set before importing datasets/huggingface_hub ──────────
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from loguru import logger

# Ensure project root is importable when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import get_settings

settings = get_settings()

# ── Paths from config ─────────────────────────────────────────────────────────
RAW_DIR = Path(settings.data_raw_dir)
PROCESSED_DIR = Path(settings.data_processed_dir)
OUTPUT_JSONL = PROCESSED_DIR / "medquad_parsed.jsonl"
QUALITY_REPORT = PROCESSED_DIR / "data_quality_report.json"

# ── GitHub fallback constants ─────────────────────────────────────────────────
MEDQUAD_FOLDERS = [
    "1_CancerGov_QA",
    "2_GARD_QA",
    "3_GHR_QA",
    "4_MedlinePlus_DISS_QA",
    "6_NINDS_QA",
    "7_SeniorHealth_QA",
    "8_NIDDK_QA",
    "9_CDC_QA",
    "10_MedlinePlus_QA",
]

# ─────────────────────────────────────────────────────────────────────────────
# Cleaning helpers
# ─────────────────────────────────────────────────────────────────────────────

_MULTI_WHITESPACE = re.compile(r"[ \t]+")
_MULTI_NEWLINE = re.compile(r"\n{3,}")


def _normalise(text: str) -> str:
    """
    Normalise whitespace in a string:
      - Strip leading/trailing whitespace
      - Collapse runs of spaces/tabs to a single space
      - Collapse runs of 3+ newlines to two newlines (preserve paragraph breaks)
    """
    text = text.strip()
    text = _MULTI_WHITESPACE.sub(" ", text)
    text = _MULTI_NEWLINE.sub("\n\n", text)
    return text


def clean_record(
    raw: dict[str, Any],
    drop_counters: dict[str, int],
) -> dict[str, Any] | None:
    """
    Clean and validate one raw HuggingFace record.

    Returns a cleaned dict, or None if the record should be dropped.
    Increments the relevant counter in drop_counters on drop.
    """
    try:
        question = _normalise(raw.get("query", "") or "")
        answer = _normalise(raw.get("answers", "") or "")
        source = _normalise(raw.get("group_name", "") or "")
        raw_text = _normalise(raw.get("text", "") or "")

        if not question:
            drop_counters["empty_question"] += 1
            return None
        if not answer:
            drop_counters["empty_answer"] += 1
            return None

        return {
            "question": question,
            "answer": answer,
            "source": source,
            "raw_text": raw_text,
        }
    except Exception as exc:
        logger.warning(f"Unexpected error cleaning record: {exc}")
        drop_counters["parse_error"] += 1
        return None


# ─────────────────────────────────────────────────────────────────────────────
# HuggingFace source
# ─────────────────────────────────────────────────────────────────────────────

def load_from_huggingface() -> list[dict[str, Any]]:
    """
    Load the MedQuAD dataset from HuggingFace (uses local cache if available).
    Returns a list of cleaned records with deterministic IDs.

    Raises SystemExit if the dataset cannot be loaded and no cache exists.
    """
    dataset_id = settings.hf_dataset_id
    logger.info(f"Loading HuggingFace dataset: {dataset_id}")
    logger.info(f"HF_ENDPOINT = {os.environ.get('HF_ENDPOINT', 'not set')}")

    try:
        from datasets import load_dataset  # type: ignore
        dataset = load_dataset(dataset_id)
    except Exception as exc:
        logger.error(
            f"Failed to load HuggingFace dataset '{dataset_id}': {exc}\n"
            "If offline, ensure the dataset is in the HuggingFace cache "
            "or use --source github."
        )
        sys.exit(1)

    split_name = "train" if "train" in dataset else list(dataset.keys())[0]
    split = dataset[split_name]
    total_loaded = len(split)
    logger.info(f"Dataset loaded — split: '{split_name}', rows: {total_loaded:,}")
    logger.info(f"Columns: {split.column_names}")

    # ── Clean each record ────────────────────────────────────────────────────
    logger.info("Cleaning records...")
    drop_counters: dict[str, int] = {
        "empty_question": 0,
        "empty_answer": 0,
        "parse_error": 0,
    }

    raw_cleaned: list[dict[str, Any]] = []
    for row in split:
        cleaned = clean_record(dict(row), drop_counters)
        if cleaned is not None:
            raw_cleaned.append(cleaned)

    total_dropped = sum(drop_counters.values())
    logger.info(
        f"Cleaning complete — written: {len(raw_cleaned):,}, "
        f"dropped: {total_dropped:,} {drop_counters}"
    )

    # ── Assign deterministic IDs (sort by source then original position) ──────
    # Sort by (source, original list position via enumerate) so IDs are stable
    # across re-runs as long as the dataset itself doesn't change.
    indexed = sorted(enumerate(raw_cleaned), key=lambda t: (t[1]["source"], t[0]))
    records: list[dict[str, Any]] = []
    for seq, (_, rec) in enumerate(indexed, start=1):
        record_with_id = {"id": f"medquad_{seq:06d}"}
        record_with_id.update(rec)
        records.append(record_with_id)

    # ── Build quality report ─────────────────────────────────────────────────
    source_dist: dict[str, int] = {}
    q_lengths: list[int] = []
    a_lengths: list[int] = []
    for rec in records:
        src = rec.get("source", "unknown")
        source_dist[src] = source_dist.get(src, 0) + 1
        q_lengths.append(len(rec["question"]))
        a_lengths.append(len(rec["answer"]))

    report: dict[str, Any] = {
        "dataset_id": dataset_id,
        "total_records_loaded": total_loaded,
        "total_records_written": len(records),
        "total_records_dropped": total_dropped,
        "drop_reasons": drop_counters,
        "source_distribution": dict(
            sorted(source_dist.items(), key=lambda kv: -kv[1])
        ),
        "avg_question_length_chars": round(
            sum(q_lengths) / len(q_lengths), 1
        ) if q_lengths else 0,
        "avg_answer_length_chars": round(
            sum(a_lengths) / len(a_lengths), 1
        ) if a_lengths else 0,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    return records, total_loaded, report


# ─────────────────────────────────────────────────────────────────────────────
# GitHub XML fallback (original implementation, preserved)
# ─────────────────────────────────────────────────────────────────────────────

def _parse_xml_file(xml_path: Path) -> list[dict[str, Any]]:
    """Parse a single MedQuAD XML file into a list of Q&A dicts."""
    records: list[dict[str, Any]] = []
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError as exc:
        logger.warning(f"Failed to parse {xml_path}: {exc}")
        return records

    source_url = root.attrib.get("url", "")
    focus_area = root.attrib.get("Focus", "")

    for qa_pair in root.findall(".//QAPair"):
        question_el = qa_pair.find("Question")
        answer_el = qa_pair.find("Answer")

        question_text = (question_el.text or "").strip() if question_el is not None else ""
        answer_text = (answer_el.text or "").strip() if answer_el is not None else ""
        qtype = question_el.attrib.get("qtype", "") if question_el is not None else ""

        if not question_text or not answer_text:
            continue

        records.append(
            {
                "question": question_text,
                "answer": answer_text,
                "source_url": source_url,
                "focus_area": focus_area,
                "qtype": qtype,
                "source_file": str(xml_path.name),
            }
        )
    return records


def load_from_local(raw_dir: Path) -> list[dict[str, Any]]:
    """Walk raw_dir for all .xml files and parse them."""
    all_records: list[dict[str, Any]] = []
    xml_files = list(raw_dir.rglob("*.xml"))
    logger.info(f"Found {len(xml_files)} XML files in {raw_dir}")
    for xml_path in xml_files:
        all_records.extend(_parse_xml_file(xml_path))
    return all_records


def download_medquad_sample(raw_dir: Path, max_files: int = 200) -> None:
    """Download a sample of MedQuAD XML files from GitHub."""
    import json as json_lib
    import urllib.request

    raw_dir.mkdir(parents=True, exist_ok=True)
    downloaded = 0

    for folder in MEDQUAD_FOLDERS:
        if downloaded >= max_files:
            break
        api_url = (
            f"https://api.github.com/repos/abachaa/MedQuAD/contents/{folder}"
        )
        try:
            with urllib.request.urlopen(api_url, timeout=15) as resp:
                files = json_lib.loads(resp.read())
        except Exception as exc:
            logger.warning(f"Could not list {folder}: {exc}")
            continue

        folder_dir = raw_dir / folder
        folder_dir.mkdir(exist_ok=True)

        for file_info in files:
            if downloaded >= max_files:
                break
            if not file_info.get("name", "").endswith(".xml"):
                continue
            dest = folder_dir / file_info["name"]
            if dest.exists():
                downloaded += 1
                continue
            try:
                with urllib.request.urlopen(
                    file_info["download_url"], timeout=15
                ) as resp:
                    dest.write_bytes(resp.read())
                downloaded += 1
                if downloaded % 20 == 0:
                    logger.info(f"Downloaded {downloaded}/{max_files} files...")
            except Exception as exc:
                logger.warning(f"Failed to download {file_info['name']}: {exc}")

    logger.info(f"Downloaded {downloaded} XML files to {raw_dir}")


def load_from_github() -> tuple[list[dict[str, Any]], int, dict[str, Any]]:
    """
    GitHub XML fallback: download (if needed) and parse XML files.
    Returns (records, total_loaded, quality_report) in the same shape as
    load_from_huggingface().
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    xml_files = list(RAW_DIR.rglob("*.xml"))
    if not xml_files:
        logger.info("No XML files found locally — downloading MedQuAD sample from GitHub...")
        download_medquad_sample(RAW_DIR, max_files=300)
    else:
        logger.info(f"Found {len(xml_files)} existing XML files in {RAW_DIR}")

    raw_records = load_from_local(RAW_DIR)
    total_loaded = len(raw_records)
    logger.info(f"Parsed {total_loaded:,} Q&A records from XML")

    if not raw_records:
        logger.error("No records parsed from XML. Exiting.")
        sys.exit(1)

    # Assign IDs sorted by source_file then index
    indexed = sorted(
        enumerate(raw_records), key=lambda t: (t[1].get("source_file", ""), t[0])
    )
    records: list[dict[str, Any]] = []
    for seq, (_, rec) in enumerate(indexed, start=1):
        record_with_id = {"id": f"medquad_{seq:06d}"}
        record_with_id.update(rec)
        records.append(record_with_id)

    # Quality report (GitHub schema slightly different — no HF columns)
    source_dist: dict[str, int] = {}
    q_lengths: list[int] = []
    a_lengths: list[int] = []
    for rec in records:
        src = rec.get("focus_area", "unknown")
        source_dist[src] = source_dist.get(src, 0) + 1
        q_lengths.append(len(rec["question"]))
        a_lengths.append(len(rec["answer"]))

    report: dict[str, Any] = {
        "dataset_id": "github:abachaa/MedQuAD",
        "total_records_loaded": total_loaded,
        "total_records_written": len(records),
        "total_records_dropped": 0,
        "drop_reasons": {"empty_question": 0, "empty_answer": 0, "parse_error": 0},
        "source_distribution": dict(
            sorted(source_dist.items(), key=lambda kv: -kv[1])
        ),
        "avg_question_length_chars": round(
            sum(q_lengths) / len(q_lengths), 1
        ) if q_lengths else 0,
        "avg_answer_length_chars": round(
            sum(a_lengths) / len(a_lengths), 1
        ) if a_lengths else 0,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    return records, total_loaded, report


# ─────────────────────────────────────────────────────────────────────────────
# Shared I/O
# ─────────────────────────────────────────────────────────────────────────────

def save_jsonl(records: list[dict[str, Any]], output_path: Path) -> None:
    """Write records to a JSON Lines file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info(f"Saved {len(records):,} records → {output_path}")


def save_quality_report(report: dict[str, Any], report_path: Path) -> None:
    """Write the quality report JSON."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    logger.info(f"Quality report saved → {report_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="MedQuAD ingestion — produces data/processed/medquad_parsed.jsonl"
    )
    parser.add_argument(
        "--source",
        choices=["hf", "github"],
        default="hf",
        help="Data source: 'hf' (HuggingFace, default) or 'github' (XML fallback).",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("=== MedQuAD Ingestion ===")
    logger.info(f"Source : {args.source}")
    logger.info(f"Output : {OUTPUT_JSONL}")
    logger.info("=" * 60)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    if args.source == "hf":
        records, total_loaded, report = load_from_huggingface()
    else:
        records, total_loaded, report = load_from_github()

    # ── Save JSONL ────────────────────────────────────────────────────────────
    save_jsonl(records, OUTPUT_JSONL)

    # ── Save quality report ───────────────────────────────────────────────────
    save_quality_report(report, QUALITY_REPORT)

    # ── Summary to log ────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("=== INGESTION COMPLETE ===")
    logger.info(f"  Records loaded   : {total_loaded:,}")
    logger.info(f"  Records written  : {report['total_records_written']:,}")
    logger.info(f"  Records dropped  : {report['total_records_dropped']:,}")
    logger.info(f"  Drop reasons     : {report['drop_reasons']}")
    logger.info(f"  Avg Q length     : {report['avg_question_length_chars']} chars")
    logger.info(f"  Avg A length     : {report['avg_answer_length_chars']} chars")
    top3 = list(report["source_distribution"].items())[:3]
    logger.info(f"  Top 3 sources    : {top3}")
    logger.info(f"  JSONL ready for  : python -m src.ingestion.chunk_documents")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
