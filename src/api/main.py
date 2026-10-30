"""
src/api/main.py

FastAPI application for ClarifAI-Med.

Endpoints:
  GET  /health          — Liveness probe
  POST /ask             — Answer a medical question (full verified RAG)
  POST /ask/plain       — Answer without verification (plain RAG, for eval)

Auto-docs at: http://localhost:8000/docs
"""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import get_settings
from src.pipeline.rag_pipeline import RAGPipeline, VerifiedAnswer

settings = get_settings()

# ---------------------------------------------------------------------------
# Application setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="ClarifAI-Med API",
    description=(
        "Self-checking medical RAG system. "
        "Every answer is verified claim-by-claim using NLI before reaching the user."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Allow the Streamlit frontend (port 8501) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single pipeline instance (shared across requests)
_pipeline: RAGPipeline | None = None


def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class AskRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=5,
        max_length=1000,
        description="The medical question to answer.",
        examples=["What are the symptoms of multiple sclerosis?"],
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of context chunks to retrieve.",
    )


class SentenceResult(BaseModel):
    text: str
    support_label: str  # "supported" | "unsupported" | "no_info"
    confidence: float
    contradiction_score: float
    best_evidence_chunk: str


class ChunkResult(BaseModel):
    chunk_id: str
    text: str
    similarity: float
    source_url: str
    focus_area: str


class AskResponse(BaseModel):
    query: str
    answer_text: str
    overall_groundedness_score: float = Field(
        description="Fraction of sentences supported by retrieved context (0-1)."
    )
    sentences: list[SentenceResult]
    retrieved_chunks: list[ChunkResult]
    latency_seconds: float


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", tags=["Meta"])
async def health() -> dict:
    """Liveness probe. Returns 200 if the API is running."""
    return {"status": "ok", "version": "1.0.0"}


@app.post("/ask", response_model=AskResponse, tags=["Q&A"])
async def ask(request: AskRequest) -> AskResponse:
    """
    Answer a medical question with full NLI-based claim verification.

    The pipeline:
      1. Retrieves the top-k relevant passages from the MedQuAD vector store.
      2. Generates an LLM answer constrained to those passages.
      3. Verifies each sentence against the context using roberta-large-mnli.
      4. Returns a structured answer with per-sentence support labels.
    """
    pipeline = get_pipeline()
    pipeline._top_k = request.top_k

    try:
        result: VerifiedAnswer = pipeline.answer_question(
            request.question, skip_verification=False
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        logger.error(f"/ask failed: {e}")
        raise HTTPException(status_code=503, detail=f"Pipeline error: {str(e)}")
    except Exception as e:
        logger.exception(f"/ask unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")

    return AskResponse(**result.to_dict())


@app.post("/ask/plain", response_model=AskResponse, tags=["Q&A"])
async def ask_plain(request: AskRequest) -> AskResponse:
    """
    Answer a medical question WITHOUT NLI verification (plain RAG baseline).

    Use this to compare plain RAG vs. verified RAG for evaluation purposes.
    Sentences are labelled "unverified" with a neutral confidence of 0.5.
    """
    pipeline = get_pipeline()
    pipeline._top_k = request.top_k

    try:
        result: VerifiedAnswer = pipeline.answer_question(
            request.question, skip_verification=True
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        logger.error(f"/ask/plain failed: {e}")
        raise HTTPException(status_code=503, detail=f"Pipeline error: {str(e)}")
    except Exception as e:
        logger.exception(f"/ask/plain unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")

    return AskResponse(**result.to_dict())


# ---------------------------------------------------------------------------
# Global exception handler — ensure JSON error responses always
# ---------------------------------------------------------------------------


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error(f"Unhandled exception on {request.url}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected internal error occurred."},
    )
