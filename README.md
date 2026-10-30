# ClarifAI-Med 🏥🔍
### A Self-Checking Medical RAG System with NLI-Based Hallucination Auditing

**ClarifAI-Med** is a medical question-answering system that retrieves relevant documents (RAG), generates an LLM answer, and then **verifies every claim in that answer against the retrieved source text** using a Natural Language Inference (NLI) model — flagging unsupported claims before they reach the user.

The core contribution is a measurable, before/after demonstration of hallucination-rate reduction using an offline, interpretable verification layer.

> ⚠️ **Not a medical device.** "Unsupported" ≠ "wrong" — it means the claim cannot be traced to the retrieved context. See [LIMITATIONS.md](LIMITATIONS.md).

---

## Architecture

```
User Question
    │
    ▼
FastAPI Backend
    ├── 1. RETRIEVAL   → ChromaDB (MedQuAD embeddings via all-MiniLM-L6-v2)
    ├── 2. GENERATION  → LLM (Ollama/Mistral or OpenAI/Anthropic)
    └── 3. VERIFICATION → roberta-large-mnli (NLI claim checker)
                          ├── Sentence splitting (nltk)
                          ├── Per-claim entailment score vs retrieved chunks
                          └── Supported / Unsupported labels + groundedness score
    │
    ▼
Streamlit Frontend
    └── Color-coded answer (green=supported ✅, red=flagged ⚠️)
        + Overall groundedness score + source passages
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Store | ChromaDB (local, persistent) |
| LLM Generation | Ollama (Mistral, local/free) / OpenAI / Anthropic |
| NLI Verification | `roberta-large-mnli` (HuggingFace) |
| API | FastAPI + Swagger UI |
| Frontend | Streamlit (dark mode, color-coded claims) |
| Containerization | Docker + docker-compose |

---

## Quick Start

### Prerequisites
- Python **3.11** (preferred). Python **3.12** is an acceptable fallback. Do **not** use 3.14 — pinned `torch`/`numpy` wheels will not match.
- [Ollama](https://ollama.com) installed locally (or an OpenAI/Anthropic API key)

This is a student capstone prototype: a clean implementation of RAG plus an NLI audit layer. It is **not** a novel algorithm and **not** a medical device.

### 1. Clone & install

```bash
git clone https://github.com/YOUR_USERNAME/clarifai-med.git
cd clarifai-med

# Prefer 3.11; fall back to 3.12 if 3.11 is unavailable
py -3.11 -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env — set LLM_PROVIDER=ollama (default) or add your API key
```

### 3. Start Ollama and pull Mistral

```bash
# Windows: launch the Ollama system-tray app from Start menu, then:
ollama pull mistral

# Linux/Mac:
ollama serve &
ollama pull mistral
```

### 4. Download and process MedQuAD data

Uses the Hugging Face dump [`mukulb/clustered_MEDQUAD_dataset_with_groups`](https://huggingface.co/datasets/mukulb/clustered_MEDQUAD_dataset_with_groups) (16,407 QA pairs, 9 NIH sources, answers intact). The original GitHub MedQuAD XML often has answers removed for copyright.

```bash
python -m src.ingestion.load_medquad       # Download/cache CSV, clean, write JSONL + quality report
python -m src.ingestion.chunk_documents    # Token chunks (~256 tokens, overlap 50) via the embedding tokenizer
python scripts/generate_test_questions.py --n 100
python -m src.ingestion.build_vector_store # Embeds + stores in ChromaDB (Phase 2)
```

### 5. Run the API

```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```
Swagger UI: http://localhost:8000/docs

### 6. Run the frontend

```bash
streamlit run frontend/app.py
```
Open: http://localhost:8501

### 7. (Optional) Generate test questions & run evaluation

```bash
python scripts/generate_test_questions.py --n 100
python evaluation/run_evaluation.py --n 50
# Results + charts saved to evaluation/results/
```

---

## Docker (full stack)

```bash
docker-compose up --build
```
- Frontend: http://localhost:8501
- API + Swagger: http://localhost:8000/docs

> Note: For Ollama with Docker, you need the Ollama server accessible from inside containers. Set `OLLAMA_BASE_URL=http://host.docker.internal:11434` in your `.env`.

---

## Running Tests

```bash
pytest tests/ -v --cov=src
```

All tests use mocked ML models — no GPU or network required.

---

## Evaluation Results

The evaluation pipeline measures **hallucination rate** = percentage of generated sentences not entailed by retrieved context.

| System | Hallucination Rate |
|---|---|
| Plain RAG (no verification) | ~XX% |
| Verified RAG (with NLI layer) | ~XX% |

*(Populated after running `python evaluation/run_evaluation.py`)*

See `evaluation/results/` for the comparison chart.

---

## Repository Structure

```
ClarifAI-Med/
├── src/
│   ├── ingestion/         # Data loading, chunking, vector store building
│   │   ├── load_medquad.py
│   │   ├── chunk_documents.py
│   │   └── build_vector_store.py
│   ├── retrieval/
│   │   └── retriever.py   # ChromaDB semantic search
│   ├── generation/
│   │   ├── prompts.py     # Strict context-only prompt templates
│   │   └── generator.py   # Ollama/OpenAI/Anthropic wrapper
│   ├── verification/      # 🔑 Core contribution
│   │   ├── claim_splitter.py
│   │   ├── nli_checker.py # roberta-large-mnli claim verification
│   │   └── verifier.py    # Orchestrator → VerificationResult
│   ├── pipeline/
│   │   └── rag_pipeline.py # End-to-end: retrieve→generate→verify
│   └── api/
│       └── main.py        # FastAPI application
├── frontend/
│   └── app.py             # Streamlit UI (dark mode, color-coded claims)
├── evaluation/
│   ├── metrics.py         # hallucination_rate, precision@k
│   └── run_evaluation.py  # Batch eval: plain RAG vs verified RAG
├── tests/                 # pytest unit tests (all mocked)
├── scripts/               # Utility scripts
├── docker/                # Dockerfiles
├── data/                  # MedQuAD raw/processed data
└── notebooks/             # Exploration (not production code)
```

---

## Key Design Decisions

1. **NLI over LLM-as-judge**: We use a dedicated NLI model (`roberta-large-mnli`) rather than asking the generator LLM to self-verify. This avoids the circular problem where the same model that hallucinated is asked to check itself.

2. **Max entailment across chunks**: A claim is `SUPPORTED` if ANY retrieved chunk entails it above the threshold. This is a conservative strategy that avoids penalising correct facts that are spread across multiple passages.

3. **Sentence-level granularity**: Verification is applied per sentence (not per paragraph) so users can see exactly which part of an answer is problematic, not just a binary flag on the whole response.

4. **Threshold is configurable**: `ENTAILMENT_THRESHOLD` in `.env` controls the strictness. Default is 0.7 — adjust based on your precision/recall trade-off requirements.

---

## Limitations

See [LIMITATIONS.md](LIMITATIONS.md) for a full discussion.

---

## License
MIT — see LICENSE for details.
