"""
config.py — Central configuration for ClarifAI-Med.
All settings are loaded from environment variables (via .env file).
No hardcoded secrets or magic numbers anywhere in the codebase.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from functools import lru_cache


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM Provider ---
    llm_provider: str = Field(default="ollama", description="LLM provider: openai | anthropic | ollama")
    openai_api_key: str = Field(default="", description="OpenAI API key")
    openai_model: str = Field(default="gpt-4o-mini", description="OpenAI model name")
    anthropic_api_key: str = Field(default="", description="Anthropic API key")
    anthropic_model: str = Field(default="claude-3-haiku-20240307", description="Anthropic model name")
    ollama_base_url: str = Field(default="http://localhost:11434", description="Ollama server URL")
    ollama_model: str = Field(default="mistral", description="Ollama model name")

    # --- Embedding ---
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="HuggingFace sentence-transformer model for embeddings",
    )

    # --- NLI Verification ---
    nli_model: str = Field(
        default="roberta-large-mnli",
        description="HuggingFace NLI model for claim verification",
    )

    # --- Vector Store ---
    chroma_persist_dir: str = Field(default="./chroma_db", description="ChromaDB persistence directory")
    chroma_collection_name: str = Field(default="medquad_chunks", description="ChromaDB collection name")

    # --- Retrieval ---
    retrieval_top_k: int = Field(default=5, description="Number of chunks to retrieve per query")
    chunk_size: int = Field(default=256, description="Target tokens per chunk (embedding tokenizer)")
    chunk_overlap: int = Field(default=50, description="Overlap tokens between consecutive chunks")
    random_seed: int = Field(default=42, description="Global seed for ingestion and sampling")
    hf_dataset_id: str = Field(
        default="mukulb/clustered_MEDQUAD_dataset_with_groups",
        description="Hugging Face dataset id for MedQuAD-derived QA pairs",
    )

    # --- Verification ---
    entailment_threshold: float = Field(
        default=0.7,
        description="NLI entailment score above which a claim is considered 'supported'",
    )

    # --- API ---
    api_host: str = Field(default="0.0.0.0", description="FastAPI host")
    api_port: int = Field(default=8000, description="FastAPI port")

    # --- Data paths ---
    data_raw_dir: str = Field(default="./data/raw", description="Raw data directory")
    data_processed_dir: str = Field(default="./data/processed", description="Processed data directory")
    test_questions_path: str = Field(
        default="./data/test_questions.json", description="Held-out eval questions"
    )

    # --- Logging ---
    log_level: str = Field(default="INFO", description="Logging level")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    return Settings()
