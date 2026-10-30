"""
src/generation/generator.py

LLM generation layer. Supports three providers via the same OpenAI-compatible
interface:
  - Ollama (local/cloud) — via OpenAI SDK pointed at http://localhost:11434/v1/
  - OpenAI — standard API
  - Anthropic — via OpenAI-compatible shim (with fallback to native SDK)

Provider is selected by `settings.llm_provider` ("ollama" | "openai" | "anthropic").

Usage:
    from src.generation.generator import Generator
    gen = Generator()
    answer = gen.generate("What are the symptoms of Parkinson's disease?", chunks)
"""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import get_settings
from src.generation.prompts import build_messages

settings = get_settings()

# Token limit for the generated response
MAX_TOKENS = 512
TEMPERATURE = 0.1  # Low temperature for factual, deterministic outputs


class Generator:
    """
    LLM answer generator. Wraps Ollama / OpenAI / Anthropic behind a uniform
    `.generate(question, chunks) -> str` interface.
    """

    def __init__(self) -> None:
        self._provider = settings.llm_provider.lower()
        self._client = None  # lazy init

    def _get_openai_client(self):
        """Create an OpenAI-compatible client (works for Ollama and OpenAI)."""
        from openai import OpenAI

        if self._provider == "ollama":
            return OpenAI(
                base_url=f"{settings.ollama_base_url}/v1/",
                api_key="ollama",  # required by SDK, ignored by Ollama
            )
        elif self._provider == "openai":
            if not settings.openai_api_key:
                raise ValueError(
                    "OPENAI_API_KEY not set. Add it to .env or switch to LLM_PROVIDER=ollama."
                )
            return OpenAI(api_key=settings.openai_api_key)
        else:
            raise ValueError(f"Unsupported provider for OpenAI client: {self._provider}")

    def _get_anthropic_client(self):
        """Create a native Anthropic client."""
        import anthropic

        if not settings.anthropic_api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY not set. Add it to .env or switch LLM_PROVIDER."
            )
        return anthropic.Anthropic(api_key=settings.anthropic_api_key)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def generate(self, question: str, chunks: list) -> str:
        """
        Generate an answer for the given question using retrieved context chunks.

        Args:
            question: The medical question to answer.
            chunks: List of RetrievedChunk objects providing context.

        Returns:
            Generated answer string.

        Raises:
            RuntimeError: If generation fails after retries.
        """
        if not question or not question.strip():
            return "Please provide a valid question."

        messages = build_messages(question, chunks)

        try:
            if self._provider in ("ollama", "openai"):
                return self._generate_openai_style(messages)
            elif self._provider == "anthropic":
                return self._generate_anthropic(messages)
            else:
                raise ValueError(f"Unknown LLM provider: {self._provider}")
        except Exception as e:
            logger.error(f"Generation failed ({self._provider}): {e}")
            raise RuntimeError(f"LLM generation failed: {e}") from e

    def _generate_openai_style(self, messages: list[dict]) -> str:
        """Use OpenAI-compatible API (works for both Ollama and OpenAI)."""
        if self._client is None:
            self._client = self._get_openai_client()

        model = (
            settings.ollama_model
            if self._provider == "ollama"
            else settings.openai_model
        )

        logger.debug(f"Calling {self._provider} model: {model}")

        response = self._client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
        )
        answer = response.choices[0].message.content
        if not answer:
            return "The model returned an empty response."
        return answer.strip()

    def _generate_anthropic(self, messages: list[dict]) -> str:
        """Use native Anthropic SDK."""
        if self._client is None:
            self._client = self._get_anthropic_client()

        # Extract system message from messages list
        system_msg = ""
        user_msgs = []
        for m in messages:
            if m["role"] == "system":
                system_msg = m["content"]
            else:
                user_msgs.append(m)

        logger.debug(f"Calling Anthropic model: {settings.anthropic_model}")

        response = self._client.messages.create(
            model=settings.anthropic_model,
            max_tokens=MAX_TOKENS,
            system=system_msg,
            messages=user_msgs,
        )
        content = response.content[0].text if response.content else ""
        return content.strip() if content else "The model returned an empty response."


# Module-level singleton
_default_generator: Generator | None = None


def get_generator() -> Generator:
    """Return the default Generator instance (created once)."""
    global _default_generator
    if _default_generator is None:
        _default_generator = Generator()
    return _default_generator


def generate(question: str, chunks: list) -> str:
    """
    Convenience function: generate an answer using the default generator.

    Args:
        question: Medical question.
        chunks: Retrieved context chunks.

    Returns:
        Generated answer string.
    """
    return get_generator().generate(question, chunks)
