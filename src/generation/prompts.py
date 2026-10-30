"""
src/generation/prompts.py

Strict prompt templates for medical Q&A generation.
The LLM is instructed to answer ONLY from the provided context,
preventing it from drawing on prior training knowledge.
"""

# ---------------------------------------------------------------------------
# System prompt: sets the persona and strict context-only constraint
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are ClarifAI-Med, a medical information assistant. Your sole job is to \
answer the user's question using ONLY the context passages provided below. \
You must NOT use any information from your training data that is not \
present in the context.

Rules you must follow without exception:
1. Answer using ONLY the information in the context passages.
2. If the context does not contain enough information to answer the question, \
say exactly: "I don't have enough information in the retrieved context to \
answer this question reliably."
3. Do NOT fabricate drug names, dosages, statistics, or medical facts.
4. Write in clear, plain language. Do not use bullet lists unless the context \
itself is structured as a list.
5. Keep the answer concise — aim for 3-6 sentences unless the question \
genuinely requires more detail.
6. At the end of your answer, do NOT add disclaimers or "consult a doctor" \
notes — those are added by the system automatically.
"""

# ---------------------------------------------------------------------------
# User-turn template: injects retrieved context + question
# ---------------------------------------------------------------------------
USER_PROMPT_TEMPLATE = """\
--- CONTEXT PASSAGES ---
{context}
--- END OF CONTEXT ---

QUESTION: {question}

Please answer the question using ONLY the context passages above.\
"""


def format_context(chunks: list) -> str:
    """
    Format a list of RetrievedChunk objects into a numbered context string.

    Args:
        chunks: List of RetrievedChunk objects (from retriever).

    Returns:
        Formatted multi-passage string for injection into the prompt.
    """
    if not chunks:
        return "(No relevant context passages were retrieved.)"

    parts = []
    for i, chunk in enumerate(chunks, start=1):
        source = chunk.source_url or chunk.focus_area or "Unknown source"
        parts.append(
            f"[Passage {i}] Source: {source}\n{chunk.text}"
        )
    return "\n\n".join(parts)


def build_messages(question: str, chunks: list) -> list[dict]:
    """
    Build the messages list for the OpenAI-compatible chat API.

    Args:
        question: User's medical question.
        chunks: Retrieved context chunks.

    Returns:
        List of message dicts: [{"role": "system", "content": ...}, ...]
    """
    context_str = format_context(chunks)
    user_content = USER_PROMPT_TEMPLATE.format(
        context=context_str,
        question=question,
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
