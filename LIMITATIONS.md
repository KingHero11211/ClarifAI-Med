# ClarifAI-Med — Self-Checking Medical RAG System

## LIMITATIONS

> **Read this before using, citing, or extending this system.**

This document is a mandatory companion to the system. It is not a disclaimer to be skimmed — it is a precise statement of what the verification layer can and cannot guarantee.

---

### 1. "Unsupported" ≠ "Wrong"

When the NLI verifier flags a sentence as **unsupported**, it means:

> *The sentence cannot be traced back to the retrieved context passages with sufficient entailment confidence.*

It does **not** mean the sentence is factually incorrect. A sentence can be medically accurate but unsupported if:
- The retrieved passages happen not to mention that specific fact
- The sentence is phrased in a way the NLI model doesn't recognise as entailed
- The relevant passage scored below the retrieval top-k cutoff

**What to do:** Treat flagged sentences as "unverified claims requiring additional source lookup," not as confirmed errors.

---

### 2. The NLI Model Itself Can Be Wrong

`roberta-large-mnli` is a high-quality but imperfect model. It:
- Can mark correct sentences as **unsupported** (false negatives) — especially for numerical claims, long sentences, or indirect entailment
- Can mark incorrect sentences as **supported** (false positives) — especially if the premise contains similar vocabulary without genuine entailment
- Was trained on general English text (MultiNLI), not specifically on medical language

The threshold (default: 0.7) is a tunable parameter. A lower threshold catches fewer real hallucinations but also reduces false alarms; a higher threshold is stricter but may over-flag.

---

### 3. Not a Medical Device

This system is a **research and demonstration project**. It:
- Has not been clinically validated
- Has not been tested for safety in any real-world healthcare setting
- Does not constitute medical advice
- Does not replace consultation with a qualified healthcare professional

**Do not use the output of this system for any clinical decision-making.**

---

### 4. Dataset and Coverage Limitations

The vector store is built from a **MedQuAD-derived Hugging Face dump** (`mukulb/clustered_MEDQUAD_dataset_with_groups`):
- 16,407 question–answer pairs from 9 NIH sources (not the full original 12-site XML collection)
- Answers are retained; many answers in the original [abachaa/MedQuAD](https://github.com/abachaa/MedQuAD) GitHub XML were stripped for copyright
- Coverage is limited to those NIH topics and may be incomplete or outdated for rapidly evolving areas (e.g., recent drug approvals, updated treatment guidelines)
- Does not cover rare diseases, experimental treatments, or very recent research
- This is a research prototype dataset, not a clinically validated knowledge base

For questions outside the coverage of MedQuAD, the system will correctly return "I don't have enough information" — but may also occasionally generate plausible-sounding but unsupported answers (the hallucination the verification layer is designed to catch).

---

### 5. Single-Turn Only

This system handles single-turn Q&A only. It has no conversation memory, cannot follow up on previous questions, and cannot resolve pronouns or context across multiple turns. Each query is processed independently.

---

### 6. Language Coverage

The system is trained and evaluated on **English-language** text only. Inputs in other languages may produce degraded or meaningless results.

---

### 7. Retrieval Quality Affects Everything

The verification layer can only verify claims against the **retrieved** passages. If the retriever returns poor or irrelevant passages (e.g., for unusual query phrasing), the downstream generation and verification are both impacted. A high groundedness score on poor retrieval can be misleading.

---

### 8. Reproducibility Notes

- LLM outputs are non-deterministic even at low temperature; re-running the same query may produce slightly different answers and different verification results
- The NLI model is deterministic given the same input, but floating-point scores may vary across hardware and library versions
- All library versions are pinned in `requirements.txt` for reproducibility

---

### Summary Table

| Claim | True? |
|---|---|
| "Supported" sentences are guaranteed to be medically correct | ❌ No |
| "Unsupported" sentences are guaranteed to be wrong | ❌ No |
| The system reduces hallucination rate on average | ✅ Yes (empirically, per evaluation) |
| This system is safe to use for clinical decisions | ❌ No |
| The NLI checker can itself be wrong | ✅ Yes |
| Results are fully reproducible across all hardware | ⚠️ Mostly, with pinned deps |
