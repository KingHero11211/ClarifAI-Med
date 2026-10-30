"""
frontend/app.py

ClarifAI-Med Streamlit frontend.

Displays:
  - Text input for medical questions
  - Color-coded answer (green = supported, red = flagged/unsupported)
  - Overall groundedness score prominently
  - Retrieved source passages for transparency
  - Side-by-side plain RAG vs verified RAG comparison (optional)

Run:
    streamlit run frontend/app.py
"""

import json
import sys
from pathlib import Path

import requests
import streamlit as st

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
API_BASE = "http://localhost:8000"

# ──────────────────────────────────────────────────────────────────────────────
# Page config (must be first Streamlit call)
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ClarifAI-Med — Self-Checking Medical Q&A",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# Custom CSS: dark modern theme
# ──────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Dark gradient background */
    .stApp {
        background: linear-gradient(135deg, #0d1117 0%, #161b27 50%, #0d1117 100%);
    }

    /* Header banner */
    .clarifai-header {
        background: linear-gradient(90deg, #1a2740 0%, #0f3460 50%, #1a2740 100%);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 2rem;
        border: 1px solid rgba(99, 179, 237, 0.2);
        box-shadow: 0 8px 32px rgba(0,0,0,0.4);
    }

    .clarifai-header h1 {
        color: #63b3ed;
        font-size: 2.4rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }

    .clarifai-header p {
        color: #a0aec0;
        font-size: 1rem;
        margin: 0.5rem 0 0 0;
    }

    /* Groundedness score card */
    .score-card {
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        font-weight: 700;
        font-size: 2.2rem;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }

    .score-high { background: linear-gradient(135deg, #1a4731, #276749); color: #68d391; border: 1px solid #276749; }
    .score-mid  { background: linear-gradient(135deg, #5c3a00, #9c6400); color: #f6c05c; border: 1px solid #9c6400; }
    .score-low  { background: linear-gradient(135deg, #5c1a1a, #9b2c2c); color: #fc8181; border: 1px solid #9b2c2c; }

    /* Sentence chips */
    .sentence-supported {
        display: inline-block;
        background: rgba(39, 174, 96, 0.12);
        border-left: 3px solid #27ae60;
        border-radius: 6px;
        padding: 0.5rem 0.75rem;
        margin: 0.3rem 0;
        color: #c6f6d5;
        font-size: 0.95rem;
        width: 100%;
    }

    .sentence-unsupported {
        display: inline-block;
        background: rgba(231, 76, 60, 0.12);
        border-left: 3px solid #e74c3c;
        border-radius: 6px;
        padding: 0.5rem 0.75rem;
        margin: 0.3rem 0;
        color: #fed7d7;
        font-size: 0.95rem;
        width: 100%;
    }

    .sentence-noinfo {
        display: inline-block;
        background: rgba(99, 179, 237, 0.1);
        border-left: 3px solid #63b3ed;
        border-radius: 6px;
        padding: 0.5rem 0.75rem;
        margin: 0.3rem 0;
        color: #bee3f8;
        font-size: 0.95rem;
        width: 100%;
    }

    .badge-supported { background: #27ae60; color: white; font-size: 0.7rem; padding: 2px 8px; border-radius: 10px; font-weight: 600; }
    .badge-unsupported { background: #e74c3c; color: white; font-size: 0.7rem; padding: 2px 8px; border-radius: 10px; font-weight: 600; }
    .badge-noinfo { background: #63b3ed; color: #1a202c; font-size: 0.7rem; padding: 2px 8px; border-radius: 10px; font-weight: 600; }

    /* Source chunk cards */
    .chunk-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 0.75rem;
    }

    .chunk-score { color: #63b3ed; font-size: 0.85rem; font-weight: 600; }
    .chunk-source { color: #718096; font-size: 0.75rem; }
    .chunk-text { color: #e2e8f0; font-size: 0.9rem; line-height: 1.6; }

    /* Disclaimer box */
    .disclaimer {
        background: rgba(245, 158, 11, 0.08);
        border: 1px solid rgba(245, 158, 11, 0.3);
        border-radius: 8px;
        padding: 0.75rem 1rem;
        color: #fbd38d;
        font-size: 0.85rem;
        margin-top: 1rem;
    }

    /* Hide Streamlit default footer */
    footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="clarifai-header">
        <h1>🏥 ClarifAI-Med</h1>
        <p>Self-Checking Medical Q&A · Every claim verified against retrieved sources using NLI</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    top_k = st.slider("Passages to retrieve (k)", min_value=1, max_value=15, value=5)
    show_sources = st.checkbox("Show source passages", value=True)
    compare_mode = st.checkbox("Compare Plain vs. Verified RAG", value=False)

    st.markdown("---")
    st.markdown("### ℹ️ How it works")
    st.markdown(
        """
        1. **Retrieve** — Top-k passages from MedQuAD
        2. **Generate** — LLM answers using *only* the retrieved context
        3. **Verify** — Each sentence checked via NLI model (`roberta-large-mnli`)
        4. **Display** — Green = supported ✅ · Red = flagged ⚠️
        """
    )

    st.markdown("---")
    st.markdown("### ⚠️ Disclaimer")
    st.markdown(
        """
        *Not a medical device.* 'Unsupported' does not mean 'wrong' —
        it means the claim cannot be traced to the retrieved context.
        Do not use for clinical decisions.
        """,
        unsafe_allow_html=False,
    )

# ──────────────────────────────────────────────────────────────────────────────
# Sample questions
# ──────────────────────────────────────────────────────────────────────────────
SAMPLE_QUESTIONS = [
    "What are the symptoms of multiple sclerosis?",
    "How is type 2 diabetes treated?",
    "What causes Parkinson's disease?",
    "What are the risk factors for heart disease?",
    "How is asthma diagnosed and managed?",
]

st.markdown("**Try an example question or type your own:**")
col_samples = st.columns(len(SAMPLE_QUESTIONS))
selected_sample = None
for i, (col, q) in enumerate(zip(col_samples, SAMPLE_QUESTIONS)):
    with col:
        if st.button(f"💬 {q[:30]}…", key=f"sample_{i}", use_container_width=True):
            selected_sample = q

# ──────────────────────────────────────────────────────────────────────────────
# Question input
# ──────────────────────────────────────────────────────────────────────────────
if "question_input" not in st.session_state:
    st.session_state["question_input"] = ""

if selected_sample:
    st.session_state["question_input"] = selected_sample

question = st.text_area(
    "Your medical question:",
    value=st.session_state["question_input"],
    height=90,
    placeholder="e.g. What are the early warning signs of Alzheimer's disease?",
    key="question_textarea",
)

ask_button = st.button("🔍 Get Verified Answer", type="primary", use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────────────────────────────────────
def call_api(endpoint: str, question: str, top_k: int) -> dict | None:
    """Call the FastAPI backend."""
    try:
        resp = requests.post(
            f"{API_BASE}{endpoint}",
            json={"question": question, "top_k": top_k},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error(
            "❌ Cannot connect to the API backend. "
            "Make sure the API is running: `uvicorn src.api.main:app --reload`"
        )
        return None
    except requests.exceptions.Timeout:
        st.error("⏱️ The request timed out. The model may still be loading — please retry.")
        return None
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def render_score_card(score: float) -> None:
    """Render the groundedness score with color-coded styling."""
    pct = round(score * 100, 1)
    if score >= 0.75:
        css_class = "score-high"
        emoji = "✅"
        label = "Highly Grounded"
    elif score >= 0.5:
        css_class = "score-mid"
        emoji = "⚠️"
        label = "Partially Grounded"
    else:
        css_class = "score-low"
        emoji = "🚨"
        label = "Low Groundedness"

    st.markdown(
        f"""
        <div class="score-card {css_class}">
            {emoji} {pct}%
            <div style="font-size:0.9rem; font-weight:400; margin-top:4px; opacity:0.85;">{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sentences(sentences: list[dict]) -> None:
    """Render the annotated answer sentences."""
    for s in sentences:
        label = s.get("support_label", "unverified")
        confidence = s.get("confidence", 0.0)
        text = s.get("text", "")

        if label == "supported":
            badge = '<span class="badge-supported">✓ Supported</span>'
            css = "sentence-supported"
        elif label == "no_info":
            badge = '<span class="badge-noinfo">ℹ Info</span>'
            css = "sentence-noinfo"
        else:
            badge = '<span class="badge-unsupported">⚠ Flagged</span>'
            css = "sentence-unsupported"

        conf_str = f" · {confidence:.0%}" if label not in ("no_info", "unverified") else ""

        st.markdown(
            f"""
            <div class="{css}">
                {badge}{conf_str}&nbsp;&nbsp;{text}
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_sources(chunks: list[dict]) -> None:
    """Render retrieved source passages."""
    for i, chunk in enumerate(chunks, 1):
        sim = chunk.get("similarity", 0.0)
        source = chunk.get("source_url") or chunk.get("focus_area") or "Unknown"
        text = chunk.get("text", "")[:300]

        st.markdown(
            f"""
            <div class="chunk-card">
                <span class="chunk-score">📄 Passage {i} · Similarity: {sim:.0%}</span>
                <br/><span class="chunk-source">🔗 {source}</span>
                <p class="chunk-text">{text}{'…' if len(chunk.get('text','')) > 300 else ''}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Main answer flow
# ──────────────────────────────────────────────────────────────────────────────
if ask_button and question.strip():
    st.session_state["question_input"] = question

    if compare_mode:
        # Side-by-side comparison
        col_plain, col_verified = st.columns(2)

        with col_plain:
            st.markdown("#### 🔵 Plain RAG (no verification)")
            with st.spinner("Generating plain answer..."):
                plain_data = call_api("/ask/plain", question, top_k)
            if plain_data:
                render_score_card(plain_data.get("overall_groundedness_score", 0))
                render_sentences(plain_data.get("sentences", []))

        with col_verified:
            st.markdown("#### 🟢 Verified RAG (with NLI)")
            with st.spinner("Generating and verifying answer..."):
                verified_data = call_api("/ask", question, top_k)
            if verified_data:
                render_score_card(verified_data.get("overall_groundedness_score", 0))
                render_sentences(verified_data.get("sentences", []))

        if plain_data and verified_data:
            plain_h = 1.0 - plain_data.get("overall_groundedness_score", 0)
            verified_h = 1.0 - verified_data.get("overall_groundedness_score", 0)
            st.info(
                f"📊 Hallucination rate: Plain RAG = {plain_h:.0%}  →  "
                f"Verified RAG = {verified_h:.0%}  "
                f"({'▼ ' + f'{abs(plain_h - verified_h):.0%} reduction' if verified_h < plain_h else '▲ ' + f'{abs(plain_h - verified_h):.0%} increase'})"
            )

    else:
        # Standard verified mode
        with st.spinner("🔍 Retrieving · 💬 Generating · 🔬 Verifying claims..."):
            data = call_api("/ask", question, top_k)

        if data:
            latency = data.get("latency_seconds", 0)

            # Score + metadata
            meta_col, score_col = st.columns([3, 1])
            with meta_col:
                st.markdown(f"**Query:** _{question}_")
                st.caption(f"⏱️ {latency:.1f}s · {len(data.get('sentences', []))} sentences verified")
            with score_col:
                render_score_card(data.get("overall_groundedness_score", 0))

            # Annotated answer
            st.markdown("### 📝 Verified Answer")
            render_sentences(data.get("sentences", []))

            # Legend
            st.markdown(
                "<div class='disclaimer'>"
                "<strong>Legend:</strong> "
                "<span class='badge-supported'>✓ Supported</span> — entailed by retrieved context · "
                "<span class='badge-unsupported'>⚠ Flagged</span> — not traceable to context "
                "(does not necessarily mean wrong) · "
                "<span class='badge-noinfo'>ℹ Info</span> — system acknowledged insufficient context"
                "</div>",
                unsafe_allow_html=True,
            )

            # Source passages
            if show_sources:
                with st.expander("📚 Retrieved Source Passages", expanded=False):
                    render_sources(data.get("retrieved_chunks", []))

elif ask_button and not question.strip():
    st.warning("Please enter a question before clicking 'Get Verified Answer'.")
