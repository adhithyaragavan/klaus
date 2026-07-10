"""Free-hosted, lightweight preview of Klaus's retrieval + citation-grounding pipeline.

Reuses the real pipeline (src.ingest, src.chunker, src.index, src.retrieve,
src.generate.generate_answer) — same citation-grounding and reject/retry logic as production.
The only thing swapped out is the LLM backend: this preview calls Groq's free API instead of
the AMD ROCm/vLLM backend, since Streamlit Community Cloud has no GPU. See docs/AMD_VERIFICATION.md
for the real, verified AMD-backed run.

Every query below triggers a genuine retrieve() + generate_answer() call — nothing here is
cached or hardcoded per-input.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Streamlit Community Cloud always runs from the repo root, but make this resilient either way.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from groq_backend import DEFAULT_MODEL, GroqPreviewBackend  # noqa: E402
from src.chunker import chunk_document  # noqa: E402
from src.generate import generate_answer  # noqa: E402
from src.index import build_index  # noqa: E402
from src.ingest import load_documents  # noqa: E402
from src.retrieve import retrieve  # noqa: E402

DATA_DIR = "data/sample_contracts"
TOP_K = 6

st.set_page_config(page_title="Klaus — Contract Compliance Preview", page_icon="📄")

st.title("Klaus — Contract Compliance Preview")

st.warning(
    "**This is a free, lightweight public preview** — it runs the same retrieval and "
    "citation-grounding pipeline as production, but calls Groq's free API instead of an AMD "
    "GPU (Streamlit's free tier has no GPU). The verified production backend serves "
    "`Qwen/Qwen3-14B` via vLLM on real AMD ROCm hardware — see "
    "[docs/AMD_VERIFICATION.md](https://github.com/adhithyaragavan/klaus/blob/main/docs/AMD_VERIFICATION.md) "
    "for the actual captured run.",
    icon="⚠️",
)

st.write(
    "Ask a question about the sample vendor contracts below. Every answer includes a citation "
    "to the exact document and clause it came from — Klaus never answers from general knowledge."
)


@st.cache_resource(show_spinner="Loading contracts and building index (one-time)...")
def get_index():
    documents = load_documents(DATA_DIR)
    clauses = [clause for document in documents for clause in chunk_document(document)]
    return build_index(clauses), len(documents), len(clauses)


index, doc_count, clause_count = get_index()
st.caption(f"Loaded {doc_count} sample contracts, {clause_count} clauses.")

api_key = st.secrets.get("GROQ_API_KEY")
model = st.secrets.get("GROQ_MODEL", DEFAULT_MODEL)

example_queries = [
    "What is the data-breach notification window in each vendor contract?",
    "Which contracts lack a liability cap, and what is the exposure?",
    "Summarize the confidentiality obligations that survive termination.",
]
query = st.text_input("Your question", placeholder=example_queries[0])
st.caption("Try: " + " · ".join(f"*{q}*" for q in example_queries))

if st.button("Ask", type="primary") and query:
    if not api_key:
        st.error("GROQ_API_KEY is not configured for this deployment.")
    else:
        with st.spinner("Retrieving relevant clauses and generating a grounded answer..."):
            retrieved = retrieve(query, index, top_k=TOP_K)
            answer = generate_answer(query, retrieved, backend=GroqPreviewBackend(api_key, model))

        st.subheader("Answer")
        st.write(answer.answer)

        if answer.citations:
            st.subheader("Citations")
            for citation in answer.citations:
                st.markdown(f"- **{citation.document}** {citation.clause}: {citation.quote_or_paraphrase}")
        else:
            st.info("No grounded citation was found for this question.")
