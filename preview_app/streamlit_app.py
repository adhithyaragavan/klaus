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

import html
import re
import sys
import tempfile
import textwrap
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

# --- Visual identity: matches the pitch deck's ink / gold / verified-sage palette ---
# NOTE: Streamlit's markdown renderer terminates a raw <style>...</style> passthrough block at
# the first blank line inside it (rather than continuing to the closing tag) -- so this CSS is
# deliberately written with zero blank lines between rules. Don't reformat with blank-line
# separation between rule blocks, it will silently break and spill CSS text onto the page.
_KLAUS_CSS = """<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root {
    --bg: #14171C;
    --panel: #1B2028;
    --border: #333840;
    --text: #ECE8E0;
    --muted: #8A8578;
    --gold: #C98A3D;
    --verified: #6FAE8C;
}
/* hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }
[data-testid="stDecoration"] { display: none; }
.stApp {
    background: var(--bg);
    color: var(--text);
}
.stApp, .stApp p, .stApp label, .stMarkdown, .stCaption {
    font-family: "IBM Plex Sans", -apple-system, sans-serif;
}
.stApp span:not([data-testid="stIconMaterial"]) {
    font-family: "IBM Plex Sans", -apple-system, sans-serif;
}
.klaus-topbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 4px 0 20px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 26px;
    font-family: "IBM Plex Mono", monospace;
    font-size: 11px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--muted);
}
.klaus-h1 {
    font-family: Georgia, "Iowan Old Style", serif;
    font-weight: 400;
    font-size: 2.4rem;
    color: var(--text);
    margin: 0 0 20px;
    line-height: 1.1;
}
.klaus-answer {
    font-family: Georgia, "Iowan Old Style", serif;
    font-size: 1.08rem;
    line-height: 1.6;
    color: var(--text);
    white-space: pre-wrap;
}
.klaus-panel {
    background: var(--panel);
    border: 1px solid var(--border);
    border-left: 3px solid var(--gold);
    border-radius: 4px;
    padding: 16px 20px;
    margin-bottom: 22px;
}
.klaus-panel .klaus-label {
    font-family: "IBM Plex Mono", monospace;
    font-size: 11px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--gold);
    margin-bottom: 8px;
}
.klaus-panel p {
    font-family: "IBM Plex Sans", sans-serif;
    font-size: 0.92rem;
    color: var(--muted);
    line-height: 1.55;
    margin: 0;
}
.klaus-panel a { color: var(--text); }
.klaus-panel code {
    background: var(--bg);
    color: var(--text);
    border-radius: 3px;
    padding: 1px 5px;
}
.klaus-eyebrow {
    font-family: "IBM Plex Mono", monospace;
    font-size: 11px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--muted);
    margin: 22px 0 8px;
}
.klaus-citations { display: flex; flex-direction: column; gap: 10px; margin-top: 6px; }
.klaus-citation-row { display: flex; flex-wrap: wrap; align-items: baseline; gap: 10px; }
.klaus-badge {
    font-family: "IBM Plex Mono", monospace;
    font-size: 12px;
    color: var(--verified);
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 4px 12px;
    white-space: nowrap;
}
.klaus-citation-quote {
    font-family: "IBM Plex Sans", sans-serif;
    font-size: 0.88rem;
    color: var(--muted);
}
.stTextInput input {
    background: var(--panel) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    font-family: "IBM Plex Sans", sans-serif !important;
}
.stTextInput label, [data-testid="stCaptionContainer"], [data-testid="stFileUploader"] label {
    font-family: "IBM Plex Sans", sans-serif !important;
    color: var(--muted) !important;
}
.stButton > button, button[kind="primary"] {
    background: var(--panel) !important;
    color: var(--gold) !important;
    border: 1px solid var(--gold) !important;
    font-family: "IBM Plex Mono", monospace !important;
    letter-spacing: 0.04em;
    border-radius: 4px !important;
    transition: background 0.15s ease, color 0.15s ease;
}
.stButton > button:hover, button[kind="primary"]:hover {
    background: var(--gold) !important;
    color: var(--bg) !important;
    border-color: var(--gold) !important;
}
[data-testid="stFileUploader"] section {
    background: var(--panel) !important;
    border: 1px dashed var(--border) !important;
}
[data-testid="stFileUploaderFile"], [data-testid="stFileUploaderFileName"] {
    background: var(--panel) !important;
    border-radius: 4px;
    font-family: "IBM Plex Sans", sans-serif !important;
    color: var(--text) !important;
}
</style>"""

st.markdown(_KLAUS_CSS, unsafe_allow_html=True)

st.markdown(
    textwrap.dedent(
        """
    <div class="klaus-topbar">
        <span>Klaus</span>
        <span>Track 3 · Preview backend: Groq</span>
    </div>
    """
    ),
    unsafe_allow_html=True,
)

st.markdown('<h1 class="klaus-h1">Contract Compliance Preview</h1>', unsafe_allow_html=True)

st.markdown(
    textwrap.dedent(
        """
    <div class="klaus-panel">
      <div class="klaus-label">Preview — not production</div>
      <p>This is a free, lightweight public preview — it runs the same retrieval and
      citation-grounding pipeline as production, but calls Groq's free API instead of an AMD
      GPU (Streamlit's free tier has no GPU). The verified production backend serves
      <code>Qwen/Qwen3-14B</code> via vLLM on real AMD ROCm hardware — see
      <a href="https://github.com/adhithyaragavan/klaus/blob/main/docs/AMD_VERIFICATION.md" target="_blank">docs/AMD_VERIFICATION.md</a>
      for the actual captured run.</p>
    </div>
    """
    ),
    unsafe_allow_html=True,
)

st.write(
    "Ask a question about the sample vendor contracts, or upload your own document below. "
    "Every answer includes a citation to the exact document and clause it came from — Klaus "
    "never answers from general knowledge."
)


@st.cache_resource(show_spinner="Loading contracts and building index (one-time)...")
def get_index():
    documents = load_documents(DATA_DIR)
    clauses = [clause for document in documents for clause in chunk_document(document)]
    return build_index(clauses), len(documents), len(clauses)


@st.cache_resource(show_spinner="Indexing your uploaded document(s)...")
def build_uploaded_index(file_data: tuple[tuple[str, bytes], ...]):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        for name, data in file_data:
            (tmp_path / name).write_bytes(data)
        documents = load_documents(tmpdir)
        clauses = [clause for document in documents for clause in chunk_document(document)]
        return build_index(clauses), len(documents), len(clauses)


uploaded_files = st.file_uploader(
    "Upload your own contract(s) (.txt or .pdf) to query instead of the sample corpus",
    type=["txt", "pdf"],
    accept_multiple_files=True,
)

if uploaded_files:
    file_data = tuple((f.name, f.getvalue()) for f in uploaded_files)
    index, doc_count, clause_count = build_uploaded_index(file_data)
    st.caption(f"Indexed {doc_count} uploaded document(s), {clause_count} clauses.")
    using_sample_corpus = False
else:
    index, doc_count, clause_count = get_index()
    st.caption(f"Using the sample corpus: {doc_count} contracts, {clause_count} clauses.")
    using_sample_corpus = True

api_key = st.secrets.get("GROQ_API_KEY")
model = st.secrets.get("GROQ_MODEL", DEFAULT_MODEL)

example_queries = [
    "What is the data-breach notification window in each vendor contract?",
    "Which contracts lack a liability cap, and what is the exposure?",
    "Summarize the confidentiality obligations that survive termination.",
]
query = st.text_input("Your question", placeholder=example_queries[0])
if using_sample_corpus:
    st.caption("Try: " + " · ".join(f"*{q}*" for q in example_queries))

_CLAUSE_SHORT_RE = re.compile(r"^(Section|Article)\s+", re.IGNORECASE)
_DOC_EXT_RE = re.compile(r"\.(txt|pdf)$", re.IGNORECASE)


def _badge_label(document: str, clause: str) -> str:
    short_doc = _DOC_EXT_RE.sub("", document)
    short_clause = "§" + _CLAUSE_SHORT_RE.sub("", clause)
    return f"{short_doc} {short_clause}"


if st.button("Ask", type="primary") and query:
    if not api_key:
        st.error("GROQ_API_KEY is not configured for this deployment.")
    else:
        with st.spinner("Retrieving relevant clauses and generating a grounded answer..."):
            retrieved = retrieve(query, index, top_k=TOP_K)
            answer = generate_answer(query, retrieved, backend=GroqPreviewBackend(api_key, model))

        st.markdown('<div class="klaus-eyebrow">Answer</div>', unsafe_allow_html=True)
        # answer.answer comes from the LLM (indirectly influenced by uploaded, user-controlled
        # document content) — escape before interpolating into unsafe HTML.
        st.markdown(
            f'<div class="klaus-answer">{html.escape(answer.answer)}</div>',
            unsafe_allow_html=True,
        )

        if answer.citations:
            st.markdown('<div class="klaus-eyebrow">Citations</div>', unsafe_allow_html=True)
            rows = "".join(
                '<div class="klaus-citation-row">'
                f'<span class="klaus-badge">{html.escape(_badge_label(c.document, c.clause))}</span>'
                f'<span class="klaus-citation-quote">{html.escape(c.quote_or_paraphrase)}</span>'
                "</div>"
                for c in answer.citations
            )
            st.markdown(f'<div class="klaus-citations">{rows}</div>', unsafe_allow_html=True)
        else:
            st.markdown(
                '<div class="klaus-panel"><p>No grounded citation was found for this question.</p></div>',
                unsafe_allow_html=True,
            )
