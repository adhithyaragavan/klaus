# Klaus — System Architecture

On-prem, citation-grounded document compliance intelligence. This document is the technical spec Claude Code should build against — it defines components, data flow, interfaces, and constraints so implementation stays consistent across build sessions.

---

## 1. Design Principles

1. **Nothing leaves the box.** Every component in the core path (embedding, retrieval, generation) must be runnable with zero outbound network calls. Any cloud call (e.g. optional Fireworks/Gemma routing) is an explicit, opt-in side-path — never the only path.
2. **No un-cited answers.** The generation layer must refuse to return a claim without a traceable source (document + clause/section id). This is enforced structurally (schema-level), not just prompted for.
3. **Audit everything.** Every query, retrieval, and answer is logged immutably. The audit log is a first-class output, not an afterthought.
4. **Clause-level, not chunk-level.** Retrieval granularity is the actual unit a compliance reviewer thinks in (a clause, a section) — not an arbitrary fixed-token window.
5. **Swap-friendly compute.** The LLM and embedding backends sit behind a thin interface so the same pipeline runs on `llama-cpp`/Ollama in dev and on ROCm/vLLM in production without touching business logic.

---

## 2. High-Level Diagram

```
                         ┌─────────────────────────────────────────┐
                         │              Customer Perimeter          │
                         │                                          │
   ┌──────────┐          │   ┌──────────┐    ┌────────────────┐     │
   │  Contract │  ingest  │   │  Ingest  │───▶│  Chunker /      │     │
   │  Files    │─────────▶│   │  Loader  │    │  Section Parser │     │
   │ (PDF/txt) │          │   └──────────┘    └────────┬────────┘     │
   └──────────┘          │                             │              │
                         │                             ▼              │
                         │                    ┌──────────────────┐    │
                         │                    │   PII/PHI Scan    │    │
                         │                    └────────┬─────────┘    │
                         │                             ▼              │
                         │                 ┌───────────────────────┐  │
                         │                 │   Hybrid Index Build   │  │
                         │                 │  (dense + BM25)        │  │
                         │                 └───────────┬───────────┘  │
                         │                             │               │
              query      │                             ▼               │
   ┌──────────┐          │                 ┌───────────────────────┐  │
   │   User    │─────────▶│                 │   Retriever            │  │
   │  Query    │          │                 │ (hybrid search, top-k) │  │
   └──────────┘          │                 └───────────┬───────────┘  │
                         │                             │               │
                         │                             ▼               │
                         │                 ┌───────────────────────┐  │
                         │                 │  Generator (local LLM) │  │
                         │                 │  → structured JSON     │  │
                         │                 │    answer + citations  │  │
                         │                 └───────────┬───────────┘  │
                         │                             │               │
                         │                 ┌───────────▼───────────┐  │
                         │                 │     Audit Logger        │  │
                         │                 │   (append-only JSONL)   │  │
                         │                 └─────────────────────────┘  │
                         │                                              │
                         └──────────────────────────┬───────────────────┘
                                                     │ (optional, opt-in)
                                                     ▼
                                        ┌────────────────────────┐
                                        │ Fireworks AI (Gemma)     │
                                        │ non-sensitive triage tier│
                                        └────────────────────────┘
```

---

## 3. Components

### 3.1 Ingest Loader (`src/ingest.py`)
- Input: a directory of contract files (PDF or `.txt` for the hackathon demo).
- Output: raw text per document, with page boundaries preserved where available.
- Responsibility: file I/O only. No chunking logic here.

### 3.2 Chunker / Section Parser (`src/ingest.py` or `src/chunker.py`)
- Input: raw document text.
- Output: list of `Clause` objects:
  ```json
  {
    "doc_id": "vendor_acme_nda.txt",
    "clause_id": "Section 4.2",
    "text": "...",
    "page": 3
  }
  ```
- Responsibility: split on section/clause numbering and headers (regex-based is fine for the demo: patterns like `^\d+(\.\d+)*\s`, `^Section \d+`, `^Article \w+`). Falls back to paragraph-level splitting if no numbering is detected — but flag those chunks as `"granularity": "paragraph"` so downstream citation is honest about precision.

### 3.3 PII/PHI Scan (`src/pii_scan.py`)
- Input: list of `Clause` objects.
- Output: a warnings list `[{"doc_id", "clause_id", "span", "type"}]` — does not modify or block ingestion, just surfaces findings (this is a detection-and-log demo, not redaction).
- Method: regex for structured PII (SSN, email, phone) + spaCy `en_core_web_sm` NER for named individuals.

### 3.4 Hybrid Index Build (`src/index.py`)
- Input: list of `Clause` objects (post-PII-scan).
- Output: two aligned indices sharing the same clause-id keyspace:
  - Dense: `sentence-transformers/all-MiniLM-L6-v2` embeddings in Chroma (or a flat FAISS index).
  - Sparse: BM25 index (e.g. `rank_bm25`) over the same clause text.
- Interface: `build_index(clauses: list[Clause]) -> IndexHandle`

### 3.5 Retriever (`src/retrieve.py`)
- Input: query string, `IndexHandle`, `top_k` (default 6).
- Output: ranked list of `Clause` + score, combining dense and BM25 results (simple approach: normalize both score sets 0-1 and take a weighted sum, e.g. 0.6 dense / 0.4 BM25 — tune if time allows, don't over-engineer).
- Interface: `retrieve(query: str, index: IndexHandle, top_k: int = 6) -> list[RetrievedClause]`

### 3.6 Generator (`src/generate.py`)
- Input: query string, `list[RetrievedClause]`.
- Output: structured JSON:
  ```json
  {
    "answer": "...",
    "citations": [
      {"document": "vendor_acme_nda.txt", "clause": "Section 4.2", "quote_or_paraphrase": "..."}
    ]
  }
  ```
- Responsibility:
  - Build the prompt so the model is instructed to answer **only** from the retrieved clauses and to cite every claim.
  - Parse and validate the model's output against the schema above.
  - **Reject and retry once** (with a stricter instruction) if the output has zero citations or fails to parse; if it still fails, return a clear "insufficient grounded information" response rather than an ungrounded answer.
- Backend interface (swap-friendly):
  ```python
  class LLMBackend(Protocol):
      def generate(self, prompt: str) -> str: ...
  ```
  Three concrete implementations sit behind this protocol:

  1. **`ROCmVLLMBackend` — primary (on-prem, the demoed default).** Serves `Qwen/Qwen3-14B`
     via `vLLM` on an AMD Developer Cloud Radeon/RDNA3 (gfx1100) instance — chosen over Gemma
     because vLLM has native, first-class support for Qwen3 (Gemma 4 was too new to have
     anything but a generic, bug-prone fallback implementation at the time of the hackathon
     build). This is the backend that *proves* the "nothing leaves the box" claim as a
     demonstrated fact, not a slide bullet — the recorded demo runs its grounded queries
     through this. Document setup/run instructions even if the AMD instance is only available
     briefly, since the production pitch depends on it.
  2. **`FireworksGemmaBackend` — optional, opt-in cloud triage tier only.** Calls Gemma via
     Fireworks AI (`gemma-4-31b-it` for answer quality on citation-critical generation;
     `gemma-4-26b-a4b-it` as a faster/cheaper option when latency matters more). Reads
     `FIREWORKS_API_KEY` and `FIREWORKS_BASE_URL` from the environment — never hardcoded.
     Gated behind `KLAUS_ALLOW_CLOUD_TRIAGE=true` and **never the default** (see Immutable
     Rule #2): sending clauses here is an outbound call, so it is reserved for explicitly
     non-sensitive queries only.
  3. **`LocalDevBackend` — offline iteration only.** `llama-cpp-python` or Ollama with any small
     GGUF model (need not be Gemma). Used only while iterating on retrieval/chunking/citation
     logic before AMD cloud access is wired up; never shown in the live/recorded demo or
     referenced in the pitch's technical claims.

### 3.7 Audit Logger (`src/audit_log.py`)
- Input: query, retrieved clauses, generated answer, timestamp.
- Output: one JSONL line appended to `audit_log/klaus_audit.jsonl`:
  ```json
  {"timestamp": "...", "query": "...", "retrieved_clause_ids": [...], "answer": {...}, "backend": "local-llama"}
  ```
- Responsibility: append-only, never overwrite or mutate prior entries.

### 3.8 App / Entry Point (`src/app.py`)
- Wires the above into a CLI query loop for the demo, and optionally a minimal FastAPI endpoint (`POST /query`) if time allows.
- Responsibility: orchestration only — no business logic lives here.

---

## 4. Data Flow (single query, sequence)

1. User submits a query via CLI or API.
2. `retrieve()` returns top-k `RetrievedClause` objects from the hybrid index.
3. `generate()` builds a grounded prompt from those clauses, calls the configured `LLMBackend`, validates/parses structured output.
4. If validation fails twice, return a graceful "not enough grounded information" response — never fall through to an ungrounded answer.
5. `audit_log()` appends the full record (query, retrieval, answer) regardless of success/failure path.
6. Response returned to the user with answer + citations.

---

## 5. Deployment Targets

| Environment | LLM backend | Notes |
|---|---|---|
| Local dev (offline iteration only) | `LocalDevBackend` — `llama-cpp-python` / Ollama, any small GGUF (need not be Gemma) | Fast iteration on retrieval/chunking/citation logic; never the demoed backend |
| Hackathon demo / on-prem proof | `ROCmVLLMBackend` — `vLLM` serving `Qwen/Qwen3-14B` on an AMD Developer Cloud Radeon/RDNA3 (gfx1100) instance | **Primary demoed backend.** Runs the grounded queries in the recorded demo to prove "nothing leaves the box"; container still builds for `linux/amd64` |
| Production target | `ROCmVLLMBackend` — `vLLM` on ROCm against AMD GPUs (Radeon/RDNA3 for the demo; Instinct/CDNA datacenter cards for larger-scale production deployments) | Same backend as the demo; this is the deployment the pitch's cost/privacy claims depend on |
| Optional triage path | `FireworksGemmaBackend` — Fireworks AI hosted Gemma (`gemma-4-31b-it`, or `gemma-4-26b-a4b-it` for lower latency) | Opt-in only, **never the default**; explicitly non-sensitive queries only, toggled via `KLAUS_ALLOW_CLOUD_TRIAGE=true`; `FIREWORKS_API_KEY` / `FIREWORKS_BASE_URL` from env |

---

## 6. Non-Functional Requirements

- **No hardcoded secrets.** All API keys/config via environment variables (`.env.example` documents every one).
- **Containerized.** `docker/Dockerfile` builds a `linux/amd64` image that runs the full pipeline end-to-end via `scripts/run_demo.sh`.
- **Deterministic-ish citations.** Citation clause-ids must reference real `clause_id` values from the index — never fabricated section numbers.
- **Graceful degradation.** If retrieval returns nothing relevant, the generator must say so rather than hallucinating an answer.

---

## 7. Directory Structure (matches build prompt)

```
klaus/
  README.md
  ARCHITECTURE.md         <- this file
  .env.example
  docker/Dockerfile
  data/sample_contracts/
  src/
    ingest.py
    chunker.py
    pii_scan.py
    index.py
    retrieve.py
    generate.py
    audit_log.py
    app.py
  scripts/run_demo.sh
  tests/test_pipeline.py
```

---

## 8. Open Questions / Future Architecture (post-hackathon)

- Multi-tenant isolation for the "Enterprise" tier (separate indices per department/customer).
- Replacing regex-based clause splitting with a fine-tuned section-boundary detector for messier real-world contracts.
- Formal evaluation harness (precision/recall on citation correctness against a labeled contract set).
