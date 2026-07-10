# Klaus — System Architecture

On-prem, citation-grounded document compliance intelligence. This document describes Klaus's
system architecture — components, data flow, interfaces, and constraints — as the reference for
how the pieces fit together and why.

---

## 1. Design Principles

1. **Nothing leaves the box.** Every component in the core path (embedding, retrieval, generation) must be runnable with zero outbound network calls. There is no cloud LLM path in this build.
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
                         └──────────────────────────────────────────────┘
```

---

## 3. Components

### 3.1 Ingest Loader (`src/ingest.py`)
- Input: a directory of contract files (PDF or `.txt`).
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
- Responsibility: split on section/clause numbering and headers (regex-based patterns like `^\d+(\.\d+)*\s`, `^Section \d+`, `^Article \w+`). Falls back to paragraph-level splitting if no numbering is detected — but flags those chunks as `"granularity": "paragraph"` so downstream citation is honest about precision.

### 3.3 PII/PHI Scan (`src/pii_scan.py`)
- Input: list of `Clause` objects.
- Output: a warnings list `[{"doc_id", "clause_id", "span", "type"}]` — does not modify or block ingestion, just surfaces findings (detection-and-log, not redaction).
- Method: regex for structured PII (SSN, email, phone) + spaCy `en_core_web_sm` NER for named individuals.

### 3.4 Hybrid Index Build (`src/index.py`)
- Input: list of `Clause` objects (post-PII-scan).
- Output: two aligned indices sharing the same clause-id keyspace:
  - Dense: `sentence-transformers/all-MiniLM-L6-v2` embeddings in Chroma (or a flat FAISS index).
  - Sparse: BM25 index (e.g. `rank_bm25`) over the same clause text.
- Interface: `build_index(clauses: list[Clause]) -> IndexHandle`

### 3.5 Retriever (`src/retrieve.py`)
- Input: query string, `IndexHandle`, `top_k` (default 6).
- Output: ranked list of `Clause` + score, combining dense and BM25 results (normalize both score sets 0-1 and take a weighted sum, currently 0.6 dense / 0.4 BM25).
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
  Two concrete implementations sit behind this protocol:

  1. **`ROCmVLLMBackend` — primary, on-prem production backend.** Serves `Qwen/Qwen3-14B` via
     `vLLM` on AMD ROCm hardware (verified against a Radeon/RDNA3, gfx1100, instance) — chosen
     over Gemma because vLLM has native, first-class support for Qwen3 (Gemma 4 was too new at
     the time this was built to have anything but a generic, bug-prone fallback implementation).
     This is the backend that makes the "nothing leaves the box" claim a verified fact rather
     than a design assumption.
  2. **`LocalDevBackend` — offline iteration only.** `llama-cpp-python` or Ollama with any small
     GGUF model. Used for fast local iteration on retrieval/chunking/citation logic without
     needing GPU access; not intended for production use.

### 3.7 Audit Logger (`src/audit_log.py`)
- Input: query, retrieved clauses, generated answer, timestamp.
- Output: one JSONL line appended to `audit_log/klaus_audit.jsonl`:
  ```json
  {"timestamp": "...", "query": "...", "retrieved_clause_ids": [...], "answer": {...}, "backend": "rocm_vllm"}
  ```
- Responsibility: append-only, never overwrite or mutate prior entries.

### 3.8 App / Entry Point (`src/app.py`)
- Wires the above into a CLI query loop, and optionally a minimal FastAPI endpoint (`POST /query`).
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
| Local dev (offline iteration only) | `LocalDevBackend` — `llama-cpp-python` / Ollama, any small GGUF | Fast iteration on retrieval/chunking/citation logic; not the production backend |
| On-prem reference deployment | `ROCmVLLMBackend` — `vLLM` serving `Qwen/Qwen3-14B` on an AMD Radeon/RDNA3 (gfx1100) instance | **Primary, verified backend.** Demonstrates "nothing leaves the box" in practice, not just as a design claim; container builds for `linux/amd64` |
| Production target | `ROCmVLLMBackend` — `vLLM` on ROCm against AMD GPUs (Radeon/RDNA3 verified; Instinct/CDNA datacenter cards for larger-scale deployments) | Same backend as the reference deployment; this is the deployment the system's cost/privacy guarantees depend on |

---

## 6. Non-Functional Requirements

- **No hardcoded secrets.** All API keys/config via environment variables (`.env.example` documents every one).
- **Containerized.** `docker/Dockerfile` builds a `linux/amd64` image that runs the full pipeline end-to-end via `scripts/run.sh`.
- **Deterministic-ish citations.** Citation clause-ids must reference real `clause_id` values from the index — never fabricated section numbers.
- **Graceful degradation.** If retrieval returns nothing relevant, the generator must say so rather than hallucinating an answer.

---

## 7. Directory Structure

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
  scripts/run.sh
  tests/test_pipeline.py
```

---

## 8. Open Questions / Future Architecture

- Multi-tenant isolation for the "Enterprise" tier (separate indices per department/customer).
- Replacing regex-based clause splitting with a fine-tuned section-boundary detector for messier real-world contracts.
- Formal evaluation harness (precision/recall on citation correctness against a labeled contract set).
