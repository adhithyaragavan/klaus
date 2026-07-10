# Klaus

[![tests](https://github.com/adhithyaragavan/klaus/actions/workflows/tests.yml/badge.svg)](https://github.com/adhithyaragavan/klaus/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**On-prem, citation-grounded document compliance assistant.**

Klaus answers natural-language questions about a set of contracts and never returns a claim
without a traceable source — every answer includes the exact document and clause it came from.
It's built to run entirely inside an organization's own infrastructure: ingest, embedding,
retrieval, and generation all happen with zero outbound network calls in the default
configuration — which isn't just a technical nicety, it's what makes the product usable at all
for buyers under real legal constraints: a HIPAA Business Associate Agreement most consumer LLM
APIs don't cover by default, GDPR data-residency and Standard Contractual Clause requirements,
or CMMC/ITAR air-gap mandates for defense and government contractors.

Full system design lives in [`ARCHITECTURE.md`](ARCHITECTURE.md).

## Why citation-grounding, not drafting speed

Most AI contract tools compete on how fast they draft or redline. That's a real feature, but it
sidesteps the actual adoption blocker: **92% of contract-management errors are human error**,
and organizations lose an average of **8.6% of total contract spend annually to cost leakage**
from exactly the kind of missed clause a fast-but-unverifiable AI summary would also miss.
Meanwhile **80% of procurement teams already use AI in contracting** — the market isn't
AI-skeptical, it's evidence-skeptical. The open question for regulated buyers isn't "can AI read
my contracts," it's "can I trust an answer I didn't verify myself."

Klaus's answer is structural, not a prompting trick: every claim is checked against real
retrieved clause metadata before it's returned, a failed citation gets one retry then a clear
refusal rather than a guess, and every query — successful or rejected — is written to an
append-only audit log. That's the trust mechanism a compliance team actually needs before they
can adopt AI-assisted review at all, not a speed optimization on top of a workflow they already
trust.

> **AMD compute usage.** The production backend (`ROCmVLLMBackend`) serves `Qwen/Qwen3-14B` via
> `vLLM` on AMD ROCm hardware (verified against a Radeon/RDNA3, `gfx1100`, instance) — this is
> what makes the "nothing leaves the box" guarantee a verified fact rather than a design claim.
> A real, captured end-to-end run (server startup logs, hardware specs, verified grounded Q&A,
> and the actual append-only audit log entry) is documented in
> [`docs/AMD_VERIFICATION.md`](docs/AMD_VERIFICATION.md).

**Example domain: vendor contract compliance review.** `data/sample_contracts/` contains 7
synthetic vendor agreements (NDA, MSA, DPA, SaaS subscription, consulting, supply, cloud hosting)
covering confidentiality, termination, liability-cap, and data-handling clauses — including two
deliberately built-in compliance risks (a vendor with a vague, undefined data-breach notification
window, and a vendor with explicitly uncapped liability) to exercise the citation-grounding
guarantees against realistic edge cases.

## Setup

Requires Python 3.11+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm

cp .env.example .env
# edit .env to configure an LLM backend — see "LLM Backends" below
```

## Usage

```bash
bash scripts/run.sh
```

This loads `data/sample_contracts/`, runs the PII/PHI scan, builds the hybrid index, and drops
you into an interactive query loop. Try asking:

- `What is the data-breach notification window in each vendor contract?`
- `Which contracts lack a liability cap, and what is the exposure?`
- `Summarize the confidentiality obligations that survive termination.`

Every answer is printed with its citations (document + clause) and appended to
`audit_log/klaus_audit.jsonl`. Type `exit` to quit.

> **No AMD ROCm/vLLM instance handy?** The default backend (`ROCmVLLMBackend`) is the primary,
> production-grade path, but it needs a real vLLM server reachable at `ROCM_VLLM_BASE_URL`.
> To try Klaus locally without one, set `KLAUS_LLM_BACKEND=local_dev` in `.env` and point
> `OLLAMA_BASE_URL`/`OLLAMA_MODEL` at a local [Ollama](https://ollama.com) instance. This is the
> offline-iteration-only path (see `LLM Backends` below) — fine for trying out retrieval and
> citation grounding, but it runs a much smaller model than the production backend.

## Live Demo

A free-hosted, lightweight public preview runs the real retrieval + citation-grounding pipeline
(`preview_app/`) — clearly labeled as a preview, since it swaps the verified AMD backend for
Groq's free API (Streamlit's free tier has no GPU). See
[`docs/AMD_VERIFICATION.md`](docs/AMD_VERIFICATION.md) for the actual AMD-backed run.

You can either query the bundled sample contracts, or **upload your own `.txt`/`.pdf` document**
and query it directly — the same real ingest → chunk → index → retrieve → generate pipeline runs
against whatever you upload, in-session, nothing precomputed.

- **Hosted URL**: [jmfqee34jqzxmdav9bxqws.streamlit.app](https://jmfqee34jqzxmdav9bxqws.streamlit.app)
  (Community Cloud apps sleep after inactivity — first load after a while may take ~30-60s to wake up.)
- **Run it yourself**:
  ```bash
  pip install -r preview_app/requirements.txt
  # requires a free Groq API key (https://console.groq.com) set as GROQ_API_KEY in
  # .streamlit/secrets.toml (gitignored) or as an environment variable
  streamlit run preview_app/streamlit_app.py
  ```

### Via Docker

Docker is provided for production-completeness, not because it's required for submission —
Track 3 explicitly does not require a container image.

```bash
# note the explicit --platform: required if building on Apple Silicon, since the target
# deployment platform is linux/amd64
docker buildx build --platform linux/amd64 -t klaus:latest -f docker/Dockerfile .
docker run --rm -it --env-file .env klaus:latest
```

The image is ~525MB (well under the 10GB budget), built on `python:3.11-slim` with the CPU-only
torch wheel to avoid pulling in unused CUDA runtime libraries.

### Running tests

```bash
pytest tests/
```

## LLM Backends

Klaus's generation layer sits behind a swap-friendly `LLMBackend` protocol
(`src/generate.py`) with two concrete implementations, selected via `KLAUS_LLM_BACKEND`:

| Backend | `KLAUS_LLM_BACKEND` | Role | Network |
|---|---|---|---|
| `ROCmVLLMBackend` | `rocm_vllm` (default) | **Primary, production backend.** Serves `Qwen/Qwen3-14B` via vLLM on AMD ROCm hardware (verified against a Radeon/RDNA3, gfx1100, instance). This is what makes "nothing leaves the box" a verified, tested fact rather than a design claim. | On-prem only |
| `LocalDevBackend` | `local_dev` | Offline iteration only, via a local Ollama instance — fast local dev loop for retrieval/chunking/citation logic without needing a GPU. **Not intended for production use.** | Local only |

See `.env.example` for the full set of environment variables each backend reads (base URLs,
model names, API key). No key or secret is ever hardcoded — everything comes from the
environment.

## Current Scope vs. Production Roadmap

| Area | Current implementation | Production roadmap |
|---|---|---|
| Core pipeline network calls | Zero, by design (ingest → embed → retrieve → generate) | Same — this is a hard requirement, not a simplification |
| Primary LLM backend | `ROCmVLLMBackend` (`Qwen/Qwen3-14B`) against an AMD Radeon/RDNA3 instance | Same backend/approach: `vLLM` on ROCm, scaling to AMD Instinct/CDNA GPUs for larger deployments |
| Vector store | FAISS flat index, in-memory, rebuilt on every run | Persistent, likely distributed vector store for larger corpora |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` — small, CPU-friendly, sufficient for the ~100-clause sample corpus | Possibly a larger or domain-tuned embedding model |
| PII/PHI scan | Regex (SSN/email/phone) + spaCy `en_core_web_sm` NER, detect-and-log only — the small NER model does produce some false positives (e.g. flagging "AES-256" as a person name), which a human reviewer would triage | Larger/more accurate NER, likely with redaction workflows, not just detection |
| Audit log | Local append-only JSONL file | Same append-only guarantee, but production would add replication/rotation/backup |
| Multi-tenancy | None — single corpus, single tenant | Per-customer/per-department index isolation (see `ARCHITECTURE.md` §8) |

## Track 3 (Unicorn) Compliance

| Requirement | Status |
|---|---|
| GitHub repository URL | This repo |
| AMD compute usage demonstrated | Yes — real verified run in [`docs/AMD_VERIFICATION.md`](docs/AMD_VERIFICATION.md); production backend is `ROCmVLLMBackend` (`Qwen/Qwen3-14B` on AMD ROCm) |
| Live demo / hosted URL (optional) | Free-hosted preview, see "Live Demo" above |
| No hardcoded/cached answers | Every query runs a real `retrieve()` + `generate_answer()` call, in both the core CLI and the hosted preview — no answer lookup tables anywhere |
| English-only responses | Yes |
| Docker image | Not required for Track 3; provided anyway for production-completeness (see "Via Docker" above) |

## Project layout

```
src/ingest.py            - file loading (.txt, .pdf)
src/chunker.py           - clause/section splitting
src/pii_scan.py          - PII/PHI detection (regex + spaCy)
src/index.py             - hybrid (dense + BM25) index build
src/retrieve.py          - hybrid retrieval
src/generate.py          - LLM backends + structured citation output
src/audit_log.py         - append-only query/answer logging
src/app.py               - CLI orchestration
docs/AMD_VERIFICATION.md - real, captured evidence of the AMD ROCm/vLLM run
preview_app/             - free-hosted lightweight public preview (Streamlit + Groq)
```

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the full system design, component interfaces, and
data flow.
