# Klaus

**On-prem, citation-grounded document compliance assistant.**

Klaus answers natural-language questions about a set of contracts and never returns a claim
without a traceable source — every answer includes the exact document and clause it came from.
It's built to run entirely inside a customer's own infrastructure: ingest, embedding, retrieval,
and generation all happen with zero outbound network calls in the default configuration.

Built for the AMD Developer Hackathon: ACT II, Track 3 (Unicorn). Full system design lives in
[`ARCHITECTURE.md`](ARCHITECTURE.md).

**Demo vertical:** vendor contract compliance review. `data/sample_contracts/` contains 7
synthetic vendor agreements (NDA, MSA, DPA, SaaS subscription, consulting, supply, cloud hosting)
covering confidentiality, termination, liability-cap, and data-handling clauses — including two
deliberately built-in compliance risks: a vendor with a vague, undefined data-breach notification
window, and a vendor with explicitly uncapped liability.

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

## Running the demo

```bash
bash scripts/run_demo.sh
```

This loads `data/sample_contracts/`, runs the PII/PHI scan, builds the hybrid index, and drops
you into an interactive query loop. Try asking:

- `What is the data-breach notification window in each vendor contract?`
- `Which contracts lack a liability cap, and what is the exposure?`
- `Summarize the confidentiality obligations that survive termination.`

Every answer is printed with its citations (document + clause) and appended to
`audit_log/klaus_audit.jsonl`. Type `exit` to quit.

> **No AMD ROCm/vLLM instance handy?** The default backend (`ROCmVLLMBackend`) is what the
> live demo actually runs on, but it needs a real vLLM server reachable at `ROCM_VLLM_BASE_URL`.
> To try Klaus locally without one, set `KLAUS_LLM_BACKEND=local_dev` in `.env` and point
> `OLLAMA_BASE_URL`/`OLLAMA_MODEL` at a local [Ollama](https://ollama.com) instance. This is the
> offline-iteration-only path (see `LLM Backends` below) — fine for trying out retrieval and
> citation grounding, but it's a small model and not what the recorded demo uses.

### Via Docker

```bash
# note the explicit --platform: required if building on Apple Silicon, since the hackathon
# submission target is linux/amd64
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
(`src/generate.py`) with three concrete implementations, selected via `KLAUS_LLM_BACKEND`:

| Backend | `KLAUS_LLM_BACKEND` | Role | Network |
|---|---|---|---|
| `ROCmVLLMBackend` | `rocm_vllm` (default) | **Primary — the backend the live demo runs on.** `Qwen/Qwen3-14B` served via vLLM on an AMD Developer Cloud Radeon/RDNA3 (gfx1100) instance — chosen for vLLM's native, first-class support (Gemma 4 was too new at build time and only had a bug-prone generic fallback implementation). This is what makes "nothing leaves the box" a demonstrated fact rather than a slide claim. | On-prem only |
| `FireworksGemmaBackend` | `fireworks` | Opt-in cloud triage tier for explicitly non-sensitive queries. Gated behind `KLAUS_ALLOW_CLOUD_TRIAGE=true` — refuses to construct otherwise. **Never the default, never shown as the primary demo path.** | Outbound to Fireworks AI |
| `LocalDevBackend` | `local_dev` | Offline iteration only, via a local Ollama instance. Used during this build to test retrieval/chunking/citation logic before AMD cloud access was wired up. **Never used in the live/recorded demo.** | Local only |

See `.env.example` for the full set of environment variables each backend reads (base URLs,
model names, API key). No key or secret is ever hardcoded — everything comes from the
environment.

## Hackathon-simplified vs. production-real

| Area | This demo | Production |
|---|---|---|
| Core pipeline network calls | Zero, by design (ingest → embed → retrieve → generate) | Same — this is a hard requirement, not a simplification |
| Primary LLM backend | `ROCmVLLMBackend` (`Qwen/Qwen3-14B`) against an AMD Developer Cloud Radeon/RDNA3 instance | Same backend/approach: `vLLM` on ROCm, scaling to AMD Instinct/CDNA GPUs for larger deployments |
| Vector store | FAISS flat index, in-memory, rebuilt on every run | Persistent, likely distributed vector store for larger corpora |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` — small, CPU-friendly, sufficient for the ~100-clause demo corpus | Possibly a larger or domain-tuned embedding model |
| PII/PHI scan | Regex (SSN/email/phone) + spaCy `en_core_web_sm` NER, detect-and-log only — the small NER model does produce some false positives (e.g. flagging "AES-256" as a person name), which a human reviewer would triage | Larger/more accurate NER, likely with redaction workflows, not just detection |
| Audit log | Local append-only JSONL file | Same append-only guarantee, but production would add replication/rotation/backup |
| Multi-tenancy | None — single corpus, single tenant | Per-customer/per-department index isolation (see `ARCHITECTURE.md` §8) |

## Project layout

```
src/ingest.py      - file loading (.txt, .pdf)
src/chunker.py     - clause/section splitting
src/pii_scan.py    - PII/PHI detection (regex + spaCy)
src/index.py       - hybrid (dense + BM25) index build
src/retrieve.py    - hybrid retrieval
src/generate.py    - LLM backends + structured citation output
src/audit_log.py   - append-only query/answer logging
src/app.py         - CLI orchestration
```

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the full system design, component interfaces, and
data flow.
