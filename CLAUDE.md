# CLAUDE.md — Klaus

## Project Overview

Klaus is an on-prem, citation-grounded document compliance assistant, built for AMD Developer Hackathon: ACT II, Track 3 (Unicorn). Full system design lives in `ARCHITECTURE.md` — read that before making structural changes, and update it if a real design decision changes during the build. Build sequencing lives in `klaus_claude_code_prompt.md`.

Demo vertical: vendor contract compliance review (confidentiality, termination, liability-cap, data-handling clauses over a synthetic contract set in `data/sample_contracts/`).

## Immutable Rules

1. **No un-cited answers.** Every generated answer must include at least one citation with a real `document` + `clause_id` from the index. If retrieval can't ground an answer, return "insufficient grounded information" — never let the model answer from general knowledge.
2. **Nothing leaves the box in the core path.** Ingest → embed → retrieve → generate must run with zero outbound network calls. Any cloud LLM path (e.g. Fireworks) is opt-in only, gated behind `KLAUS_ALLOW_CLOUD_TRIAGE=true`, and never the default.
3. **No hardcoded secrets.** All keys/config via environment variables, documented in `.env.example`. Never commit a real key.
4. **Audit log is append-only.** Every query gets a JSONL entry in `audit_log/klaus_audit.jsonl` — including failed/rejected generations. Never overwrite prior entries.
5. **Citations reference real clause IDs.** Never let the generator invent a section number that isn't in the retrieved chunk metadata.
6. **The live demo runs on-prem.** The recorded/live demo must exercise `ROCmVLLMBackend` as the primary grounded path (this is what makes Rule #2 a demonstrated fact). `LocalDevBackend` must never be the backend shown live, and `FireworksGemmaBackend` may appear only as the explicitly opt-in cloud tier — never as the default demo path.

## Build Commands

- Install deps: `pip install -r requirements.txt`
- Run the demo query loop: `bash scripts/run_demo.sh`
- Run tests: `pytest tests/`
- Build the container: `docker buildx build --platform linux/amd64 -t klaus:latest -f docker/Dockerfile .`
- Run the container: `docker run --rm -it klaus:latest`

## Project Layout

```
src/ingest.py      - file loading
src/chunker.py     - clause/section splitting
src/pii_scan.py    - PII/PHI detection (regex + spaCy)
src/index.py       - hybrid (dense + BM25) index build
src/retrieve.py    - hybrid retrieval
src/generate.py    - LLM call + structured citation output
src/audit_log.py   - append-only query/answer logging
src/app.py         - orchestration (CLI / optional FastAPI)
```

See `ARCHITECTURE.md` §3 for each component's exact interface — implement to that spec rather than improvising a different shape, since later components depend on it.

## Conventions

- Python, type hints on all public functions.
- Keep the `LLMBackend` protocol swap-friendly across the three concrete backends — don't hardcode a specific one into business logic:
  - `LocalDevBackend` (`llama-cpp-python`/Ollama, any small GGUF) — offline iteration only, never demoed.
  - `ROCmVLLMBackend` (Qwen3-14B via vLLM on an AMD Developer Cloud Radeon/RDNA3 instance) — the on-prem production backend and the primary demo backend.
  - `FireworksGemmaBackend` (Gemma via Fireworks AI) — opt-in cloud triage tier only, gated behind `KLAUS_ALLOW_CLOUD_TRIAGE=true`, never the default.
- Chunk by clause/section numbering first; fall back to paragraph split only when no numbering is detected, and mark those chunks accordingly (see `ARCHITECTURE.md` §3.2).
- Don't add new top-level dependencies without a good reason — this needs to build inside a 10GB container.

## Terminology

- "Clause" = the retrieval/citation unit — not "chunk" (chunk is fine in code comments, but keep "clause" in anything user-facing or in citation output).
- The product name is **Klaus** everywhere — code, README, container tags, comments. Don't reintroduce the earlier working name ("Warden").

## What NOT to put here

Don't add hackathon deadline countdowns, sprint-specific TODOs, or a running task checklist to this file — those belong in the conversation or an issue tracker, not in persistent project memory. Keep this file to things that stay true for the life of the repo.
