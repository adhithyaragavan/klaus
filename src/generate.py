from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Protocol

import requests

from src.retrieve import RetrievedClause

INSUFFICIENT_GROUNDED_INFORMATION = "insufficient grounded information"


class LLMBackend(Protocol):
    def generate(self, prompt: str) -> str: ...


class ROCmVLLMBackend:
    """Primary backend: Gemma served via vLLM on an AMD Instinct instance (OpenAI-compatible HTTP API)."""

    def __init__(self, base_url: str | None = None, model: str | None = None) -> None:
        self.base_url = base_url or os.environ["ROCM_VLLM_BASE_URL"]
        self.model = model or os.environ["ROCM_VLLM_MODEL"]

    def generate(self, prompt: str) -> str:
        response = requests.post(
            f"{self.base_url}/chat/completions",
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            timeout=180,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]


class LocalDevBackend:
    """Offline iteration only, via Ollama's local HTTP API — never used in the live demo."""

    def __init__(self, base_url: str | None = None, model: str | None = None) -> None:
        self.base_url = base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model = model or os.environ.get("OLLAMA_MODEL", "gemma2:2b")

    def generate(self, prompt: str) -> str:
        response = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {"temperature": 0.0},
            },
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["message"]["content"]


def get_backend() -> LLMBackend:
    choice = os.environ.get("KLAUS_LLM_BACKEND", "rocm_vllm")
    if choice == "rocm_vllm":
        return ROCmVLLMBackend()
    if choice == "local_dev":
        return LocalDevBackend()
    raise ValueError(f"Unknown KLAUS_LLM_BACKEND: {choice!r}")


@dataclass
class Citation:
    document: str
    clause: str
    quote_or_paraphrase: str


@dataclass
class GeneratedAnswer:
    answer: str
    citations: list[Citation]


def generate_answer(
    query: str,
    retrieved: list[RetrievedClause],
    backend: LLMBackend | None = None,
) -> GeneratedAnswer:
    if not retrieved:
        return GeneratedAnswer(answer=INSUFFICIENT_GROUNDED_INFORMATION, citations=[])

    backend = backend or get_backend()
    prompt = _build_prompt(query, retrieved)

    raw = backend.generate(prompt)
    result = _parse_and_validate(raw, retrieved)
    if result is not None:
        return result

    raw_retry = backend.generate(_build_retry_prompt(prompt))
    result = _parse_and_validate(raw_retry, retrieved)
    if result is not None:
        return result

    return GeneratedAnswer(answer=INSUFFICIENT_GROUNDED_INFORMATION, citations=[])


def _build_prompt(query: str, retrieved: list[RetrievedClause]) -> str:
    clause_block = "\n\n".join(
        f'[{i}] document="{r.clause.doc_id}" clause="{r.clause.clause_id}"\n{r.clause.text}'
        for i, r in enumerate(retrieved, start=1)
    )
    return f"""You are a contract compliance assistant. Answer the question using ONLY the numbered \
clauses below. Do not use any outside knowledge, and do not assume facts about contracts that are \
not shown below.

Every factual claim in your answer must be backed by at least one citation to a clause below, using \
that clause's exact "document" and "clause" values — never invent a document name or clause id that \
is not listed below.

If the clauses below do not contain enough information to answer the question, set "answer" to \
"{INSUFFICIENT_GROUNDED_INFORMATION}" and "citations" to an empty list.

Respond with ONLY a single JSON object matching this exact schema, and no other text:
{{"answer": "<answer text>", "citations": [{{"document": "<doc_id>", "clause": "<clause_id>", "quote_or_paraphrase": "<short quote or paraphrase from that clause>"}}]}}

Clauses:
{clause_block}

Question: {query}
"""


def _build_retry_prompt(original_prompt: str) -> str:
    return (
        f"{original_prompt}\n\n"
        "Your previous response was not a valid JSON object with at least one citation whose "
        '"document" and "clause" values exactly match one of the clauses listed above. '
        "Respond again with ONLY the JSON object described above — no markdown, no extra text — "
        "and ensure every citation matches a listed clause exactly."
    )


_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$")


def _parse_and_validate(raw: str, retrieved: list[RetrievedClause]) -> GeneratedAnswer | None:
    json_text = _extract_json_object(raw)
    if json_text is None:
        return None

    try:
        data = json.loads(json_text)
    except json.JSONDecodeError:
        return None

    if not isinstance(data, dict) or "answer" not in data or "citations" not in data:
        return None

    citations_raw = data["citations"]
    if not isinstance(citations_raw, list):
        return None

    # Keyed on a normalized label so e.g. a model-emitted "2.3" still matches our canonical
    # "Section 2.3" — the output always uses our canonical clause_id, never the model's string.
    valid_clauses = {(r.clause.doc_id, _normalize_clause_label(r.clause.clause_id)): r.clause.clause_id for r in retrieved}
    citations: list[Citation] = []
    for entry in citations_raw:
        if not isinstance(entry, dict):
            continue
        document, clause = entry.get("document"), entry.get("clause")
        if not isinstance(document, str) or not isinstance(clause, str):
            continue
        canonical_clause = valid_clauses.get((document, _normalize_clause_label(clause)))
        if canonical_clause is None:
            continue  # drop citations that don't match a real retrieved clause — never trust a fabricated one
        citations.append(
            Citation(
                document=document,
                clause=canonical_clause,
                quote_or_paraphrase=str(entry.get("quote_or_paraphrase", "")),
            )
        )

    if not citations:
        return None

    return GeneratedAnswer(answer=str(data["answer"]), citations=citations)


def _extract_json_object(raw: str) -> str | None:
    text = _THINK_BLOCK_RE.sub("", raw).strip()
    text = _CODE_FENCE_RE.sub("", text).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    return text[start : end + 1]


_CLAUSE_PREFIX_RE = re.compile(r"^(Section|Article)\s+", re.IGNORECASE)


def _normalize_clause_label(label: str) -> str:
    return _CLAUSE_PREFIX_RE.sub("", label.strip()).strip().lower()
