from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from src.generate import GeneratedAnswer
from src.retrieve import RetrievedClause

AUDIT_LOG_PATH = Path("audit_log/klaus_audit.jsonl")


def log_query(
    query: str,
    retrieved: list[RetrievedClause],
    answer: GeneratedAnswer,
    backend: str,
    log_path: Path = AUDIT_LOG_PATH,
) -> None:
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": query,
        "retrieved_clause_ids": [f"{r.clause.doc_id}::{r.clause.clause_id}" for r in retrieved],
        "answer": asdict(answer),
        "backend": backend,
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
