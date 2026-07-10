from __future__ import annotations

import re
from dataclasses import dataclass

from src.chunker import Clause

_REGEX_PATTERNS = [
    ("ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("email", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    # No leading \b: it doesn't reliably match right before a literal "(", which would drop it from the match.
    ("phone", re.compile(r"(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b")),
]

_nlp = None  # lazy-loaded spaCy pipeline; loading is expensive


def _get_nlp():
    global _nlp
    if _nlp is None:
        import spacy

        _nlp = spacy.load("en_core_web_sm")
    return _nlp


@dataclass
class PIIWarning:
    doc_id: str
    clause_id: str
    span: str
    type: str


def scan_clauses(clauses: list[Clause]) -> list[PIIWarning]:
    warnings: list[PIIWarning] = []
    nlp = _get_nlp()

    for clause in clauses:
        found_spans: set[str] = set()

        for pii_type, pattern in _REGEX_PATTERNS:
            for match in pattern.finditer(clause.text):
                span = " ".join(match.group(0).split())  # normalize whitespace from source line-wraps
                if span in found_spans:
                    continue
                found_spans.add(span)
                warnings.append(PIIWarning(doc_id=clause.doc_id, clause_id=clause.clause_id, span=span, type=pii_type))

        for entity in nlp(clause.text).ents:
            if entity.label_ != "PERSON":
                continue
            span = " ".join(entity.text.split())
            if span in found_spans:
                continue
            found_spans.add(span)
            warnings.append(PIIWarning(doc_id=clause.doc_id, clause_id=clause.clause_id, span=span, type="person"))

    return warnings
