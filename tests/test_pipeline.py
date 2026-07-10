import json

from src.audit_log import log_query
from src.chunker import Clause
from src.generate import INSUFFICIENT_GROUNDED_INFORMATION, Citation, GeneratedAnswer, generate_answer
from src.retrieve import RetrievedClause


class _FakeBackend:
    def __init__(self, responses: list[str]) -> None:
        self._responses = iter(responses)

    def generate(self, prompt: str) -> str:
        return next(self._responses)


def _sample_retrieved() -> list[RetrievedClause]:
    clause = Clause(
        doc_id="vendor_acme_nda.txt",
        clause_id="Section 4.1",
        text="Each Party's aggregate liability shall not exceed five hundred thousand dollars ($500,000).",
        page=None,
        granularity="clause",
    )
    return [RetrievedClause(clause=clause, score=0.9)]


def test_every_answer_has_a_citation_or_is_marked_insufficient():
    retrieved = _sample_retrieved()

    grounded_response = (
        '{"answer": "The liability cap is $500,000.", '
        '"citations": [{"document": "vendor_acme_nda.txt", "clause": "Section 4.1", '
        '"quote_or_paraphrase": "shall not exceed five hundred thousand dollars"}]}'
    )
    result = generate_answer("What is the liability cap?", retrieved, backend=_FakeBackend([grounded_response]))
    assert result.citations
    assert result.answer != INSUFFICIENT_GROUNDED_INFORMATION

    # Model cites a clause that was never retrieved -> must be rejected on both attempts and fall
    # back to the safe response, never trusted as a real citation.
    fabricated_response = (
        '{"answer": "The cap is $9,999,999.", '
        '"citations": [{"document": "vendor_acme_nda.txt", "clause": "Section 99.9", '
        '"quote_or_paraphrase": "made up"}]}'
    )
    result = generate_answer(
        "What is the liability cap?",
        retrieved,
        backend=_FakeBackend([fabricated_response, fabricated_response]),
    )
    assert result.answer == INSUFFICIENT_GROUNDED_INFORMATION
    assert result.citations == []


def test_audit_log_is_written_for_every_query(tmp_path):
    log_path = tmp_path / "klaus_audit.jsonl"
    retrieved = _sample_retrieved()

    grounded = GeneratedAnswer(
        answer="The liability cap is $500,000.",
        citations=[Citation(document="vendor_acme_nda.txt", clause="Section 4.1", quote_or_paraphrase="...")],
    )
    rejected = GeneratedAnswer(answer=INSUFFICIENT_GROUNDED_INFORMATION, citations=[])

    log_query("query one", retrieved, grounded, backend="local_dev", log_path=log_path)
    log_query("query two", [], rejected, backend="local_dev", log_path=log_path)  # failed generations are logged too

    lines = log_path.read_text().strip().split("\n")
    assert len(lines) == 2  # append-only: both queries present, neither overwrote the other

    first, second = (json.loads(line) for line in lines)
    assert first["query"] == "query one"
    assert first["answer"]["citations"]
    assert second["query"] == "query two"
    assert second["answer"]["answer"] == INSUFFICIENT_GROUNDED_INFORMATION
