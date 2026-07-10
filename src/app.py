from __future__ import annotations

import os
import sys

from dotenv import load_dotenv

from src.audit_log import log_query
from src.chunker import chunk_document
from src.generate import generate_answer, get_backend
from src.ingest import load_documents
from src.index import build_index
from src.pii_scan import scan_clauses
from src.retrieve import retrieve

DEFAULT_DATA_DIR = "data/sample_contracts"
TOP_K = 6


def main() -> None:
    load_dotenv()
    data_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATA_DIR

    print(f"Klaus — loading contracts from {data_dir}...")
    documents = load_documents(data_dir)
    clauses = [clause for document in documents for clause in chunk_document(document)]
    print(f"Loaded {len(documents)} document(s), {len(clauses)} clause(s).")

    warnings = scan_clauses(clauses)
    if warnings:
        print(f"PII/PHI scan: {len(warnings)} finding(s):")
        for warning in warnings:
            print(f"  [{warning.type}] {warning.doc_id} {warning.clause_id}: {warning.span!r}")
    else:
        print("PII/PHI scan: no findings.")

    index = build_index(clauses)

    backend_name = os.environ.get("KLAUS_LLM_BACKEND", "rocm_vllm")
    backend = get_backend()
    print(f"LLM backend: {backend_name}")

    print("\nKlaus is ready. Ask a question about the loaded contracts (or type 'exit' to quit).\n")
    while True:
        try:
            query = input("> ").strip()
        except EOFError:
            break
        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            break

        retrieved = retrieve(query, index, top_k=TOP_K)
        answer = generate_answer(query, retrieved, backend=backend)
        log_query(query, retrieved, answer, backend=backend_name)

        print(f"\n{answer.answer}\n")
        if answer.citations:
            print("Citations:")
            for citation in answer.citations:
                print(f"  - {citation.document} {citation.clause}: {citation.quote_or_paraphrase}")
        print()


if __name__ == "__main__":
    main()
