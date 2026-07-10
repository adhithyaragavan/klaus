from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

SUPPORTED_SUFFIXES = {".txt", ".pdf"}


@dataclass
class RawDocument:
    doc_id: str
    pages: list[str]  # one entry per page; .txt sources yield a single page (page numbers are then unknown downstream)


def load_documents(source_dir: str | Path) -> list[RawDocument]:
    source_dir = Path(source_dir)
    documents: list[RawDocument] = []
    for path in sorted(source_dir.iterdir()):
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        if path.suffix.lower() == ".txt":
            pages = [path.read_text(encoding="utf-8")]
        else:
            from pypdf import PdfReader  # local import: keeps the .txt-only path free of the pypdf dependency

            reader = PdfReader(str(path))
            pages = [page.extract_text() or "" for page in reader.pages]
        documents.append(RawDocument(doc_id=path.name, pages=pages))
    return documents
