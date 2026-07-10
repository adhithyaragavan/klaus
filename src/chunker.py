from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from src.ingest import RawDocument

SECTION_HEADER_RE = re.compile(r"^Section\s+(\d+)\b")
ARTICLE_HEADER_RE = re.compile(r"^Article\s+(\w+)\b")
SUBSECTION_RE = re.compile(r"^(\d+(?:\.\d+)+)\s")


@dataclass
class Clause:
    doc_id: str
    clause_id: str
    text: str
    page: int | None
    granularity: Literal["clause", "paragraph"]


def chunk_document(document: RawDocument) -> list[Clause]:
    lines, line_pages = _flatten_pages(document.pages)
    boundaries = _find_clause_boundaries(lines)

    if not boundaries:
        return _paragraph_fallback(document.doc_id, lines, line_pages)

    clauses: list[Clause] = []
    boundary_starts = [start for start, _ in boundaries] + [len(lines)]
    for (start, clause_id), end in zip(boundaries, boundary_starts[1:]):
        # A span can run past the last subsection of its own section into the header line of the
        # *next* section (e.g. "Section 2. Confidentiality") before hitting that section's first
        # subsection boundary. Drop such stray headers; keep one only as a clause's own opening line.
        span_lines = [
            line
            for idx, line in enumerate(lines[start:end], start=start)
            if idx == start or not (SECTION_HEADER_RE.match(line) or ARTICLE_HEADER_RE.match(line))
        ]
        text = "\n".join(span_lines).strip()
        if not text:
            continue
        clauses.append(
            Clause(
                doc_id=document.doc_id,
                clause_id=clause_id,
                text=text,
                page=line_pages[start],
                granularity="clause",
            )
        )
    return clauses


def _flatten_pages(pages: list[str]) -> tuple[list[str], list[int | None]]:
    lines: list[str] = []
    line_pages: list[int | None] = []
    single_page = len(pages) == 1
    for page_num, page_text in enumerate(pages, start=1):
        page_lines = page_text.split("\n")
        lines.extend(page_lines)
        line_pages.extend([None if single_page else page_num] * len(page_lines))
    return lines, line_pages


def _find_clause_boundaries(lines: list[str]) -> list[tuple[int, str]]:
    section_headers: list[tuple[int, str]] = []  # (line_idx, "Section N" or "Article W")
    subsections: list[tuple[int, str]] = []  # (line_idx, "Section N.M")

    for idx, line in enumerate(lines):
        if match := SUBSECTION_RE.match(line):
            subsections.append((idx, f"Section {match.group(1)}"))
        elif match := SECTION_HEADER_RE.match(line):
            section_headers.append((idx, f"Section {match.group(1)}"))
        elif match := ARTICLE_HEADER_RE.match(line):
            section_headers.append((idx, f"Article {match.group(1)}"))

    # A top-level Section/Article header is only a clause boundary in its own right when the
    # section it introduces has no finer-grained subsection numbering — otherwise the
    # subsections are the real citation unit and the header is just a title.
    header_starts = [idx for idx, _ in section_headers] + [len(lines)]
    standalone_headers = [
        (idx, label)
        for (idx, label), next_idx in zip(section_headers, header_starts[1:])
        if not any(idx < sub_idx < next_idx for sub_idx, _ in subsections)
    ]

    return sorted(subsections + standalone_headers, key=lambda pair: pair[0])


def _paragraph_fallback(doc_id: str, lines: list[str], line_pages: list[int | None]) -> list[Clause]:
    clauses: list[Clause] = []
    para_lines: list[str] = []
    para_start: int | None = None
    counter = 0

    def flush() -> None:
        nonlocal counter
        text = "\n".join(para_lines).strip()
        if text and para_start is not None:
            counter += 1
            clauses.append(
                Clause(
                    doc_id=doc_id,
                    clause_id=f"Paragraph {counter}",
                    text=text,
                    page=line_pages[para_start],
                    granularity="paragraph",
                )
            )

    for idx, line in enumerate(lines):
        if line.strip() == "":
            flush()
            para_lines = []
            para_start = None
        else:
            if para_start is None:
                para_start = idx
            para_lines.append(line)
    flush()
    return clauses
