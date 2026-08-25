"""
Chat with a PDF — RAG from scratch, built stage by stage.

Stage 1: PDF loading & text extraction (with cleanup)
Stage 2: Chunking
"""

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pdfplumber

PDF_PATH = Path("data/sample.pdf")

# Chunk size and overlap are measured in characters, not tokens - a rough
# proxy (about 4 characters per token in English), not an exact budget.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


def extract_page_content(pdf_path: Path) -> list[str]:
    """
    Extract text per page, with tables pulled out and formatted separately
    from the prose. A plain extract_text() call flattens tables into a
    single unstructured line — pulling out each table's bounding box first
    keeps its row/column structure intact.
    """
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            tables = page.find_tables()

            # Exclude each table's bounding box before extracting prose text,
            # so the flattened table text isn't duplicated alongside the
            # clean structured version we build below.
            prose_page = page
            for table in tables:
                prose_page = prose_page.outside_bbox(table.bbox)
            prose = prose_page.extract_text() or ""

            table_blocks = []
            for table in tables:
                rows = table.extract()
                formatted_rows = [
                    " | ".join(cell or "" for cell in row) for row in rows
                ]
                table_blocks.append("\n".join(formatted_rows))

            combined = prose
            if table_blocks:
                combined += "\n\n" + "\n\n".join(table_blocks)
            pages.append(combined)
    return pages


def remove_repeated_lines(pages: list[str]) -> list[str]:
    """
    Strip lines that repeat across most pages (running headers/footers like
    "Page 1 of 4"), and drop blank lines. Detection normalizes digits so a
    header with a changing page number still counts as "the same line".
    """
    page_lines = [p.split("\n") for p in pages]

    def normalize(line: str) -> str:
        return re.sub(r"\d+", "#", line.strip())

    normalized_per_page = [
        {normalize(line) for line in lines if line.strip()} for lines in page_lines
    ]
    line_counts = Counter()
    for normalized_set in normalized_per_page:
        line_counts.update(normalized_set)

    threshold = len(pages) / 2
    repeated = {line for line, count in line_counts.items() if count > threshold}

    cleaned_pages = []
    for lines in page_lines:
        kept = [
            line for line in lines if line.strip() and normalize(line) not in repeated
        ]
        cleaned_pages.append("\n".join(kept))
    return cleaned_pages


def build_document(cleaned_pages: list[str]) -> tuple[str, list[tuple[int, int, int]]]:
    """
    Join cleaned per-page text into one continuous document string, and
    record which character range came from which page (1-indexed). This
    lets a chunk that spans a page break still report which page(s) it
    came from, for citations later.
    """
    document = ""
    page_ranges: list[tuple[int, int, int]] = []
    for i, page_text in enumerate(cleaned_pages, start=1):
        start = len(document)
        document += page_text + "\n\n"
        end = len(document)
        page_ranges.append((start, end, i))
    return document, page_ranges


def pages_for_range(
    start: int, end: int, page_ranges: list[tuple[int, int, int]]
) -> list[int]:
    """Return the page numbers whose character range overlaps [start, end)."""
    return [
        page_num
        for range_start, range_end, page_num in page_ranges
        if range_start < end and range_end > start
    ]


@dataclass
class Chunk:
    text: str
    start: int
    end: int
    pages: list[int]


def chunk_text(
    document: str,
    page_ranges: list[tuple[int, int, int]],
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[Chunk]:
    """
    Split `document` into overlapping fixed-size chunks. Each chunk shares
    `overlap` characters with the next one, so a fact sitting right on a
    chunk boundary still appears whole in at least one chunk.
    """
    assert chunk_size > overlap, "chunk_size must be greater than overlap"

    chunks = []
    step = chunk_size - overlap
    start = 0
    while start < len(document):
        end = min(start + chunk_size, len(document))
        text = document[start:end]
        chunks.append(
            Chunk(
                text=text,
                start=start,
                end=end,
                pages=pages_for_range(start, end, page_ranges),
            )
        )
        if end == len(document):
            break
        start += step
    return chunks


if __name__ == "__main__":
    raw_pages = extract_page_content(PDF_PATH)
    cleaned_pages = remove_repeated_lines(raw_pages)
    total_chars = sum(len(p) for p in cleaned_pages)
    print(f"Stage 1: extracted and cleaned {len(cleaned_pages)} pages ({total_chars} chars total)\n")

    document, page_ranges = build_document(cleaned_pages)
    chunks = chunk_text(document, page_ranges)
    print(
        f"Stage 2: split into {len(chunks)} chunks "
        f"(chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})\n"
    )

    for i, chunk in enumerate(chunks):
        print(
            f"--- Chunk {i + 1} (chars {chunk.start}-{chunk.end}, "
            f"{len(chunk.text)} chars, page(s) {chunk.pages}) ---"
        )
        print(chunk.text)
        print()
