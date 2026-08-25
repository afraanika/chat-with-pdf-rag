"""
Chat with a PDF — RAG from scratch, built stage by stage.

Stage 1: PDF loading & text extraction (with cleanup)
Stage 2: Chunking
Stage 3: Embeddings
Stage 4: Vector storage & retrieval
"""

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
import pdfplumber
from sentence_transformers import SentenceTransformer

PDF_PATH = Path("data/sample.pdf")

# Chunk size and overlap are measured in characters, not tokens - a rough
# proxy (about 4 characters per token in English), not an exact budget.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# Small (~80MB), fast on CPU, 384-dimensional vectors. Larger models like
# all-mpnet-base-v2 (768-dim) capture meaning more precisely but cost more
# to run - MiniLM trades some quality for fast local iteration.
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


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


def load_embedding_model(model_name: str = EMBEDDING_MODEL) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def embed_texts(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    """
    Turn a list of texts into fixed-length vectors. Vectors are
    L2-normalized so that a plain dot product between two vectors equals
    their cosine similarity - this keeps the similarity math simple
    everywhere it's used, including the index search in Stage 4.
    """
    return model.encode(texts, normalize_embeddings=True)


def build_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Build a FAISS flat (exact) index over the given embeddings, using inner
    product as the similarity metric. Because our embeddings are already
    L2-normalized, inner product IS cosine similarity here.

    "Flat" means every search still checks every stored vector - no
    approximation. At 8 chunks that is the right call: an approximate index
    (IVF, HNSW, ...) exists to skip most of a much larger haystack, and
    would be pure overhead here. It only pays off once a linear scan is
    actually slow - realistically hundreds of thousands of vectors or more.
    """
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    return index


def retrieve(
    query: str,
    chunks: list[Chunk],
    index: faiss.Index,
    model: SentenceTransformer,
    top_k: int = 3,
) -> list[tuple[Chunk, float]]:
    """
    Embed the query with the SAME model used for the chunks - comparing
    vectors from two different models would run without error but be
    meaningless, since they wouldn't share a coordinate system - then ask
    the index for the top_k closest chunk vectors.
    """
    query_vector = embed_texts(model, [query])
    similarities, indices = index.search(query_vector, top_k)
    return [(chunks[i], float(sim)) for sim, i in zip(similarities[0], indices[0])]


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

    embedding_model = load_embedding_model()
    embeddings = embed_texts(embedding_model, [chunk.text for chunk in chunks])
    print(f"Stage 3: embedded {embeddings.shape[0]} chunks into {embeddings.shape[1]}-dim vectors\n")
    print(f"First 8 numbers of chunk 1's vector: {embeddings[0][:8]}\n")

    # A concrete look at "distance = similarity": vectors are normalized,
    # so a dot product between two of them IS their cosine similarity.
    print("Similarity demo (cosine similarity, 1.0 = identical direction):")
    pairs_to_compare = [(0, 1), (0, 4), (0, 7)]
    for i, j in pairs_to_compare:
        similarity = float(np.dot(embeddings[i], embeddings[j]))
        print(f"  chunk {i + 1} vs chunk {j + 1}: {similarity:.3f}")
    print()

    index = build_index(embeddings)
    print(f"Stage 4: built a FAISS flat index over {index.ntotal} vectors\n")

    query = "Where should I place the ap_bookmark.bmk file?"
    print(f"Query: {query!r}\n")
    results = retrieve(query, chunks, index, embedding_model, top_k=3)
    for rank, (chunk, score) in enumerate(results, start=1):
        print(f"--- Rank {rank} (score {score:.3f}, page(s) {chunk.pages}) ---")
        print(chunk.text)
        print()
