"""
Chat with a PDF — RAG from scratch, built stage by stage.

Stage 1: PDF loading & text extraction (with cleanup)
Stage 2: Chunking
Stage 3: Embeddings
Stage 4: Vector storage & retrieval
Stage 5: Prompt construction
Stage 6: Generation & answering
Stage 7 (optional/stretch): Basic evaluation
"""

import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import faiss
import httpx
import numpy as np
import ollama
import pdfplumber
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

PDF_PATH = Path("data/sample.pdf")

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3")

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

    # A line that appears on every page of a short document (1-2 pages)
    # is as likely to be real content as a running header. Below 3 pages,
    # keep every line instead of guessing.
    if len(pages) < 3:
        return ["\n".join(line for line in lines if line.strip()) for lines in page_lines]

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
    # FAISS returns -1 for a slot with no match, which happens when top_k
    # is larger than the number of stored vectors. A raw -1 index would
    # silently wrap around to the last chunk in the list below, so drop it.
    return [
        (chunks[i], float(sim))
        for sim, i in zip(similarities[0], indices[0])
        if i != -1
    ]


def build_prompt(query: str, results: list[tuple[Chunk, float]]) -> str:
    """
    Assemble retrieved chunks and the question into one prompt string for
    the LLM. Two instructions matter most here for controlling
    hallucination: telling the model to rely ONLY on the provided context,
    and explicitly permitting it to say "I don't know" instead of
    guessing. Stage 4 showed that retrieval always returns *something*,
    even for an unrelated question - so the model needs explicit
    permission to reject an unhelpful context rather than force an answer
    out of it.
    """
    context_blocks = []
    for i, (chunk, _score) in enumerate(results, start=1):
        pages = ", ".join(str(p) for p in chunk.pages)
        context_blocks.append(f"[{i}] (page {pages})\n{chunk.text}")
    context = "\n\n".join(context_blocks)

    return (
        "You are a helpful assistant that answers questions using ONLY the "
        "context provided below. If the context does not contain the "
        "answer, say \"I don't have enough information in the document to "
        "answer that\" instead of guessing.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )


def generate_answer(prompt: str, model: str = OLLAMA_MODEL, host: str = OLLAMA_HOST) -> str:
    """
    Send the constructed prompt to a local Ollama model and return its text
    response. This is the only place in the whole pipeline that actually
    produces new, human-readable text - every earlier stage only found and
    rearranged existing text from the document.
    """
    client = ollama.Client(host=host)
    try:
        response = client.generate(model=model, prompt=prompt)
    except httpx.ConnectError as exc:
        raise ConnectionError(
            f"Could not reach Ollama at {host}. Start it and make sure "
            f"'{model}' is pulled (run: ollama pull {model})."
        ) from exc
    return response["response"]


def format_sources(results: list[tuple[Chunk, float]]) -> str:
    """
    Render the chunks that were actually placed in the prompt as a
    human-readable sources list, using the same [n] numbering the model
    saw in the prompt. This lets the answer's grounding be checked against
    the real source text instead of taken on faith.
    """
    lines = []
    for i, (chunk, score) in enumerate(results, start=1):
        pages = ", ".join(str(p) for p in chunk.pages)
        preview = " ".join(chunk.text.split())
        if len(preview) > 120:
            preview = preview[:120] + "..."
        lines.append(f"[{i}] page(s) {pages} (score {score:.3f}): {preview}")
    return "\n".join(lines)


# Each (question, expected_page) pair is a fact we already know the page
# number for, by having read the document ourselves. This tests retrieval
# (Stage 4) in isolation, not the final generated answer.
EVAL_QUESTIONS: list[tuple[str, int]] = [
    ("Where should I place the ap_bookmark.bmk file?", 4),
    ("Where should I place the ap_bookmark.mdf file?", 4),
    ("What files are included in this sample package?", 3),
    ("What command identifies which bookmark file to use?", 3),
    ("How do I sort invoices by transaction amount?", 2),
    ("What software was used to create and test this sample?", 1),
    ("How many separate pages does the sample normally produce by default?", 1),
]


def evaluate_retrieval(
    questions: list[tuple[str, int]],
    chunks: list[Chunk],
    index: faiss.Index,
    model: SentenceTransformer,
    top_k: int = 3,
) -> None:
    """
    For each (question, expected_page) pair, check whether the expected
    page shows up anywhere among the top_k retrieved chunks' pages. A
    "pass" doesn't mean the exact right sentence was retrieved - only that
    the right page was in the mix. That's a deliberately loose bar, and a
    real limitation of this evaluation - see the tradeoff writeup.
    """
    passed = 0
    for question, expected_page in questions:
        results = retrieve(question, chunks, index, model, top_k=top_k)
        retrieved_pages = {p for chunk, _ in results for p in chunk.pages}
        ok = expected_page in retrieved_pages
        passed += ok
        status = "PASS" if ok else "FAIL"
        print(
            f"[{status}] expected page {expected_page}, "
            f"got pages {sorted(retrieved_pages)} - {question!r}"
        )
    print(f"\n{passed}/{len(questions)} passed")


if __name__ == "__main__":
    if not PDF_PATH.exists():
        raise FileNotFoundError(
            f"No PDF at {PDF_PATH}. Put a PDF there, or change PDF_PATH "
            f"in chat_with_pdf.py."
        )

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

    prompt = build_prompt(query, results)
    print("Stage 5: constructed prompt (this is the literal text the LLM will see)\n")
    print(prompt)
    print()

    print(f"Stage 6: asking {OLLAMA_MODEL} via Ollama...\n")
    answer = generate_answer(prompt)
    print("Answer:")
    print(answer)
    print()
    print("Sources used:")
    print(format_sources(results))
    print()

    print(f"Stage 7: evaluating retrieval on {len(EVAL_QUESTIONS)} test questions\n")
    evaluate_retrieval(EVAL_QUESTIONS, chunks, index, embedding_model)
