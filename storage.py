"""
On-disk persistence for a document's chunks and FAISS index.

FAISS indices don't carry arbitrary payload data (like chunk text or page
numbers), so each document gets a small directory with two files: the raw
index (index.faiss) and the chunk metadata FAISS doesn't store (chunks.json).
Keeping these out of the SQL database avoids storing large binary blobs in
SQLite, and keeps loading a document as simple as reading two files.
"""

import json
from pathlib import Path

import faiss

from chat_with_pdf import Chunk

STORAGE_ROOT = Path("storage")


def document_dir(user_id: str, doc_id: str) -> Path:
    return STORAGE_ROOT / user_id / doc_id


def save_document(user_id: str, doc_id: str, chunks: list[Chunk], index: faiss.Index) -> None:
    directory = document_dir(user_id, doc_id)
    directory.mkdir(parents=True, exist_ok=True)

    chunks_payload = [
        {"text": c.text, "start": c.start, "end": c.end, "pages": c.pages} for c in chunks
    ]
    (directory / "chunks.json").write_text(json.dumps(chunks_payload))
    faiss.write_index(index, str(directory / "index.faiss"))


def load_document(user_id: str, doc_id: str) -> tuple[list[Chunk], faiss.Index]:
    directory = document_dir(user_id, doc_id)
    chunks_payload = json.loads((directory / "chunks.json").read_text())
    chunks = [Chunk(**c) for c in chunks_payload]
    index = faiss.read_index(str(directory / "index.faiss"))
    return chunks, index


def delete_document(user_id: str, doc_id: str) -> None:
    directory = document_dir(user_id, doc_id)
    for f in directory.glob("*"):
        f.unlink()
    if directory.exists():
        directory.rmdir()
