"""
In-memory LRU cache of loaded (chunks, FAISS index) pairs, keyed by
(user_id, doc_id).

Every document is durably persisted to disk (storage.py) - this cache only
avoids re-reading and re-parsing that disk state on every chat message.
Bounded size keeps memory use predictable when many users have documents
open; the alternative (load-once-keep-forever) would leak memory as more
documents get opened over the app's lifetime.
"""

from collections import OrderedDict

import faiss

import storage
from chat_with_pdf import Chunk

MAX_CACHED_DOCUMENTS = 8

_cache: "OrderedDict[tuple[str, str], tuple[list[Chunk], faiss.Index]]" = OrderedDict()


def get(user_id: str, doc_id: str) -> tuple[list[Chunk], faiss.Index]:
    key = (user_id, doc_id)
    if key in _cache:
        _cache.move_to_end(key)
        return _cache[key]

    chunks, index = storage.load_document(user_id, doc_id)
    _cache[key] = (chunks, index)
    _cache.move_to_end(key)
    if len(_cache) > MAX_CACHED_DOCUMENTS:
        _cache.popitem(last=False)
    return chunks, index


def evict(user_id: str, doc_id: str) -> None:
    _cache.pop((user_id, doc_id), None)
