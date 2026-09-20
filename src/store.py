from __future__ import annotations

from typing import Any, Callable

from .chunking import compute_similarity
from .embeddings import _mock_embed
from .models import Document


class EmbeddingStore:
    """
    A vector store for text chunks, backed by a plain in-memory list.

    The scaffold offered an optional ChromaDB branch; it is deliberately gone.
    No test needs it, requirements.txt does not install it, and the skeleton set
    self._use_chroma = True *before* creating the client — so any machine that
    happened to have chromadb installed would route every method into an unbuilt
    code path and fail all 14 store tests.

    The embedding_fn parameter allows injection of mock embeddings for tests.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        embedding_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self._embedding_fn = embedding_fn or _mock_embed
        self._collection_name = collection_name
        self._store: list[dict[str, Any]] = []
        self._next_index = 0

    def _make_record(self, doc: Document) -> dict[str, Any]:
        """Normalize one Document into the dict shape kept in self._store."""
        # Copy, don't alias: the caller keeps its own dict, and a later mutation
        # on their side must not silently rewrite what we already stored.
        metadata = dict(doc.metadata or {})
        # delete_document() keys on doc_id. When one file is split into chunks the
        # ids look like "file#0", "file#1", so strip the chunk suffix to point the
        # fallback back at the source file rather than at a single chunk.
        metadata.setdefault("doc_id", doc.id.split("#")[0])

        record = {
            # The same doc.id can be added twice; the running index keeps ids unique.
            "id": f"{doc.id}::{self._next_index}",
            "content": doc.content,
            "metadata": metadata,
            "embedding": self._embedding_fn(doc.content),
        }
        self._next_index += 1
        return record

    def _search_records(self, query: str, records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        """Rank `records` against `query`. Both public searches route through here."""
        if not records or top_k <= 0:
            return []

        query_embedding = self._embedding_fn(query)
        # The embedding is left out of the result on purpose: a 64-1536 dim vector
        # per hit makes the output unreadable when printed in a terminal.
        scored = [
            {
                "id": record["id"],
                "content": record["content"],
                "metadata": record["metadata"],
                "score": compute_similarity(query_embedding, record["embedding"]),
            }
            for record in records
        ]
        scored.sort(key=lambda result: result["score"], reverse=True)
        return scored[:top_k]

    def add_documents(self, docs: list[Document]) -> None:
        """Embed each document's content and store it. One Document = one record."""
        self._store.extend(self._make_record(doc) for doc in docs)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Find the top_k most similar documents to query."""
        return self._search_records(query, self._store, top_k)

    def get_collection_size(self) -> int:
        """Return the total number of stored chunks."""
        return len(self._store)

    def search_with_filter(self, query: str, top_k: int = 3, metadata_filter: dict = None) -> list[dict]:
        """
        Search with optional metadata pre-filtering.

        Filter first, rank second. Ranking first would let non-matching documents
        occupy the k slots and then be dropped, returning fewer hits — possibly
        zero — even when the store still holds plenty of matching documents.
        """
        if not metadata_filter:
            candidates = self._store
        else:
            candidates = [
                record
                for record in self._store
                if all(record["metadata"].get(key) == value for key, value in metadata_filter.items())
            ]
        return self._search_records(query, candidates, top_k)

    def delete_document(self, doc_id: str) -> bool:
        """Remove all chunks of a document. True if anything was removed."""
        remaining = [record for record in self._store if record["metadata"].get("doc_id") != doc_id]
        if len(remaining) == len(self._store):
            return False
        self._store = remaining
        return True
