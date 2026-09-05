"""
knowledge_base.py

Loads the documents table (populated by ingest_kb.py) into memory and
builds three search methods over it, directly following the course:

- keyword_search: minsearch.Index, same as 01-agentic-rag/code/ingest.py
  and 07-project-example's fitness-assistant.
- vector_search: ONNX Runtime embeddings (embedder.py, ported from
  02-vector-search/lessons/09-onnx-embedder.md) + numpy dot product.
  The model's output is unit-length, so the dot product equals cosine
  similarity -- same math as sentence-transformers, just without the
  PyTorch/CUDA dependency weight (see embedder.py's docstring for why
  this swap was made, and the README's "Notes & limitations" for what
  was verified).
- hybrid_search: Reciprocal Rank Fusion over both ranked lists, ported
  from 06-best-practices/lessons/02-hybrid-search.md's `rrf()` function.

This module is deliberately framework-free and in-memory, matching the
course's teaching pattern (Postgres is the durable store; the search
index is rebuilt at process startup -- same split as
02-vector-search/lessons/07-sqlitesearch-vector.md, just without the
extra persistent-index layer since our corpus is small).

Requires the ONNX model to be downloaded first: run
`download_embedding_model.py` (or `make download-model`) once.
"""

from functools import lru_cache

from minsearch import Index

from embedder import Embedder

TEXT_FIELDS = ["title", "text"]
KEYWORD_FIELDS = ["ticker", "doc_type"]


@lru_cache(maxsize=1)
def _embedding_model():
    return Embedder()


class KnowledgeBase:
    def __init__(self, documents: list[dict]):
        self.documents = documents
        self._keyword_index = None
        self._doc_matrix = None  # numpy array of shape (n_docs, dim), or None until built

    # ------------------------------------------------------------------
    # Keyword search (minsearch)
    # ------------------------------------------------------------------

    def _ensure_keyword_index(self):
        if self._keyword_index is None:
            index = Index(text_fields=TEXT_FIELDS, keyword_fields=KEYWORD_FIELDS)
            index.fit(self.documents)
            self._keyword_index = index
        return self._keyword_index

    def keyword_search(self, query: str, ticker: str | None = None, num_results: int = 5) -> list[dict]:
        index = self._ensure_keyword_index()
        filter_dict = {"ticker": ticker} if ticker else {}
        boost_dict = {"title": 2.0, "text": 1.0}
        return index.search(query, filter_dict=filter_dict, boost_dict=boost_dict, num_results=num_results)

    # ------------------------------------------------------------------
    # Vector search (ONNX embedder + numpy)
    # ------------------------------------------------------------------

    def _ensure_doc_matrix(self):
        if self._doc_matrix is None:
            model = _embedding_model()
            texts = [f"{doc['title']}. {doc['text']}" for doc in self.documents]
            self._doc_matrix = model.encode_batch(texts)
        return self._doc_matrix

    def vector_search(self, query: str, ticker: str | None = None, num_results: int = 5) -> list[dict]:
        if not self.documents:
            return []

        matrix = self._ensure_doc_matrix()
        model = _embedding_model()
        v_query = model.encode(query)

        scores = matrix.dot(v_query)

        candidate_idx = range(len(self.documents))
        if ticker:
            candidate_idx = [i for i in candidate_idx if self.documents[i]["ticker"] == ticker]

        ranked = sorted(candidate_idx, key=lambda i: -scores[i])[:num_results]
        return [self.documents[i] for i in ranked]

    # ------------------------------------------------------------------
    # Hybrid search (Reciprocal Rank Fusion)
    # ------------------------------------------------------------------

    def hybrid_search(self, query: str, ticker: str | None = None, num_results: int = 5) -> list[dict]:
        keyword_results = self.keyword_search(query, ticker=ticker, num_results=num_results)
        vector_results = self.vector_search(query, ticker=ticker, num_results=num_results)
        return rrf([keyword_results, vector_results], num_results=num_results)


def rrf(search_results: list[list[dict]], k: int = 1, num_results: int = 10) -> list[dict]:
    """
    Reciprocal Rank Fusion, ported from
    06-best-practices/lessons/02-hybrid-search.md. Documents are keyed
    by `id` here (the course's FAQ documents don't have a stable id in
    that lesson, so it keys by `question`; ours always has `id`).
    """
    scores: dict[str, float] = {}
    doc_map: dict[str, dict] = {}

    for results in search_results:
        for rank, doc in enumerate(results):
            key = doc["id"]
            if key not in scores:
                scores[key] = 0
                doc_map[key] = doc
            scores[key] += 1 / (k + rank + 1)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_map[key] for key, _ in ranked[:num_results]]


def load_knowledge_base() -> KnowledgeBase:
    from db_documents import get_all_documents

    documents = get_all_documents()
    return KnowledgeBase(documents)
