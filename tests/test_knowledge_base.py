"""
tests/test_knowledge_base.py

Unit tests for knowledge_base.py. rrf() is tested directly against the
worked example from 06-best-practices/lessons/02-hybrid-search.md.
keyword_search is tested against a tiny real minsearch index (no
network needed). vector_search is tested with a fake embedding model
(patched in) so these tests don't require downloading
all-MiniLM-L6-v2 -- that download was verified manually while building
this module; see the note at the top of knowledge_base.py.
"""

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))

import knowledge_base as kb_module  # noqa: E402
from knowledge_base import KnowledgeBase, rrf  # noqa: E402

DOCS = [
    {"id": "overview:AAPL", "ticker": "AAPL", "doc_type": "overview",
     "title": "Apple Inc. — Company Overview",
     "text": "Apple designs iPhones, Macs, and wearables. Exchange: XNAS."},
    {"id": "news:AAPL:1", "ticker": "AAPL", "doc_type": "news",
     "title": "Apple reports record iPhone sales",
     "text": "Apple posted strong quarterly iPhone revenue growth."},
    {"id": "overview:KO", "ticker": "KO", "doc_type": "overview",
     "title": "Coca-Cola — Company Overview",
     "text": "Coca-Cola makes beverages including soft drinks and juices."},
]


def test_rrf_matches_lesson_example():
    # From 06-best-practices/lessons/02-hybrid-search.md: text=[A,B,C,D,E], vector=[C,B,F,G,A]
    text = [{"id": "A"}, {"id": "B"}, {"id": "C"}, {"id": "D"}, {"id": "E"}]
    vector = [{"id": "C"}, {"id": "B"}, {"id": "F"}, {"id": "G"}, {"id": "A"}]

    ranked = rrf([text, vector], k=1, num_results=10)
    ranked_ids = [d["id"] for d in ranked]

    assert ranked_ids[0] == "C"  # C has the highest combined score per the lesson
    assert set(ranked_ids[1:3]) == {"A", "B"}  # tied for second


def test_keyword_search_finds_relevant_doc():
    kb = KnowledgeBase(DOCS)
    results = kb.keyword_search("iPhone sales", num_results=3)
    assert results
    assert results[0]["id"] == "news:AAPL:1"


def test_keyword_search_filters_by_ticker():
    kb = KnowledgeBase(DOCS)
    results = kb.keyword_search("company", ticker="KO", num_results=3)
    assert all(d["ticker"] == "KO" for d in results)


class FakeModel:
    """Deterministic fake embedder: each doc gets a distinct one-hot-ish vector.
    Matches embedder.Embedder's interface (encode / encode_batch), not
    sentence-transformers' -- see knowledge_base.py's docstring for why
    the project moved off sentence-transformers to the ONNX embedder.
    """

    VOCAB = ["iphone", "coca", "cola", "apple", "beverage"]

    def _embed_one(self, text):
        text = text.lower()
        vec = np.array([1.0 if word in text else 0.0 for word in self.VOCAB])
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def encode(self, text, normalize=True):
        return self._embed_one(text)

    def encode_batch(self, texts, normalize=True):
        return np.array([self._embed_one(t) for t in texts])


def test_vector_search_with_fake_model():
    kb = KnowledgeBase(DOCS)
    with patch.object(kb_module, "_embedding_model", return_value=FakeModel()):
        results = kb.vector_search("coca cola beverage", num_results=1)
    assert results[0]["id"] == "overview:KO"


def test_hybrid_search_combines_both_methods():
    kb = KnowledgeBase(DOCS)
    with patch.object(kb_module, "_embedding_model", return_value=FakeModel()):
        results = kb.hybrid_search("iphone apple", num_results=3)
    assert results
    assert results[0]["ticker"] == "AAPL"
