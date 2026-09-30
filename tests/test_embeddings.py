"""Real embedding models; opt-in because they download ~90 MB of weights."""

import os

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(not os.environ.get("SCI_TEST_REAL_MODELS"),
                                reason="set SCI_TEST_REAL_MODELS=1 to download and test real models")

TEXTS = ["bar chart comparing accuracy", '{"image_type": "diagram", "title": "Transformer"}']


def test_fastembed_matches_sentence_transformers():
    """The SaaS (fastembed) and CLI (sentence-transformers) can share a Postgres table."""
    pytest.importorskip("fastembed")
    pytest.importorskip("sentence_transformers")
    from sci_vizio_retrieval.embeddings import FastEmbedEmbedder, SentenceTransformerEmbedder

    fast, st = FastEmbedEmbedder(), SentenceTransformerEmbedder()
    assert fast.dimension == st.dimension == 384
    for a, b in zip(fast.embed(TEXTS), st.embed(TEXTS), strict=True):
        assert np.linalg.norm(a) == pytest.approx(1.0, abs=1e-5)
        assert float(np.dot(a, b)) == pytest.approx(1.0, abs=1e-4)
