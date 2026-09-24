from __future__ import annotations

import numpy as np
import pytest

from smallville_lite.embed import CachedEmbedder, EmbeddingCache, build_embedder
from smallville_lite.embed.base import EmbedBatch
from smallville_lite.embed.stub import StubEmbedder
from smallville_lite.embed.voyage import VoyageEmbedder
from smallville_lite.llm.budget import BudgetGuard
from smallville_lite.llm.types import BudgetExhausted
from smallville_lite.llm.usage import UsageSummary
from smallville_lite.sim.events import ListSink


def cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b)


def test_stub_vectors_are_deterministic_normalized_and_meaningful():
    e = StubEmbedder()
    v = e.embed(["Isabella is planning a Valentine's Day party at Hobbs Cafe",
                 "a party at Hobbs Cafe on Valentine's Day",
                 "Sam waters the garden beds in Johnson Park"]).vectors
    assert v.shape == (3, 256)
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0)
    assert cos(v[0], v[1]) > cos(v[0], v[2])
    assert np.array_equal(v, StubEmbedder().embed([
        "Isabella is planning a Valentine's Day party at Hobbs Cafe",
        "a party at Hobbs Cafe on Valentine's Day",
        "Sam waters the garden beds in Johnson Park"]).vectors)


class CountingEmbedder(StubEmbedder):
    def __init__(self) -> None:
        super().__init__()
        self.seen: list[str] = []

    def embed(self, texts, kind="document"):
        self.seen.extend(texts)
        return super().embed(texts, kind)


def test_cache_avoids_repeat_calls_and_persists(tmp_path):
    inner = CountingEmbedder()
    path = tmp_path / "emb.sqlite"
    sink, usage = ListSink(), UsageSummary()
    emb = CachedEmbedder(inner, EmbeddingCache(path), sink=sink, usage=usage)
    first = emb.embed(["alpha", "beta", "alpha"])
    assert inner.seen == ["alpha", "beta"]           # duplicates within a batch embedded once
    again = emb.embed(["beta", "alpha"])
    assert inner.seen == ["alpha", "beta"]           # served from cache
    assert np.array_equal(again[1], first[0])
    assert len(sink.of_type("embed_call")) == 1      # cache-only lookups emit no event
    assert usage.embeddings["stub/stub-hash-256"].cached == 3
    emb.cache.close()

    fresh_inner = CountingEmbedder()
    reopened = CachedEmbedder(fresh_inner, EmbeddingCache(path))
    assert np.array_equal(reopened.embed_one("alpha"), first[0])
    assert fresh_inner.seen == []                    # survived on disk


class FakeVoyageClient:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], str, str]] = []

    def embed(self, texts, model, input_type):
        self.calls.append((list(texts), model, input_type))
        vecs = [[1.0, 0.0] if input_type == "query" else [0.0, 1.0] for _ in texts]
        return type("R", (), {"embeddings": vecs, "total_tokens": 5 * len(texts)})()


def test_voyage_kind_in_cache_key_and_cost_charged():
    client = FakeVoyageClient()
    budget, sink = BudgetGuard(1.0), ListSink()
    emb = CachedEmbedder(VoyageEmbedder("voyage-4-lite", client=client), EmbeddingCache(None),
                         price_per_mtok=0.02, budget=budget, sink=sink)
    q = emb.embed_one("who is running for mayor?", "query")
    d = emb.embed_one("who is running for mayor?", "document")
    assert not np.array_equal(q, d) and len(client.calls) == 2
    assert client.calls[0][2] == "query" and client.calls[0][1] == "voyage-4-lite"
    assert budget.spent == pytest.approx(2 * 5 * 0.02 / 1e6)
    assert sink.of_type("embed_call")[0]["data"]["tokens"] == 5


def test_voyage_budget_checked_before_call():
    client = FakeVoyageClient()
    emb = CachedEmbedder(VoyageEmbedder(client=client), EmbeddingCache(None), price_per_mtok=0.02,
                         budget=BudgetGuard(0.0))
    with pytest.raises(BudgetExhausted):
        emb.embed(["anything"])
    assert client.calls == []


def test_build_embedder_from_stub_config(cfg):
    emb = build_embedder(cfg, cache_path=None)
    assert emb.provider == "stub" and emb.embed_one("hello").shape == (256,)


def test_local_embedder_reports_missing_extra():
    pytest.importorskip("numpy")
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        from smallville_lite.embed.local import LocalEmbedder

        with pytest.raises(RuntimeError, match=r"pip install -e \"\.\[local\]\""):
            LocalEmbedder().embed(["x"])
    else:
        pytest.skip("sentence-transformers installed; covered by test_local_embedder_live")


def test_local_embedder_live():
    pytest.importorskip("sentence_transformers")
    from smallville_lite.embed.local import LocalEmbedder

    v = LocalEmbedder().embed(["a party at the cafe", "a celebration at the coffee shop", "tax law"]).vectors
    assert v.shape[0] == 3 and cos(v[0], v[1]) > cos(v[0], v[2])


def test_embed_batch_type():
    assert isinstance(StubEmbedder().embed(["x"]), EmbedBatch)
