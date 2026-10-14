"""GPU budget knobs: documented defaults, and batched reranker scoring keeps order and batch bounds."""

import contextlib
from types import SimpleNamespace

from app.core.config import Settings
from app.rag import rerank


def test_gpu_budget_defaults():
    settings = Settings(_env_file=None)
    assert settings.ollama_num_ctx == 8192
    assert settings.ollama_keep_alive == -1
    assert settings.reviewer_num_gpu == 0
    assert settings.reranker_batch_size == 8


class _Tensor:
    def __init__(self, values):
        self.values = values

    def view(self, *_):
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return list(self.values)


class _Encoded(dict):
    def to(self, _device):
        return self


class _Tokenizer:
    def __call__(self, queries, passages, **_):
        return _Encoded(texts=list(passages))


class _Model:
    def __init__(self):
        self.batches: list[int] = []

    def __call__(self, *, texts):
        self.batches.append(len(texts))
        return SimpleNamespace(logits=_Tensor([float(len(text)) for text in texts]))


def _stub_reranker(model: _Model) -> rerank.Reranker:
    reranker = rerank.Reranker.__new__(rerank.Reranker)
    reranker._torch = SimpleNamespace(no_grad=contextlib.nullcontext, cuda=SimpleNamespace(empty_cache=lambda: None))
    reranker.tokenizer = _Tokenizer()
    reranker.model = model
    return reranker


def test_batched_scores_keep_order_and_respect_batch_size(monkeypatch):
    passages = [f"passage {n}" * (n % 4 + 1) for n in range(11)]
    monkeypatch.setattr(rerank, "get_settings", lambda: Settings(_env_file=None, reranker_batch_size=3))
    model = _Model()
    scores = _stub_reranker(model).scores("query", passages)
    assert scores == [float(len(text)) for text in passages]
    assert model.batches == [3, 3, 3, 2]
