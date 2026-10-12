"""Cross-encoder reranking (lever 3) over the fused top-N candidates.

The model loads lazily on first use, in fp16 on CUDA. If it cannot load, the first failure is logged as a warning and the
fused order is kept; retrieval never fails because of the reranker.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

from app.core.config import get_settings

log = logging.getLogger("rag.rerank")


class Reranker:
    def __init__(self, model_name: str) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available for the reranker")
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = (
            AutoModelForSequenceClassification.from_pretrained(model_name, torch_dtype=torch.float16).cuda().eval()
        )
        self._torch = torch

    def scores(self, query: str, passages: list[str]) -> list[float]:
        torch = self._torch
        batch_size = get_settings().reranker_batch_size
        logits: list[float] = []
        with torch.no_grad():
            for start in range(0, len(passages), batch_size):
                chunk = passages[start : start + batch_size]
                inputs = self.tokenizer(
                    [query] * len(chunk),
                    chunk,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                ).to("cuda")
                logits.extend(self.model(**inputs).logits.view(-1).float().cpu().tolist())
        # Return transient activation blocks to the driver so Ollama's free-VRAM estimate stays accurate.
        torch.cuda.empty_cache()
        return logits


@dataclass
class _Slot:
    reranker: Reranker | None = None
    failed: bool = False


_slot = _Slot()
_lock = threading.Lock()


def get_reranker(model_name: str) -> Reranker | None:
    """Lazily loaded singleton; None when unavailable (warning logged once)."""
    with _lock:
        if _slot.reranker is None and not _slot.failed:
            started = time.perf_counter()
            try:
                _slot.reranker = Reranker(model_name)
                log.info("reranker %s loaded in %.1f s", model_name, time.perf_counter() - started)
            except Exception as exc:  # any load failure falls back to the fused order
                _slot.failed = True
                log.warning("reranker %s unavailable, keeping fused order: %s", model_name, exc)
        return _slot.reranker


def rerank_scores(query: str, passages: list[str], model_name: str) -> list[float] | None:
    """Raw cross-encoder logit per passage (same order), or None when the reranker is unavailable."""
    reranker = get_reranker(model_name)
    if reranker is None:
        return None
    return [float(score) for score in reranker.scores(query, passages)]
