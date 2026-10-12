"""Embedding adapters. Harrier (local, sentence-transformers) is preferred; nomic-embed-text via Ollama is the fallback.

Whichever adapter is used is recorded with each vector (model, revision, dimensions), so vectors from
different models never mix. A fallback is always logged; it is never silent.
"""

from __future__ import annotations

import logging
from typing import Protocol

import httpx

from app.core.config import get_settings

log = logging.getLogger("ingest.embedding")

HARRIER_MODEL = "microsoft/harrier-oss-v1-0.6b"
HARRIER_TASK = "Given a question about civil engineering design criteria, retrieve passages from the criteria manual that answer it"
NOMIC_MODEL = "nomic-embed-text"


class Embedder(Protocol):
    model: str
    revision: str
    dimensions: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class HarrierEmbedder:
    def __init__(self, device: str = "cuda") -> None:
        from huggingface_hub import snapshot_download
        from sentence_transformers import SentenceTransformer

        self.model = HARRIER_MODEL
        self._st = SentenceTransformer(HARRIER_MODEL, device=device, model_kwargs={"dtype": "auto"})
        self.revision = snapshot_download(HARRIER_MODEL).replace("\\", "/").rstrip("/").split("/")[-1]
        self.dimensions = int(self._st.get_embedding_dimension())

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self._st.encode(texts, batch_size=16, convert_to_numpy=True, show_progress_bar=False)
        return [[float(x) for x in row] for row in vectors]

    def embed_query(self, text: str) -> list[float]:
        vector = self._st.encode([text], prompt=f"Instruct: {HARRIER_TASK}\nQuery: ", convert_to_numpy=True)[0]
        return [float(x) for x in vector]


class OllamaEmbedder:
    def __init__(self, base_url: str | None = None, model: str = NOMIC_MODEL) -> None:
        self._url = (base_url or get_settings().ollama_url).rstrip("/")
        self.model = model
        self._client = httpx.Client(timeout=httpx.Timeout(connect=5.0, read=300.0, write=60.0, pool=5.0))
        tags = self._client.get(f"{self._url}/api/tags").raise_for_status().json()["models"]
        match = next((m for m in tags if m.get("name") in (model, f"{model}:latest")), None)
        if match is None:
            raise RuntimeError(f"Ollama model {model!r} is not installed")
        self.revision = match["digest"]
        self.dimensions = len(self._embed(["search_document: dimension probe"])[0])

    def _embed(self, inputs: list[str]) -> list[list[float]]:
        response = self._client.post(f"{self._url}/api/embed", json={"model": self.model, "input": inputs})
        response.raise_for_status()
        return response.json()["embeddings"]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed([f"search_document: {text}" for text in texts])

    def embed_query(self, text: str) -> list[float]:
        return self._embed([f"search_query: {text}"])[0]


def load_embedder(choice: str = "auto", *, device: str = "cuda", ollama_url: str | None = None) -> Embedder:
    if choice in ("auto", "harrier"):
        try:
            embedder = HarrierEmbedder(device=device)
            log.info("embedder: %s revision=%s dims=%s", embedder.model, embedder.revision[:12], embedder.dimensions)
            return embedder
        except Exception as error:  # noqa: BLE001 - any load failure triggers the documented fallback
            if choice == "harrier":
                raise
            log.warning("Harrier is not usable (%s: %s); falling back to %s", type(error).__name__, error, NOMIC_MODEL)
    embedder = OllamaEmbedder(ollama_url)
    log.info("embedder: %s digest=%s dims=%s", embedder.model, embedder.revision[:12], embedder.dimensions)
    return embedder
