"""Contextual embedding text (lever 1): manual label, provision chain and chunk text, embedded under a variant tag.

The text is derived from the stored rows; no provision, chunk or source version is modified. Each variant is stored as
its own revision (`<embedder revision>#<variant>`), so the baseline vectors stay in place until a variant wins.
Retrieval selects the variant through `Settings.retrieval_embedding_variant`.

Variants:
- `ctx1`: `<manual source title> > <provision chain, each 'identifier - title'>` above the chunk text.
- `ctx2`: `ctx1`, plus each `(unnumbered text)` chunk takes the provision chain of the last numbered chunk that starts
  on or before its physical page (the nearest preceding provision in reading order). Pages are the only position
  signal, so chunks sharing a page with several provisions take the one ordered last by ordinal.

Run: `PYTHONPATH=. .venv-ingest/Scripts/python.exe -m app.ingest.context --variant ctx1`
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from collections.abc import Sequence
from uuid import UUID

from sqlalchemy import select

from app.core.config import get_settings
from app.db.models import Chunk, Document, DocumentVersion, Provision
from app.db.session import create_resources
from app.ingest.embedding import Embedder, load_embedder
from app.ingest.load import embed_missing

log = logging.getLogger("ingest.context")

UNNUMBERED = "(unnumbered text)"
VARIANTS = ("ctx1", "ctx2")


def embedding_revision(base_revision: str, variant: str) -> str:
    """Revision stored on chunk_embeddings rows. The baseline (empty variant) keeps the plain embedder revision."""
    return f"{base_revision}#{variant}" if variant else base_revision


def active_revision(embedder: Embedder) -> str:
    return embedding_revision(embedder.revision, get_settings().retrieval_embedding_variant)


def build_contextual_texts(
    label: str,
    provisions: Sequence,
    chunks: Sequence,
    *,
    attach_unnumbered: bool,
) -> dict[UUID, str]:
    """Contextual embedding input for every chunk. Inputs are plain rows (ids, identifiers, titles, pages, text)."""
    by_id = {provision.id: provision for provision in provisions}

    def chain(provision_id: UUID) -> str:
        parts: list[str] = []
        cursor: UUID | None = provision_id
        while cursor is not None:
            provision = by_id[cursor]
            parts.append(
                f"{provision.full_identifier} - {provision.title}" if provision.title else provision.full_identifier
            )
            cursor = provision.parent_provision_id
        return " > ".join(reversed(parts))

    hosts: list[tuple[int, int, UUID]] = []
    if attach_unnumbered:
        hosts = sorted(
            (chunk.physical_page_start, chunk.ordinal, chunk.provision_id)
            for chunk in chunks
            if chunk.physical_page_start is not None and by_id[chunk.provision_id].full_identifier != UNNUMBERED
        )

    texts: dict[UUID, str] = {}
    for chunk in chunks:
        provision_id = chunk.provision_id
        if (
            attach_unnumbered
            and by_id[provision_id].full_identifier == UNNUMBERED
            and chunk.physical_page_start is not None
        ):
            earlier = [host for host in hosts if host[0] <= chunk.physical_page_start]
            if earlier:
                provision_id = earlier[-1][2]
        texts[chunk.id] = f"{label} > {chain(provision_id)}\n{chunk.raw_text}"
    return texts


async def contextual_texts(session_factory, source_version_id: UUID, label: str, variant: str) -> dict[UUID, str]:
    async with session_factory() as session:
        provisions = (
            await session.execute(
                select(Provision.id, Provision.full_identifier, Provision.title, Provision.parent_provision_id).where(
                    Provision.source_version_id == source_version_id
                )
            )
        ).all()
        chunks = (
            await session.execute(
                select(Chunk.id, Chunk.provision_id, Chunk.ordinal, Chunk.physical_page_start, Chunk.raw_text).where(
                    Chunk.source_version_id == source_version_id
                )
            )
        ).all()
    return build_contextual_texts(label, provisions, chunks, attach_unnumbered=variant == "ctx2")


async def embed_variant(session_factory, embedder: Embedder, variant: str) -> dict[str, int]:
    """Embed every current source version under the variant revision; returns rows inserted per document."""
    async with session_factory() as session:
        versions = (
            await session.execute(
                select(Document.id, DocumentVersion.id, DocumentVersion.source_title).join(
                    DocumentVersion, DocumentVersion.id == Document.current_source_version_id
                )
            )
        ).all()
    revision = embedding_revision(embedder.revision, variant)
    inserted: dict[str, int] = {}
    for document_id, version_id, source_title in versions:
        texts = await contextual_texts(session_factory, version_id, source_title, variant)
        inserted[document_id] = await embed_missing(
            session_factory, version_id, embedder, texts=texts, revision=revision
        )
        log.info("embedded %s under %s: %d rows", document_id, revision, inserted[document_id])
    return inserted


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", required=True, choices=VARIANTS)
    args = parser.parse_args()
    settings = get_settings()
    engine, session_factory = create_resources(settings.database_url)

    async def run() -> dict[str, int]:
        try:
            embedder = await asyncio.to_thread(load_embedder, "auto", ollama_url=settings.ollama_url)
            return await embed_variant(session_factory, embedder, args.variant)
        finally:
            await engine.dispose()

    result = asyncio.run(run())
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
