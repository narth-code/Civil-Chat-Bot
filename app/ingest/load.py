"""Stage 2 and 3: persist a built Structure as Provision/Chunk rows and compute chunk embeddings.

Every write is scoped to one registered source version. Loading refuses to touch a version that
already has provisions unless a rebuild is requested; embedding is idempotent.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from uuid import UUID, uuid4

from sqlalchemy import delete, func, insert, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.db.models import Chunk, ChunkEmbedding, Document, DocumentVersion, Provision
from app.ingest.embedding import Embedder
from app.ingest.structure import Structure, provision_path

log = logging.getLogger("ingest.load")

SessionFactory = async_sessionmaker[AsyncSession]
INSERT_BATCH = 500


class VersionAlreadyLoaded(RuntimeError):
    """The source version already has provisions and no rebuild was requested."""


async def get_registered_version(session_factory: SessionFactory, document_id: str) -> tuple[UUID, str, str]:
    """Return (source_version_id, sha256, snapshot-relative path) of the document's current version."""
    async with session_factory() as session:
        row = (
            await session.execute(
                select(DocumentVersion.id, DocumentVersion.sha256, DocumentVersion.snapshot_path)
                .join(Document, Document.current_source_version_id == DocumentVersion.id)
                .where(Document.id == document_id)
            )
        ).one_or_none()
    if row is None:
        raise LookupError(
            f"document {document_id!r} has no registered current source version (run python -m app.db.bootstrap)"
        )
    return row.id, row.sha256, row.snapshot_path


async def count_rows(session_factory: SessionFactory, source_version_id: UUID) -> dict[str, int]:
    """Row counts for one source version."""
    async with session_factory() as session:
        provisions = await session.scalar(
            select(func.count()).select_from(Provision).where(Provision.source_version_id == source_version_id)
        )
        chunks = await session.scalar(
            select(func.count()).select_from(Chunk).where(Chunk.source_version_id == source_version_id)
        )
        embeddings = await session.scalar(
            select(func.count())
            .select_from(ChunkEmbedding)
            .join(Chunk, Chunk.id == ChunkEmbedding.chunk_id)
            .where(Chunk.source_version_id == source_version_id)
        )
    return {"provisions": provisions or 0, "chunks": chunks or 0, "embeddings": embeddings or 0}


async def delete_version_content(session: AsyncSession, source_version_id: UUID) -> None:
    """Delete only this version's embeddings, chunks and provisions, in foreign-key order."""
    chunk_ids = select(Chunk.id).where(Chunk.source_version_id == source_version_id)
    await session.execute(delete(ChunkEmbedding).where(ChunkEmbedding.chunk_id.in_(chunk_ids)))
    await session.execute(delete(Chunk).where(Chunk.source_version_id == source_version_id))
    await session.execute(delete(Provision).where(Provision.source_version_id == source_version_id))


async def load_structure(
    session_factory: SessionFactory,
    source_version_id: UUID,
    structure: Structure,
    rebuild: bool = False,
) -> dict[str, int]:
    """Insert the structure in one transaction; returns the provision and chunk counts written."""
    async with session_factory() as session, session.begin():
        existing = await session.scalar(
            select(func.count()).select_from(Provision).where(Provision.source_version_id == source_version_id)
        )
        if existing and not rebuild:
            raise VersionAlreadyLoaded(
                f"source version {source_version_id} already has {existing} provisions; pass rebuild=True (--rebuild) to replace them"
            )
        if existing:
            await delete_version_content(session, source_version_id)

        provision_ids = [uuid4() for _ in structure.provisions]
        provision_rows = [
            {
                "id": provision_ids[index],
                "source_version_id": source_version_id,
                "parent_provision_id": provision_ids[draft.parent_index] if draft.parent_index is not None else None,
                "full_identifier": draft.full_identifier,
                "title": draft.title,
                "sibling_order": draft.sibling_order,
            }
            for index, draft in enumerate(structure.provisions)
        ]
        for start in range(0, len(provision_rows), INSERT_BATCH):
            await session.execute(insert(Provision), provision_rows[start : start + INSERT_BATCH])

        chunk_rows = [
            {
                "id": uuid4(),
                "source_version_id": source_version_id,
                "provision_id": provision_ids[draft.provision_index],
                "ordinal": draft.ordinal,
                "raw_text": draft.raw_text,
                "physical_page_start": draft.page_start,
                "physical_page_end": draft.page_end,
                "table_markdown": draft.table_markdown,
                "extraction_method": draft.extraction_method,
                "extraction_confidence": draft.extraction_confidence,
                "quality_flags": list(draft.flags),
                "extraction_warnings": list(draft.warnings),
            }
            for draft in structure.chunks
        ]
        for start in range(0, len(chunk_rows), INSERT_BATCH):
            await session.execute(insert(Chunk), chunk_rows[start : start + INSERT_BATCH])
    return {"provisions": len(provision_rows), "chunks": len(chunk_rows)}


def embedding_input(breadcrumb: str, raw_text: str) -> str:
    return f"{breadcrumb}\n{raw_text}"


async def embed_missing(
    session_factory: SessionFactory,
    source_version_id: UUID,
    embedder: Embedder,
    batch_size: int = 32,
    progress=None,
    texts: dict[UUID, str] | None = None,
    revision: str | None = None,
) -> int:
    """Embed every chunk lacking a vector for (model, revision, input hash); returns rows inserted.

    `texts` replaces the breadcrumb input for each chunk (contextual variants); `revision` stores them under a variant tag.
    """
    rev = revision or embedder.revision
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
                select(Chunk.id, Chunk.provision_id, Chunk.raw_text)
                .where(Chunk.source_version_id == source_version_id)
                .order_by(Chunk.provision_id, Chunk.ordinal)
            )
        ).all()
        have = set(
            (
                await session.execute(
                    select(ChunkEmbedding.chunk_id, ChunkEmbedding.embedding_input_sha256)
                    .join(Chunk, Chunk.id == ChunkEmbedding.chunk_id)
                    .where(
                        Chunk.source_version_id == source_version_id,
                        ChunkEmbedding.model == embedder.model,
                        ChunkEmbedding.revision == rev,
                    )
                )
            ).all()
        )

    index_of = {row.id: position for position, row in enumerate(provisions)}
    tuples = [(row.full_identifier, row.title, index_of.get(row.parent_provision_id)) for row in provisions]
    breadcrumbs: dict[UUID, str] = {}
    todo: list[tuple[UUID, str, str]] = []
    for chunk in chunks:
        if texts is not None:
            text = texts[chunk.id]
        else:
            if chunk.provision_id not in breadcrumbs:
                breadcrumbs[chunk.provision_id] = provision_path(tuples, index_of[chunk.provision_id])
            text = embedding_input(breadcrumbs[chunk.provision_id], chunk.raw_text)
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if (chunk.id, digest) not in have:
            todo.append((chunk.id, digest, text))

    inserted = 0
    for start in range(0, len(todo), batch_size):
        batch = todo[start : start + batch_size]
        vectors = await asyncio.to_thread(embedder.embed_documents, [text for _, _, text in batch])
        if len(vectors) != len(batch):
            raise RuntimeError(f"embedder returned {len(vectors)} vectors for {len(batch)} inputs")
        rows = [
            {
                "id": uuid4(),
                "chunk_id": chunk_id,
                "model": embedder.model,
                "revision": rev,
                "dimensions": len(vector),
                "embedding_input_sha256": digest,
                "embedding": vector,
            }
            for (chunk_id, digest, _), vector in zip(batch, vectors, strict=True)
        ]
        async with session_factory() as session, session.begin():
            await session.execute(insert(ChunkEmbedding), rows)
        inserted += len(rows)
        if progress:
            progress(inserted, len(todo))
    log.info("embedded %s chunks (%s already had vectors)", inserted, len(chunks) - len(todo))
    return inserted
