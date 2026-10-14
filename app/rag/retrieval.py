"""Hybrid retrieval over the current source version of each registered document.

Postgres full-text (over identifier + title + text) and pgvector cosine (over embeddings made by exactly the
embedder's model and revision) are fused with reciprocal rank fusion. Only each document's current source
version is searched; nothing here writes to the database.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, replace
from functools import cache
from typing import Sequence
from uuid import UUID

from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import get_settings
from app.db.models import Chunk, Document, DocumentVersion, Provision
from app.ingest.abbreviations import derive_definitions, expand_query
from app.ingest.context import active_revision
from app.ingest.embedding import Embedder, load_embedder
from app.ingest.structure import provision_path
from app.rag.rerank import rerank_scores

RRF_K = 60
POOL_SIZE = 50
RERANK_HEAD = 4
FULL_TEXT = "full-text"
VECTOR = "vector"
SECTION_LOOKUP = "section-lookup"

SessionFactory = async_sessionmaker[AsyncSession]

_FTS_TOKEN = re.compile(r"[A-Za-z0-9]+")


def fts_tsquery(query: str) -> str:
    """Tokenised tsquery text for `to_tsquery('english', ...)`: the question's words joined by AND.

    Tokens are restricted to alphanumerics so tsquery syntax in the question cannot be injected. The OR joiner was
    measured (`eval/retrieval-experiments.md`, rows 1a-1d) and lost to vector ranking at every weight.
    """
    terms = [token.lower() for token in _FTS_TOKEN.findall(query) if len(token) > 1]
    return " & ".join(dict.fromkeys(terms))


FULL_TEXT_SQL = text(
    """
    SELECT c.id
    FROM chunks c
    JOIN provisions p ON p.id = c.provision_id
    JOIN documents d ON d.current_source_version_id = c.source_version_id,
         to_tsquery('english', :query) AS q
    WHERE (CAST(:document_ids AS text[]) IS NULL OR d.id = ANY(CAST(:document_ids AS text[])))
      AND to_tsvector('english', p.full_identifier || ' ' || coalesce(p.title, '') || ' ' || c.raw_text) @@ q
    ORDER BY ts_rank_cd(
        to_tsvector('english', p.full_identifier || ' ' || coalesce(p.title, '') || ' ' || c.raw_text), q
    ) DESC, c.id
    LIMIT :pool
    """
)
VECTOR_SQL = text(
    """
    SELECT c.id
    FROM chunk_embeddings e
    JOIN chunks c ON c.id = e.chunk_id
    JOIN documents d ON d.current_source_version_id = c.source_version_id
    WHERE (CAST(:document_ids AS text[]) IS NULL OR d.id = ANY(CAST(:document_ids AS text[])))
      AND e.model = :model AND e.revision = :revision AND e.dimensions = :dimensions
    ORDER BY e.embedding <=> CAST(:vector AS vector), c.id
    LIMIT :pool
    """
)
PROVISION_CHAIN_SQL = text(
    """
    WITH RECURSIVE chain AS (
        SELECT p.id AS leaf, p.parent_provision_id AS parent, p.full_identifier, p.title, 0 AS depth
        FROM provisions p WHERE p.id = ANY(:provision_ids)
        UNION ALL
        SELECT chain.leaf, p.parent_provision_id, p.full_identifier, p.title, chain.depth + 1
        FROM chain JOIN provisions p ON p.id = chain.parent
    )
    SELECT leaf, full_identifier, title FROM chain ORDER BY leaf, depth DESC
    """
)


@dataclass(frozen=True)
class RetrievedChunk:
    document_id: str
    provision_identifier: str
    provision_title: str | None
    page_start: int | None
    page_end: int | None
    text: str
    breadcrumb: str
    score: float
    found_by: tuple[str, ...]
    extraction_warnings: tuple[str, ...]
    extraction_method: str | None
    children: tuple[str, ...] = ()  # direct child provisions ("identifier title"); only set by get_provision_chunks
    chunk_id: UUID | None = None
    fts_rank: int | None = None
    vector_rank: int | None = None
    rrf_score: float | None = None
    is_duplicate: bool = False  # True when the ingest stored this as a repeated heading "<id> (duplicate N)"
    neighbor_of: UUID | None = None  # set on adjacent chunks added by expand_neighbors; the hit they extend
    rerank_score: float | None = None  # cross-encoder logit when the reranker scored this hit; None otherwise


@cache
def get_embedder() -> Embedder:
    return load_embedder("auto", ollama_url=get_settings().ollama_url)


def fuse(
    rankings: dict[str, list[UUID]], k: int, weights: dict[str, float] | None = None
) -> list[tuple[UUID, float, tuple[str, ...]]]:
    """Weighted reciprocal rank fusion: score = sum over lists of w / (RRF_K + rank); ties break on chunk id."""
    scores: dict[UUID, float] = {}
    found_by: dict[UUID, list[str]] = {}
    for name, ranked in rankings.items():
        weight = 1.0 if weights is None else weights.get(name, 1.0)
        for rank, chunk_id in enumerate(ranked, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + weight / (RRF_K + rank)
            found_by.setdefault(chunk_id, []).append(name)
    best = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], str(chunk_id)))[:k]
    return [(chunk_id, scores[chunk_id], tuple(sorted(found_by[chunk_id]))) for chunk_id in best]


def diversify(ranked: list[RetrievedChunk], k: int, per_document: int = 3) -> list[RetrievedChunk]:
    """Take at most three top-ranked hits per manual before merging by fused score."""
    if k <= 0:
        return []
    groups: dict[str, list[RetrievedChunk]] = {}
    for hit in ranked:
        groups.setdefault(hit.document_id, []).append(hit)
    selected = [hit for group in groups.values() for hit in group[:per_document]]
    selected.sort(key=lambda hit: (-hit.score, hit.document_id, hit.provision_identifier))
    if len(selected) > k:
        selected = []
        for index in range(per_document):
            selected.extend(group[index] for group in groups.values() if len(group) > index)
            if len(selected) >= k:
                return selected[:k]
    return selected[:k]


def rerank_head_order(order: list[int], count: int, head: int = RERANK_HEAD) -> list[int]:
    """Pool indices in final order: the reranker's top `head` first, then the rest in their fused (vector) order.

    Measured as design B4 (`eval/retrieval-experiments.md`, row 10). A full reranker reorder lost golden R@8 because
    the cross-encoder pushed the precise provision out of the top 8; keeping the fused order below the head does not.
    """
    leading = order[:head]
    chosen = set(leading)
    return leading + [index for index in range(count) if index not in chosen]


async def source_version_metadata(sessions: SessionFactory, document_ids: Sequence[str] | None = None):
    """Read recorded edition/effective date from current registered source versions."""
    query = select(Document.id, DocumentVersion.edition, DocumentVersion.effective_date).join(
        DocumentVersion, DocumentVersion.id == Document.current_source_version_id
    )
    if document_ids:
        query = query.where(Document.id.in_(list(document_ids)))
    async with sessions() as session:
        return (await session.execute(query)).all()


_definitions: dict[str, str] | None = None


async def abbreviation_definitions(sessions: SessionFactory) -> dict[str, str]:
    """Abbreviation definitions derived once per process from the stored chunk text (query-side only)."""
    global _definitions
    if _definitions is None:
        async with sessions() as session:
            texts = (await session.execute(select(Chunk.raw_text))).scalars().all()
        _definitions = derive_definitions(texts)
    return _definitions


async def hybrid_search(
    sessions: SessionFactory,
    embedder: Embedder,
    query: str,
    *,
    document_ids: Sequence[str] | None = None,
    k: int = 8,
    expand_neighbors: bool = False,
) -> list[RetrievedChunk]:
    """Top-k chunks from the current source versions (optionally only `document_ids`), best first.

    All manuals are ranked in one fused list. Per-manual fusion with a per-manual quota was measured and removed:
    it gave every manual's first hit the same score, so cross-manual order was arbitrary
    (`eval/retrieval-experiments.md`). The full-text leg runs only when `retrieval_fts_weight` is positive; it is off
    by default because it did not improve the measured ranking.
    """
    if k <= 0 or (document_ids is not None and not document_ids):
        return []
    settings = get_settings()
    if settings.retrieval_expand_abbreviations:
        query = expand_query(query, await abbreviation_definitions(sessions))
    query_vector = await asyncio.to_thread(embedder.embed_query, query)
    vector = "[" + ",".join(repr(float(x)) for x in query_vector) + "]"
    scope = list(document_ids) if document_ids is not None else None
    revision = active_revision(embedder)
    rankings: dict[str, list[UUID]] = {}
    weights = {VECTOR: 1.0}
    async with sessions() as session:
        nearest = list(
            (
                await session.execute(
                    VECTOR_SQL,
                    {
                        "document_ids": scope,
                        "model": embedder.model,
                        "revision": revision,
                        "dimensions": len(query_vector),
                        "vector": vector,
                        "pool": POOL_SIZE,
                    },
                )
            )
            .scalars()
            .all()
        )
        rankings[VECTOR] = nearest
        full_text: list[UUID] = []
        if settings.retrieval_fts_weight > 0:
            tsquery = fts_tsquery(query)
            full_text = list(
                (await session.execute(FULL_TEXT_SQL, {"query": tsquery, "document_ids": scope, "pool": POOL_SIZE}))
                .scalars()
                .all()
            )
            rankings[FULL_TEXT] = full_text
            weights[FULL_TEXT] = settings.retrieval_fts_weight
        pool_k = max(k, settings.retrieval_rerank_top_n) if settings.retrieval_rerank else k
        fused = fuse(rankings, pool_k, weights)
        fts_ranks = {chunk_id: rank for rank, chunk_id in enumerate(full_text, start=1)}
        vector_ranks = {chunk_id: rank for rank, chunk_id in enumerate(nearest, start=1)}
        ranked = [
            (chunk_id, score, found_by, fts_ranks.get(chunk_id), vector_ranks.get(chunk_id))
            for chunk_id, score, found_by in fused
        ]
        hits = await hydrate(session, ranked)
        if settings.retrieval_rerank and hits:
            passages = [f"{hit.breadcrumb}\n{hit.text}" for hit in hits]
            scores = await asyncio.to_thread(rerank_scores, query, passages, settings.retrieval_rerank_model)
            if scores is not None:
                order = sorted(range(len(hits)), key=lambda index: (-scores[index], index))
                hits = [
                    replace(hits[index], rerank_score=scores[index]) for index in rerank_head_order(order, len(hits))
                ]
        hits = hits[:k]
        if not expand_neighbors:
            return hits
        return hits + await neighbour_chunks(session, hits)


async def hydrate(
    session: AsyncSession, ranked: list[tuple[UUID, float, tuple[str, ...], int | None, int | None]]
) -> list[RetrievedChunk]:
    if not ranked:
        return []
    ranked = [(*item, None, None) if len(item) == 3 else item for item in ranked]
    rows = {
        row.id: row
        for row in await session.execute(
            select(
                Chunk.id,
                Chunk.provision_id,
                Chunk.raw_text,
                Chunk.physical_page_start,
                Chunk.physical_page_end,
                Chunk.extraction_method,
                Chunk.extraction_warnings,
                Provision.full_identifier,
                Provision.title,
                Document.id.label("document_id"),
            )
            .join(Provision, Provision.id == Chunk.provision_id)
            .join(Document, Document.current_source_version_id == Chunk.source_version_id)
            .where(Chunk.id.in_([item[0] for item in ranked]))
        )
    }
    chains: dict[UUID, list[tuple[str, str | None, int | None]]] = {}
    chain_rows = await session.execute(
        PROVISION_CHAIN_SQL, {"provision_ids": list({row.provision_id for row in rows.values()})}
    )
    for leaf, identifier, title in chain_rows:
        chain = chains.setdefault(leaf, [])
        chain.append((identifier, title, len(chain) - 1 if chain else None))
    return [
        RetrievedChunk(
            document_id=rows[chunk_id].document_id,
            provision_identifier=split_duplicate(rows[chunk_id].full_identifier)[0],
            is_duplicate=split_duplicate(rows[chunk_id].full_identifier)[1],
            provision_title=rows[chunk_id].title,
            page_start=rows[chunk_id].physical_page_start,
            page_end=rows[chunk_id].physical_page_end,
            text=rows[chunk_id].raw_text,
            breadcrumb=provision_path(
                chains[rows[chunk_id].provision_id], len(chains[rows[chunk_id].provision_id]) - 1
            ),
            score=score,
            found_by=found_by,
            extraction_warnings=tuple(rows[chunk_id].extraction_warnings),
            extraction_method=rows[chunk_id].extraction_method,
            chunk_id=chunk_id,
            fts_rank=fts_rank,
            vector_rank=vector_rank,
            rrf_score=score,
        )
        for chunk_id, score, found_by, fts_rank, vector_rank in ranked
    ]


_DUPLICATE_SUFFIX = re.compile(r" \(duplicate \d+\)$")


def split_duplicate(identifier: str) -> tuple[str, bool]:
    """The section identifier without the ingest-appended ' (duplicate N)' marker, and whether the marker was present.

    A repeated heading is stored as its own provision (see `app/ingest/structure.py`); its text is real, but its
    identifier can be a stub's duplicate, so callers see the section number plus a flag rather than the raw suffix.
    """
    base = _DUPLICATE_SUFFIX.sub("", identifier)
    return base, base != identifier


async def neighbour_chunks(session: AsyncSession, hits: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Adjacent chunks (ordinal -1/+1) of the same provision as each hit, excluding chunks already returned.

    Each neighbour carries `neighbor_of`, the hit it extends. Neighbours are ordered by hit rank, then ordinal, and do
    not count toward `k`.
    """
    anchors = {hit.chunk_id for hit in hits if hit.chunk_id is not None}
    if not anchors:
        return []
    anchor_rows = (
        await session.execute(select(Chunk.id, Chunk.provision_id, Chunk.ordinal).where(Chunk.id.in_(list(anchors))))
    ).all()
    wanted: dict[tuple[UUID, int], UUID] = {}
    for row in anchor_rows:
        for ordinal in (row.ordinal - 1, row.ordinal + 1):
            if ordinal >= 0:
                wanted.setdefault((row.provision_id, ordinal), row.id)
    if not wanted:
        return []
    candidates = (
        await session.execute(
            select(Chunk.id, Chunk.provision_id, Chunk.ordinal).where(
                Chunk.provision_id.in_({provision_id for provision_id, _ in wanted})
            )
        )
    ).all()
    found: dict[UUID, tuple[UUID, int]] = {}
    for row in candidates:
        source = wanted.get((row.provision_id, row.ordinal))
        if source is not None and row.id not in anchors:
            found.setdefault(row.id, (source, row.ordinal))
    if not found:
        return []
    rank = {hit.chunk_id: index for index, hit in enumerate(hits)}
    ordered = sorted(found, key=lambda chunk_id: (rank[found[chunk_id][0]], found[chunk_id][1]))
    neighbours = await hydrate(session, [(chunk_id, 0.0, (), None, None) for chunk_id in ordered])
    return [replace(item, neighbor_of=found[item.chunk_id][0]) for item in neighbours]


_NUMBER = r"\d+(?:\.\d+)+"
_SECTION_WORD = r"(?:sec(?:tion)?\.?|§)"
_DOTTED = re.compile(rf"^(?:{_SECTION_WORD}\s*)?({_NUMBER})\.?$", re.IGNORECASE)
_WHOLE_SECTION = re.compile(rf"^{_SECTION_WORD}\s*(\d+)\.?$", re.IGNORECASE)
_APPENDIX = re.compile(r"^appendix\s+([A-Za-z])\.?$", re.IGNORECASE)
_REF_IN_TEXT = re.compile(
    rf"(?<![\w.])(?:{_SECTION_WORD}\s*\d+(?:\.\d+)*|appendix\s+[A-Za-z](?![A-Za-z]))", re.IGNORECASE
)


def normalize_section_ref(text: str) -> str | None:
    """The provision identifier a reference such as 'Sec. 2.3.0', '§2.3', '2.3.0', 'Section 4' or 'Appendix B'
    names, or None. A bare number without a dot, or a number followed by words ('10.5 miles'), is not a section."""
    ref = " ".join(text.split())
    if match := _DOTTED.match(ref):
        return match.group(1)
    if match := _WHOLE_SECTION.match(ref):
        return f"SECTION {int(match.group(1))}"
    if match := _APPENDIX.match(ref):
        return f"APPENDIX {match.group(1).upper()}"
    return None


def find_section_ref(text: str) -> str | None:
    """The first identifier named with an explicit marker (sec, section, §, appendix) inside free text."""
    for match in _REF_IN_TEXT.finditer(text):
        if ref := normalize_section_ref(match.group(0)):
            return ref
    return None


async def get_provision_chunks(
    sessions: SessionFactory, identifier: str, *, document_ids: Sequence[str] | None = None
) -> list[RetrievedChunk]:
    """Every chunk of each provision whose identifier equals `identifier`, from each document's current source
    version, ordered by document then ordinal. `children` lists the provision's direct child provisions."""
    ref = normalize_section_ref(identifier)
    if ref is None or (document_ids is not None and not document_ids):
        return []
    query = (
        select(Chunk.id, Provision.id.label("provision_id"))
        .join(Provision, Provision.id == Chunk.provision_id)
        .join(Document, Document.current_source_version_id == Chunk.source_version_id)
        .where(func.upper(Provision.full_identifier) == ref)
        .order_by(Document.id, Chunk.ordinal)
    )
    if document_ids is not None:
        query = query.where(Document.id.in_(list(document_ids)))
    async with sessions() as session:
        found = (await session.execute(query)).all()
        if not found:
            return []
        chunks = await hydrate(session, [(row.id, 1.0, (SECTION_LOOKUP,)) for row in found])
        children: dict[UUID, list[str]] = {}
        child_rows = await session.execute(
            select(Provision.parent_provision_id, Provision.full_identifier, Provision.title)
            .where(Provision.parent_provision_id.in_({row.provision_id for row in found}))
            .order_by(Provision.sibling_order)
        )
        for parent, child_identifier, title in child_rows:
            children.setdefault(parent, []).append(f"{child_identifier} {title}" if title else child_identifier)
    return [replace(chunk, children=tuple(children.get(row.provision_id, ()))) for chunk, row in zip(chunks, found)]
