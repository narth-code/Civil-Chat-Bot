"""Hybrid retrieval: pure rank fusion plus database-backed search against civil_chatbot_test only."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from uuid import UUID, uuid4

import pytest
import pytest_asyncio
from sqlalchemy import delete, select, update

from app.core.config import Settings, get_settings
from app.db.models import Chunk, ChunkEmbedding, Document, DocumentVersion, Provision
from app.db.session import create_resources
from app.ingest import context
from app.rag import retrieval
from app.rag.retrieval import (
    FULL_TEXT,
    RRF_K,
    SECTION_LOOKUP,
    VECTOR,
    RetrievedChunk,
    find_section_ref,
    diversify,
    fts_tsquery,
    fuse,
    get_provision_chunks,
    hybrid_search,
    normalize_section_ref,
    rerank_head_order,
    source_version_metadata,
    split_duplicate,
)

# --- pure rank fusion ---


def ids(count: int) -> list[UUID]:
    return [UUID(int=index + 1) for index in range(count)]


def test_chunk_found_by_both_lists_outranks_chunks_found_by_one():
    a, b, c = ids(3)
    fused = fuse({FULL_TEXT: [a, b], VECTOR: [c, b]}, k=10)
    assert [chunk_id for chunk_id, _, _ in fused][0] == b
    scores = {chunk_id: score for chunk_id, score, _ in fused}
    assert scores[b] == pytest.approx(2 / (RRF_K + 2))
    assert scores[b] > scores[a] and scores[b] > scores[c]


def test_found_by_names_every_list_that_returned_the_chunk():
    a, b, c = ids(3)
    by_id = {chunk_id: found_by for chunk_id, _, found_by in fuse({FULL_TEXT: [a, b], VECTOR: [b, c]}, k=10)}
    assert by_id[a] == (FULL_TEXT,)
    assert by_id[b] == (FULL_TEXT, VECTOR)
    assert by_id[c] == (VECTOR,)


def test_ties_are_broken_deterministically_by_chunk_id():
    a, b = ids(2)
    # a is first in one list, b is first in the other: identical scores
    first = fuse({FULL_TEXT: [b, a], VECTOR: [a, b]}, k=10)
    second = fuse({VECTOR: [a, b], FULL_TEXT: [b, a]}, k=10)
    assert [chunk_id for chunk_id, _, _ in first] == [a, b]
    assert [chunk_id for chunk_id, _, _ in second] == [a, b]


def test_fusion_truncates_to_k_and_handles_empty_lists():
    a, b, c = ids(3)
    assert [chunk_id for chunk_id, _, _ in fuse({FULL_TEXT: [a, b, c], VECTOR: []}, k=2)] == [a, b]
    assert fuse({FULL_TEXT: [], VECTOR: []}, k=5) == []


def test_rerank_head_puts_reranked_top_first_then_keeps_fused_order():
    # the first four reranked candidates lead; every other pool candidate keeps its fused (vector) order
    assert rerank_head_order([5, 2, 0, 7, 6, 1, 3, 4], 8) == [5, 2, 0, 7, 1, 3, 4, 6]
    assert rerank_head_order([2, 0, 1], 3) == [2, 0, 1]
    assert rerank_head_order([1, 0], 2, head=4) == [1, 0]


def test_diverse_fusion_keeps_each_manual_when_one_has_more_high_scoring_hits():
    hits = [
        RetrievedChunk(
            document_id=doc,
            provision_identifier=str(i),
            provision_title=None,
            page_start=None,
            page_end=None,
            text=f"chunk {i}",
            breadcrumb="",
            score=1 - i / 100,
            found_by=(VECTOR,),
            extraction_warnings=(),
            extraction_method=None,
        )
        for i, doc in enumerate(["drainage"] * 9 + ["fire", "transportation", "environmental"])
    ]
    chosen = diversify(hits, 8)
    assert len(chosen) == 6
    assert {hit.document_id for hit in chosen} == {"drainage", "fire", "transportation", "environmental"}
    assert len([hit for hit in chosen if hit.document_id == "drainage"]) <= 3
    assert diversify(hits, 2)[0].provision_identifier == "0"


# --- full-text query construction and the full-text switch ---


def test_fts_tsquery_joins_words_with_and_and_drops_syntax():
    assert fts_tsquery("Detention ponds!") == "detention & ponds"
    assert fts_tsquery("pond pond") == "pond"
    assert fts_tsquery("alpha|beta & gamma") == "alpha & beta & gamma"
    assert fts_tsquery("a b") == ""


def test_full_text_leg_is_off_by_default():
    assert get_settings().retrieval_fts_weight == 0.0


BASELINE_RETRIEVAL = {"retrieval_embedding_variant": "", "retrieval_rerank": False}


def pin_retrieval_settings(monkeypatch, **update) -> None:
    base = get_settings().model_copy(update={**BASELINE_RETRIEVAL, **update})
    monkeypatch.setattr(retrieval, "get_settings", lambda: base)
    monkeypatch.setattr(context, "get_settings", lambda: base)


@pytest.fixture(autouse=True)
def baseline_retrieval_settings(monkeypatch):
    """Corpus fixtures embed under the bare embedder revision and expect no reranker: pin that configuration."""
    pin_retrieval_settings(monkeypatch)


@pytest.fixture
def fts_enabled(monkeypatch):
    pin_retrieval_settings(monkeypatch, retrieval_fts_weight=1.0)


# --- database-backed search ---

MODEL = "fake-test-embedder"
REVISION = "rev-1"


class FakeEmbedder:
    """Deterministic 3-dimensional vectors: documents are never embedded here, queries come from a table."""

    model = MODEL
    revision = REVISION
    dimensions = 3

    def __init__(self, queries: dict[str, list[float]]) -> None:
        self._queries = queries

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise AssertionError("retrieval must not embed documents")

    def embed_query(self, text: str) -> list[float]:
        return self._queries[text]


@dataclass
class Corpus:
    sessions: object
    doc_a: str
    doc_b: str
    chunks: dict[str, UUID] = field(default_factory=dict)


def digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


@pytest_asyncio.fixture
async def corpus():
    settings = Settings(postgres_db="civil_chatbot_test")
    assert settings.postgres_db == "civil_chatbot_test"
    engine, sessions = create_resources(settings.database_url)
    suffix = uuid4().hex[:8]
    doc_a, doc_b = f"retrieval-a-{suffix}", f"retrieval-b-{suffix}"
    old_version, current_version, b_version = uuid4(), uuid4(), uuid4()
    chunks: dict[str, UUID] = {}
    try:
        async with sessions() as session, session.begin():
            session.add_all([Document(id=doc_a, label="Fixture A"), Document(id=doc_b, label="Fixture B")])
            await session.flush()
            for version_id, document_id, number in (
                (old_version, doc_a, "0"),
                (current_version, doc_a, "1"),
                (b_version, doc_b, "2"),
            ):
                session.add(
                    DocumentVersion(
                        id=version_id,
                        document_id=document_id,
                        source_title="Fixture reference",
                        jurisdiction="Test",
                        sha256=digest(f"{suffix}-{number}"),
                        snapshot_path=f"{suffix}-{number}.pdf",
                    )
                )
            await session.flush()
            await session.execute(
                update(Document).where(Document.id == doc_a).values(current_source_version_id=current_version)
            )
            await session.execute(
                update(Document).where(Document.id == doc_b).values(current_source_version_id=b_version)
            )

            inlets, detention, old_rule, hydrant, stale = (uuid4() for _ in range(5))
            session.add_all(
                [
                    Provision(
                        id=inlets,
                        source_version_id=current_version,
                        full_identifier="1.0",
                        title="Inlets",
                        sibling_order=0,
                    ),
                    Provision(
                        id=old_rule,
                        source_version_id=old_version,
                        full_identifier="1.0",
                        title="Inlets",
                        sibling_order=0,
                    ),
                    Provision(
                        id=hydrant,
                        source_version_id=b_version,
                        full_identifier="9.0",
                        title="Hydrants",
                        sibling_order=0,
                    ),
                ]
            )
            await session.flush()
            session.add(
                Provision(
                    id=detention,
                    source_version_id=current_version,
                    parent_provision_id=inlets,
                    full_identifier="1.1",
                    title="Detention Ponds",
                    sibling_order=0,
                )
            )
            await session.flush()
            # (label, version, provision, text, vector, pages, warnings, method)
            specs = [
                (
                    "inlet",
                    current_version,
                    inlets,
                    "Inlet spacing shall suit the design storm.",
                    [1, 0, 0],
                    (3, 3),
                    [],
                    None,
                ),
                (
                    "outlet",
                    current_version,
                    detention,
                    "Outlet works shall limit discharge.",
                    [0, 1, 0],
                    (4, 5),
                    ["low_confidence_ocr"],
                    "ocr",
                ),
                ("old", old_version, old_rule, "Obsolete inlet spacing rule.", [1, 0, 0], (1, 1), [], None),
                ("hydrant", b_version, hydrant, "Fire hydrant spacing is 300 feet.", [0, 0, 1], None, [], None),
            ]
            for ordinal, (label, version_id, provision_id, body, vector, pages, warnings, method) in enumerate(specs):
                chunk_id = uuid4()
                chunks[label] = chunk_id
                session.add(
                    Chunk(
                        id=chunk_id,
                        source_version_id=version_id,
                        provision_id=provision_id,
                        ordinal=ordinal,
                        raw_text=body,
                        physical_page_start=pages[0] if pages else None,
                        physical_page_end=pages[1] if pages else None,
                        extraction_method=method,
                        extraction_warnings=warnings,
                    )
                )
                await session.flush()
                session.add(
                    ChunkEmbedding(
                        chunk_id=chunk_id,
                        model=MODEL,
                        revision=REVISION,
                        dimensions=3,
                        embedding_input_sha256=digest(label),
                        embedding=[float(x) for x in vector],
                    )
                )
            # embedded only by a different revision, at exactly the vector the tests query with
            await session.flush()
            session.add(
                Provision(
                    id=stale, source_version_id=current_version, full_identifier="7.0", title=None, sibling_order=1
                )
            )
            await session.flush()
            other_revision = uuid4()
            chunks["other_revision"] = other_revision
            session.add(
                Chunk(
                    id=other_revision,
                    source_version_id=current_version,
                    provision_id=stale,
                    ordinal=0,
                    raw_text="Unrelated paragraph text.",
                )
            )
            await session.flush()
            session.add(
                ChunkEmbedding(
                    chunk_id=other_revision,
                    model=MODEL,
                    revision="rev-0",
                    dimensions=3,
                    embedding_input_sha256=digest("other"),
                    embedding=[0.0, 0.0, 1.0],
                )
            )
        yield Corpus(sessions=sessions, doc_a=doc_a, doc_b=doc_b, chunks=chunks)
    finally:
        versions = [old_version, current_version, b_version]
        async with sessions() as session, session.begin():
            version_chunks = select(Chunk.id).where(Chunk.source_version_id.in_(versions))
            await session.execute(delete(ChunkEmbedding).where(ChunkEmbedding.chunk_id.in_(version_chunks)))
            await session.execute(delete(Chunk).where(Chunk.source_version_id.in_(versions)))
            await session.execute(delete(Provision).where(Provision.source_version_id.in_(versions)))
            await session.execute(
                update(Document).where(Document.id.in_([doc_a, doc_b])).values(current_source_version_id=None)
            )
            await session.execute(delete(DocumentVersion).where(DocumentVersion.id.in_(versions)))
            await session.execute(delete(Document).where(Document.id.in_([doc_a, doc_b])))
        await engine.dispose()


def embedder(**queries: list[float]) -> FakeEmbedder:
    return FakeEmbedder({text.replace("_", " "): vector for text, vector in queries.items()})


def by_text(results: list[RetrievedChunk]) -> dict[str, RetrievedChunk]:
    return {hit.text: hit for hit in results}


@pytest.mark.asyncio
async def test_current_version_metadata_preserves_unknown_edition_and_effective_date(corpus):
    assert await source_version_metadata(corpus.sessions, [corpus.doc_a]) == [(corpus.doc_a, None, None)]


@pytest.mark.asyncio
async def test_only_the_current_source_version_is_searched(corpus):
    results = await hybrid_search(
        corpus.sessions, embedder(inlet_spacing=[1, 0, 0]), "inlet spacing", document_ids=[corpus.doc_a]
    )
    texts = [hit.text for hit in results]
    assert "Inlet spacing shall suit the design storm." in texts
    assert "Obsolete inlet spacing rule." not in texts
    assert {hit.document_id for hit in results} == {corpus.doc_a}


@pytest.mark.asyncio
async def test_document_ids_filter_restricts_both_rankings(corpus):
    fake = embedder(spacing=[1, 0, 1])
    only_b = await hybrid_search(corpus.sessions, fake, "spacing", document_ids=[corpus.doc_b], k=50)
    assert [hit.document_id for hit in only_b] == [corpus.doc_b]
    only_a = await hybrid_search(corpus.sessions, fake, "spacing", document_ids=[corpus.doc_a], k=50)
    assert only_a and {hit.document_id for hit in only_a} == {corpus.doc_a}
    both = await hybrid_search(corpus.sessions, fake, "spacing", document_ids=[corpus.doc_a, corpus.doc_b], k=50)
    assert {hit.document_id for hit in both} == {corpus.doc_a, corpus.doc_b}
    assert await hybrid_search(corpus.sessions, fake, "spacing", document_ids=[], k=50) == []


@pytest.mark.asyncio
async def test_only_embeddings_from_the_embedders_model_and_revision_are_used(corpus):
    results = await hybrid_search(
        corpus.sessions, embedder(zzqq=[0, 0, 1]), "zzqq", document_ids=[corpus.doc_a, corpus.doc_b], k=50
    )
    texts = [hit.text for hit in results]
    assert "Fire hydrant spacing is 300 feet." in texts  # embedded with the right revision, nearest to the query
    assert "Unrelated paragraph text." not in texts  # nearest vector, but made by another revision


def ranked_legs(hit) -> tuple[str, ...]:
    """The legs that actually ranked this hit; `found_by` must name exactly these."""
    return tuple(name for name, rank in ((FULL_TEXT, hit.fts_rank), (VECTOR, hit.vector_rank)) if rank is not None)


@pytest.mark.asyncio
async def test_fulltext_leg_is_off_by_default(corpus):
    results = await hybrid_search(
        corpus.sessions, embedder(detention=[1, 0, 0]), "detention", document_ids=[corpus.doc_a]
    )
    assert results
    assert all(hit.fts_rank is None and hit.found_by == ranked_legs(hit) == (VECTOR,) for hit in results)


@pytest.mark.asyncio
async def test_expand_neighbors_never_crosses_provisions_and_is_off_by_default(corpus):
    fake = embedder(outlet_works=[0, 1, 0])
    plain = await hybrid_search(corpus.sessions, fake, "outlet works", document_ids=[corpus.doc_a], k=2)
    expanded = await hybrid_search(
        corpus.sessions, fake, "outlet works", document_ids=[corpus.doc_a], k=2, expand_neighbors=True
    )
    assert all(hit.neighbor_of is None for hit in plain)
    assert expanded[: len(plain)] == plain
    # each fixture provision has one chunk, so no adjacent chunk of the same provision exists
    assert expanded == plain


def test_split_duplicate_strips_only_the_ingest_marker():
    assert split_duplicate("2.3.0 (duplicate 2)") == ("2.3.0", True)
    assert split_duplicate("2.3.0") == ("2.3.0", False)
    assert split_duplicate("APPENDIX A (duplicate 1)") == ("APPENDIX A", True)


@pytest.mark.asyncio
async def test_query_matching_no_fulltext_still_returns_vector_hits(corpus):
    for query in ("zzqq", "the of"):  # no lexeme matches / only stop words (empty tsquery)
        results = await hybrid_search(
            corpus.sessions, embedder(**{query: [0, 1, 0]}), query, document_ids=[corpus.doc_a]
        )
        assert results, query
        assert results[0].text == "Outlet works shall limit discharge."
        assert all(hit.found_by == (VECTOR,) for hit in results)


@pytest.mark.asyncio
async def test_heading_word_only_in_the_provision_title_finds_the_provision(corpus, fts_enabled):
    # "detention" appears in no chunk text; the query vector is nearest the inlet chunk
    results = await hybrid_search(
        corpus.sessions, embedder(detention=[1, 0, 0]), "detention", document_ids=[corpus.doc_a]
    )
    hit = next(hit for hit in results if hit.provision_identifier == "1.1")
    assert hit.fts_rank is not None and FULL_TEXT in hit.found_by
    assert "detention" not in hit.text.lower()


@pytest.mark.asyncio
async def test_results_carry_pages_breadcrumb_and_extraction_details(corpus, fts_enabled):
    results = await hybrid_search(
        corpus.sessions, embedder(outlet_works=[0, 1, 0]), "outlet works", document_ids=[corpus.doc_a]
    )
    hit = by_text(results)["Outlet works shall limit discharge."]
    assert hit == results[0]
    assert hit.document_id == corpus.doc_a
    assert (hit.provision_identifier, hit.provision_title) == ("1.1", "Detention Ponds")
    assert (hit.page_start, hit.page_end) == (4, 5)
    assert hit.breadcrumb == "1.0 Inlets > 1.1 Detention Ponds"
    assert hit.extraction_warnings == ("low_confidence_ocr",)
    assert hit.extraction_method == "ocr"
    assert hit.found_by == ranked_legs(hit) == (FULL_TEXT, VECTOR)
    assert hit.score > by_text(results)["Inlet spacing shall suit the design storm."].score

    hydrant = by_text(
        await hybrid_search(corpus.sessions, embedder(hydrant=[0, 0, 1]), "hydrant", document_ids=[corpus.doc_b])
    )["Fire hydrant spacing is 300 feet."]
    assert (hydrant.page_start, hydrant.page_end) == (None, None)
    assert hydrant.breadcrumb == "9.0 Hydrants"
    assert hydrant.extraction_warnings == ()


# --- section references and direct lookup ---


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Sec 2.3.0", "2.3.0"),
        ("Section 2.3.0", "2.3.0"),
        ("sec. 2.3", "2.3"),
        ("§2.3.0", "2.3.0"),
        ("§ 2.3.0.", "2.3.0"),
        ("2.3.0", "2.3.0"),
        ("  Sec   1.2.4  ", "1.2.4"),
        ("Appendix B", "APPENDIX B"),
        ("appendix b", "APPENDIX B"),
        ("Section 4", "SECTION 4"),
    ],
)
def test_normalize_section_ref_accepts_section_forms(text, expected):
    assert normalize_section_ref(text) == expected


@pytest.mark.parametrize("text", ["10.5 miles", "42", "", "Sec", "2.3.0 and more", "miles 2.3", "Appendix", "4"])
def test_normalize_section_ref_rejects_non_sections(text):
    assert normalize_section_ref(text) is None


def test_find_section_ref_needs_an_explicit_marker():
    assert find_section_ref("What does Sec 2.3.0 talk about in the Drainage Manual?") == "2.3.0"
    assert find_section_ref("What does §4.2 require?") == "4.2"
    assert find_section_ref("See Appendix B for the tables") == "APPENDIX B"
    assert find_section_ref("The pipe is 10.5 miles long") is None
    assert find_section_ref("the 2.3.0 value") is None
    assert find_section_ref("in this section of the code") is None


@pytest.mark.asyncio
async def test_get_provision_chunks_returns_the_current_version_with_child_titles(corpus):
    results = await get_provision_chunks(corpus.sessions, "Sec 1.0", document_ids=[corpus.doc_a])
    assert [hit.text for hit in results] == ["Inlet spacing shall suit the design storm."]  # not the old version
    hit = results[0]
    assert hit.document_id == corpus.doc_a
    assert (hit.found_by, hit.score) == ((SECTION_LOOKUP,), 1.0)
    assert hit.children == ("1.1 Detention Ponds",)
    assert (hit.page_start, hit.page_end) == (3, 3)
    leaf = await get_provision_chunks(corpus.sessions, "1.1", document_ids=[corpus.doc_a])
    assert [hit.text for hit in leaf] == ["Outlet works shall limit discharge."]
    assert leaf[0].children == () and leaf[0].breadcrumb == "1.0 Inlets > 1.1 Detention Ponds"


@pytest.mark.asyncio
async def test_get_provision_chunks_respects_document_scope_and_finds_nothing_cleanly(corpus):
    both = await get_provision_chunks(corpus.sessions, "9.0", document_ids=[corpus.doc_a, corpus.doc_b])
    assert [hit.document_id for hit in both] == [corpus.doc_b]
    assert await get_provision_chunks(corpus.sessions, "9.0", document_ids=[corpus.doc_a]) == []
    assert await get_provision_chunks(corpus.sessions, "9.0", document_ids=[]) == []
    assert await get_provision_chunks(corpus.sessions, "5.5", document_ids=[corpus.doc_a]) == []
    assert await get_provision_chunks(corpus.sessions, "10.5 miles", document_ids=[corpus.doc_a]) == []
