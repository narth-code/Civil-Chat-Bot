"""Retrieval-storage boundaries against migrated civil_chatbot_test only."""

from uuid import uuid4

import pytest
import pytest_asyncio
from sqlalchemy import insert, select, update
from sqlalchemy.exc import IntegrityError

from app.core.config import Settings
from app.db.models import (
    Chunk,
    ChunkEmbedding,
    Conversation,
    Document,
    DocumentVersion,
    Message,
    Provision,
    RoutingPolicy,
    RoutingRun,
)
from app.db.session import create_resources

pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture
async def storage():
    settings = Settings(postgres_db="civil_chatbot_test")
    assert settings.postgres_db == "civil_chatbot_test"
    engine, _ = create_resources(settings.database_url)
    async with engine.connect() as connection:
        transaction = await connection.begin()
        try:
            yield connection
        finally:
            await transaction.rollback()
    await engine.dispose()


async def versions(connection):
    doc = "fixture-" + str(uuid4())
    await connection.execute(insert(Document).values(id=doc, label="Isolated fixture"))
    ids = [uuid4(), uuid4()]
    for index, version_id in enumerate(ids):
        await connection.execute(
            insert(DocumentVersion).values(
                id=version_id,
                document_id=doc,
                source_title="Fixture reference",
                jurisdiction="Unknown",
                sha256=str(index) * 64,
                snapshot_path=f"{index}.pdf",
            )
        )
    return doc, ids


async def rejection(connection, statement):
    savepoint = await connection.begin_nested()
    try:
        with pytest.raises(IntegrityError):
            await connection.execute(statement)
    finally:
        await savepoint.rollback()


async def test_nested_provisions_preserve_exact_quotes_and_page_provenance(storage):
    _, (version, _) = await versions(storage)
    parent, child, chunk = uuid4(), uuid4(), uuid4()
    await storage.execute(
        insert(Provision).values(
            id=parent, source_version_id=version, full_identifier="2.3.0", title=None, sibling_order=0
        )
    )
    await storage.execute(
        insert(Provision).values(
            id=child,
            source_version_id=version,
            parent_provision_id=parent,
            full_identifier="2.3.0.B.2",
            title=None,
            sibling_order=1,
        )
    )
    quote = "  Exact source spacing\nQuoted provision; no normalization.  "
    await storage.execute(
        insert(Chunk).values(
            id=chunk,
            source_version_id=version,
            provision_id=child,
            ordinal=0,
            raw_text=quote,
            physical_page_start=4,
            physical_page_end=5,
            printed_page_label="iii–iv",
            table_markdown="| A | B |",
            extraction_confidence=0.9,
            quality_flags=["table_review"],
        )
    )
    row = (await storage.execute(select(Chunk.__table__).where(Chunk.id == chunk))).mappings().one()
    assert row["raw_text"] == quote
    assert (row["physical_page_start"], row["physical_page_end"], row["printed_page_label"]) == (4, 5, "iii–iv")
    assert row["quality_flags"] == ["table_review"]
    assert (
        await storage.execute(select(Provision.parent_provision_id).where(Provision.id == child))
    ).scalar_one() == parent


async def test_cross_version_parents_chunks_and_current_document_are_rejected(storage):
    doc, (v1, v2) = await versions(storage)
    parent = uuid4()
    await storage.execute(
        insert(Provision).values(id=parent, source_version_id=v1, full_identifier="A", sibling_order=0)
    )
    await rejection(
        storage,
        insert(Provision).values(
            id=uuid4(), source_version_id=v2, parent_provision_id=parent, full_identifier="A.1", sibling_order=0
        ),
    )
    await rejection(
        storage,
        insert(Chunk).values(
            id=uuid4(), source_version_id=v2, provision_id=parent, ordinal=0, raw_text="wrong version"
        ),
    )
    other_doc, (other_version, _) = await versions(storage)
    await rejection(storage, update(Document).where(Document.id == doc).values(current_source_version_id=other_version))
    assert other_doc != doc
    await rejection(storage, update(Provision).where(Provision.id == parent).values(parent_provision_id=parent))


async def test_page_bounds_and_extraction_quality_reject_unknown_half_or_invalid_values(storage):
    _, (version, _) = await versions(storage)
    provision = uuid4()
    await storage.execute(
        insert(Provision).values(id=provision, source_version_id=version, full_identifier="B", sibling_order=0)
    )
    base = {"source_version_id": version, "provision_id": provision, "ordinal": 0, "raw_text": "quote"}
    for extra in (
        {"physical_page_start": 1, "physical_page_end": None},
        {"physical_page_start": None, "physical_page_end": 1},
        {"physical_page_start": 0, "physical_page_end": 1},
        {"physical_page_start": 4, "physical_page_end": 3},
        {"extraction_confidence": float("nan")},
        {"extraction_confidence": float("inf")},
        {"extraction_confidence": -0.1},
        {"quality_flags": ["review", 3]},
    ):
        await rejection(storage, insert(Chunk).values(id=uuid4(), **base, **extra))
    chunk = uuid4()
    await storage.execute(insert(Chunk).values(id=chunk, **base))
    row = (await storage.execute(select(Chunk.__table__).where(Chunk.id == chunk))).mappings().one()
    assert row["physical_page_start"] is None and row["physical_page_end"] is None
    assert row["extraction_confidence"] is None


async def test_embedding_dimensions_and_same_input_deduplication_are_enforced(storage):
    _, (version, _) = await versions(storage)
    provision, chunk = uuid4(), uuid4()
    await storage.execute(
        insert(Provision).values(id=provision, source_version_id=version, full_identifier="C", sibling_order=0)
    )
    await storage.execute(
        insert(Chunk).values(
            id=chunk, source_version_id=version, provision_id=provision, ordinal=0, raw_text="literal input"
        )
    )
    base = {
        "chunk_id": chunk,
        "model": "test-only-model",
        "revision": "fixture",
        "embedding_input_sha256": "a" * 64,
        "embedding": [0.1, 0.2, 0.3],
    }
    await rejection(storage, insert(ChunkEmbedding).values(id=uuid4(), dimensions=2, **base))
    embedding = uuid4()
    await storage.execute(insert(ChunkEmbedding).values(id=embedding, dimensions=3, **base))
    await rejection(storage, insert(ChunkEmbedding).values(id=uuid4(), dimensions=3, **base))
    await rejection(
        storage, insert(ChunkEmbedding).values(id=uuid4(), dimensions=0, **{**base, "embedding_input_sha256": "b" * 64})
    )
    assert (
        await storage.execute(select(ChunkEmbedding.dimensions).where(ChunkEmbedding.id == embedding))
    ).scalar_one() == 3


async def test_answer_message_link_and_outcome_are_database_enforced(storage):
    conversation_id, message_id, run_id = uuid4(), uuid4(), uuid4()
    await storage.execute(insert(Conversation).values(id=conversation_id, title="Constraint test"))
    policy = "constraint-" + str(uuid4())
    await storage.execute(insert(RoutingPolicy).values(version=policy, system_prompt="test", response_schema={}))
    await storage.execute(
        insert(Message).values(
            id=message_id,
            conversation_id=conversation_id,
            sequence=0,
            role="user",
            origin="user",
            content="Question",
            selected_manual_ids=[],
        )
    )
    await storage.execute(
        insert(RoutingRun).values(
            id=run_id,
            conversation_id=conversation_id,
            user_message_id=message_id,
            status="pending",
            policy_version=policy,
            model="fixture",
            model_options={},
            model_calls=0,
            citations=[],
            trace=[],
        )
    )
    await rejection(storage, update(RoutingRun).where(RoutingRun.id == run_id).values(outcome="made_up"))
    await rejection(
        storage,
        insert(Message).values(
            id=uuid4(),
            conversation_id=conversation_id,
            sequence=1,
            role="assistant",
            origin="assistant_answer",
            content="Unlinked",
            selected_manual_ids=[],
        ),
    )
    await storage.execute(
        insert(Message).values(
            id=uuid4(),
            conversation_id=conversation_id,
            sequence=1,
            role="assistant",
            origin="assistant_answer",
            content="Answer",
            selected_manual_ids=[],
            routing_run_id=run_id,
        )
    )
    await rejection(
        storage,
        insert(Message).values(
            id=uuid4(),
            conversation_id=conversation_id,
            sequence=2,
            role="assistant",
            origin="router_status",
            content="Duplicate",
            selected_manual_ids=[],
            routing_run_id=run_id,
        ),
    )
    await rejection(
        storage,
        insert(Message).values(
            id=uuid4(),
            conversation_id=conversation_id,
            sequence=2,
            role="user",
            origin="assistant_answer",
            content="Wrong role",
            selected_manual_ids=[],
            routing_run_id=run_id,
        ),
    )
