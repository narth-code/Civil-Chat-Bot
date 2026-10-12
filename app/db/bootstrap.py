"""Register immutable manual snapshots and the routing policy after Alembic migrations."""

from __future__ import annotations

import asyncio
from hashlib import sha256
from pathlib import Path


from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from app.core.config import get_settings
from app.core.manuals import load_manifest
from app.db.models import Document, DocumentVersion, RoutingPolicy
from app.db.session import create_resources
from app.db.snapshots import capture_snapshot
from app.rag import agent_state


AGENT_SOURCE = Path(__file__).resolve().parents[1] / "rag" / "agent.py"
STRUCTURED_SCHEMAS = (
    agent_state.EvidenceGrade,
    agent_state.ContextExtraction,
    agent_state.RouteReason,
    agent_state.AnswerWithQuotes,
    agent_state.AnswerReview,
)


async def register_agent_policy(session):
    """Record the exact agent code and structured schemas once; never mutate a prior policy."""
    source = AGENT_SOURCE.read_text(encoding="utf-8")
    version = "agent-" + sha256(source.encode("utf-8")).hexdigest()[:12]
    schema = {"schemas": {model.__name__: model.model_json_schema() for model in STRUCTURED_SCHEMAS}}
    await session.execute(
        insert(RoutingPolicy)
        .values(version=version, system_prompt=source, response_schema=schema)
        .on_conflict_do_nothing(index_elements=[RoutingPolicy.version])
    )
    policy = await session.get(RoutingPolicy, version)
    if policy.system_prompt != source or policy.response_schema != schema:
        raise RuntimeError(f"agent policy version {version!r} already has different content")
    return policy


async def bootstrap() -> None:
    manuals = load_manifest()
    captured = [(manual, *capture_snapshot(manual.filename)) for manual in manuals]
    engine, session_factory = create_resources(get_settings().database_url)
    try:
        async with session_factory() as session:
            async with session.begin():
                for manual, sha256, snapshot_path in captured:
                    document = await session.get(Document, manual.id)
                    if document is None:
                        document = Document(id=manual.id, label=manual.label)
                        session.add(document)
                    else:
                        document.label = manual.label

                    version = await session.scalar(
                        select(DocumentVersion).where(
                            DocumentVersion.document_id == manual.id,
                            DocumentVersion.sha256 == sha256,
                        )
                    )
                    if version is None:
                        version = DocumentVersion(
                            document_id=manual.id,
                            source_title=manual.title,
                            jurisdiction=manual.jurisdiction,
                            sha256=sha256,
                            snapshot_path=snapshot_path,
                            edition=None,
                            effective_date=None,
                            applicability_note=None,
                        )
                        session.add(version)
                        await session.flush()
                    elif version.snapshot_path != snapshot_path:
                        raise RuntimeError(f"registered version snapshot mismatch for {manual.id}")
                    document.current_source_version_id = version.id

                await register_agent_policy(session)
    finally:
        await engine.dispose()


def main() -> None:
    asyncio.run(bootstrap())


if __name__ == "__main__":
    main()
