"""Manual catalog and inline PDF routes backed by registered immutable snapshots."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.core.manuals import ManualEntry, load_manifest, manual_url, snapshot_file
from app.db.models import Document
from app.schemas import ManualCatalog, ManualView

router = APIRouter(prefix="/api/manuals", tags=["manuals"])


def _manifest() -> list[ManualEntry]:
    try:
        return load_manifest()
    except RuntimeError as exc:
        raise HTTPException(503, "Manual manifest is unavailable.") from exc


@router.get("", response_model=ManualCatalog)
async def list_manuals(request: Request) -> ManualCatalog:
    entries = _manifest()
    async with request.app.state.session_factory() as session:
        records = (
            await session.scalars(
                select(Document)
                .where(Document.id.in_([item.id for item in entries]))
                .options(selectinload(Document.current_source_version))
            )
        ).all()
    by_id = {record.id: record for record in records}
    result = []
    for item in entries:
        record = by_id.get(item.id)
        version = record.current_source_version if record is not None else None
        if version is None:
            continue
        result.append(
            ManualView(
                id=item.id,
                label=item.label,
                title=item.title,
                filename=item.filename,
                url=manual_url(item.id),
                current_source_version_id=version.id,
                edition=version.edition,
                jurisdiction=version.jurisdiction,
            )
        )
    return ManualCatalog(manuals=result)


@router.get("/{document_id}/pdf")
async def manual_pdf(document_id: str, request: Request) -> FileResponse:
    item = next((entry for entry in _manifest() if entry.id == document_id), None)
    if item is None:
        raise HTTPException(404, "Manual not found.")
    async with request.app.state.session_factory() as session:
        record = await session.scalar(
            select(Document).where(Document.id == document_id).options(selectinload(Document.current_source_version))
        )
        version = record.current_source_version if record is not None else None
    if version is None:
        raise HTTPException(404, "Manual not found.")
    try:
        path = snapshot_file(version.snapshot_path)
    except ValueError:
        raise HTTPException(404, "Manual PDF snapshot not found.") from None
    if not path.is_file():
        raise HTTPException(404, "Manual PDF snapshot not found.")
    return FileResponse(path, media_type="application/pdf", filename=item.filename, content_disposition_type="inline")
