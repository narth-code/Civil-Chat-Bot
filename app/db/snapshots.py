"""Capture manual PDFs as immutable, content-addressed snapshots under the snapshot root."""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path

from app.core.manuals import DOCUMENT_ROOT, SNAPSHOT_ROOT

_CHUNK_SIZE = 1024 * 1024


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(_CHUNK_SIZE):
            digest.update(block)
    return digest.hexdigest()


def capture_snapshot(filename: str) -> tuple[str, str]:
    """Snapshot docs/<filename>; return (sha256, snapshot path relative to the snapshot root).

    New files are published atomically; an existing snapshot is reused only if its bytes match.
    """
    source = DOCUMENT_ROOT / filename
    if not source.is_file():
        raise RuntimeError(f"manual PDF is missing: {source.name}")
    digest = _hash_file(source)
    SNAPSHOT_ROOT.mkdir(parents=True, exist_ok=True)
    destination = SNAPSHOT_ROOT / f"{digest}.pdf"
    relative_path = destination.relative_to(SNAPSHOT_ROOT).as_posix()
    if destination.exists():
        if _hash_file(destination) != digest:
            raise RuntimeError(f"existing snapshot digest mismatch: {destination.name}")
        return digest, relative_path

    descriptor, temporary_name = tempfile.mkstemp(prefix=".snapshot-", suffix=".tmp", dir=SNAPSHOT_ROOT)
    try:
        copied = hashlib.sha256()
        with os.fdopen(descriptor, "wb") as target, source.open("rb") as input_file:
            while block := input_file.read(_CHUNK_SIZE):
                copied.update(block)
                target.write(block)
            target.flush()
            os.fsync(target.fileno())
        if copied.hexdigest() != digest:
            raise RuntimeError(f"manual PDF changed while being captured: {source.name}")
        try:
            os.link(temporary_name, destination)
        except FileExistsError:
            pass
        else:
            os.unlink(temporary_name)
            return digest, relative_path
        if _hash_file(destination) != digest:
            raise RuntimeError(f"concurrent snapshot digest mismatch: {destination.name}")
        return digest, relative_path
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
