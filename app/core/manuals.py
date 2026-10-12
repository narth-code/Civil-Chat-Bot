"""The manual catalog: what the shared manifest says, and where registered snapshots live.

Backend modules read the manifest and resolve snapshot paths only through this module.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from app.core.config import BASE_DIR

MANIFEST_PATH = BASE_DIR / "data" / "manuals.json"
DOCUMENT_ROOT = BASE_DIR / "docs"
SNAPSHOT_ROOT = (BASE_DIR / "data" / "source_versions").resolve()

_REQUIRED_KEYS = {"id", "label", "title", "filename", "jurisdiction"}


@dataclass(frozen=True)
class ManualEntry:
    id: str
    label: str
    title: str
    filename: str
    jurisdiction: str


def load_manifest() -> list[ManualEntry]:
    """Read and validate data/manuals.json. Raises RuntimeError on any malformed content."""
    try:
        raw = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("manual manifest is unavailable") from error
    if not isinstance(raw, list) or not raw:
        raise RuntimeError("manual manifest must be a non-empty JSON array")
    entries: list[ManualEntry] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, dict) or set(item) != _REQUIRED_KEYS:
            raise RuntimeError("manual manifest entries have an unexpected shape")
        if not all(isinstance(item[key], str) and item[key] for key in _REQUIRED_KEYS):
            raise RuntimeError("manual manifest values must be non-empty strings")
        if item["id"] in seen:
            raise RuntimeError(f"duplicate manual id: {item['id']}")
        seen.add(item["id"])
        entries.append(ManualEntry(**item))
    return entries


@lru_cache(maxsize=1)
def manual_ids() -> tuple[str, ...]:
    """Ids of every manual in the manifest, in manifest order."""
    return tuple(entry.id for entry in load_manifest())


def manual_url(manual_id: str) -> str:
    return f"/api/manuals/{manual_id}/pdf"


def snapshot_file(relative_path: str) -> Path:
    """Resolve a registered snapshot path inside the snapshot root; refuse anything that escapes it."""
    path = (SNAPSHOT_ROOT / relative_path).resolve()
    if SNAPSHOT_ROOT not in path.parents:
        raise ValueError("snapshot path escapes the snapshot root")
    return path
