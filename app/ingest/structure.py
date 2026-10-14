"""Content-aware indexing: turn an ordered stream of extracted blocks into Provisions and chunks.

Pure Python (no Docling, no database) so it can be tested directly. A Provision exists only where
the source prints a numeric identifier ("1.2.2.1 - Title", "SECTION 1 - TITLE", "APPENDIX A");
nothing is invented. Hierarchy comes from the identifiers, not from font or layout guesses.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import Iterable

PREAMBLE_IDENTIFIER = "(unnumbered text)"
DEFAULT_MAX_CHARS = 1500

_HEADING = re.compile(
    r"^(?:(?P<section>SECTION)\s+(?P<sec_no>\d+)|(?P<appendix>APPENDIX)\s+(?P<app_no>[A-Z0-9]+)|(?P<numeric>\d+(?:\.\d+){1,5}))"
    r"\s*[-\u2013\u2014:.]?\s*(?P<title>.*)$",
    re.IGNORECASE,
)
_SENTENCE_END = re.compile(r"(?<=[.;:])\s+")


@dataclass(frozen=True)
class Block:
    kind: str  # heading | paragraph | list_item | table
    text: str
    page: int | None = None
    flags: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()  # "<code>: <detail> (page N)" strings measured for this block's page
    method: str | None = None  # label of what produced the text: the PDF text layer or OCR
    confidence: float | None = None  # mean OCR recognition confidence of the block's page; None without OCR


@dataclass
class DraftProvision:
    key: tuple
    full_identifier: str
    title: str | None
    parent_index: int | None
    sibling_order: int
    duplicate: bool = False


@dataclass
class DraftChunk:
    provision_index: int
    ordinal: int
    raw_text: str
    page_start: int | None
    page_end: int | None
    table_markdown: str | None
    flags: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    extraction_method: str | None = None
    extraction_confidence: float | None = None


@dataclass
class Structure:
    provisions: list[DraftProvision]
    chunks: list[DraftChunk]


def parse_heading(text: str) -> tuple[tuple, str, str | None] | None:
    """Return (hierarchy key, exact printed identifier, title) when text starts with a provision identifier."""
    line = " ".join(text.split())
    match = _HEADING.match(line)
    if not match:
        return None
    title = match.group("title").strip(" .") or None
    if match.group("section"):
        return (int(match.group("sec_no")),), f"SECTION {match.group('sec_no')}", title
    if match.group("appendix"):
        label = match.group("app_no").upper()
        return (f"APPENDIX {label}",), f"APPENDIX {label}", title
    printed = line[: match.start("title")].strip(" -\u2013\u2014:.")
    # a numeric identifier with prose that is not a heading ("10.5 miles ...") has no capitalised title
    if title is None or not title[0].isupper():
        return None
    return tuple(int(part) for part in printed.split(".")), printed, title


def find_parent(key: tuple, seen: dict[tuple, int]) -> int | None:
    """Nearest earlier Provision that structurally contains `key` (1.2.1 sits under 1.2 or 1.2.0)."""
    if len(key) < 2 or not all(isinstance(part, int) for part in key):
        return None
    base = key[:-1] if key[-1] != 0 else key[:-2]
    while base:
        for candidate in (base, base + (0,)):
            if candidate != key and candidate in seen:
                return seen[candidate]
        base = base[:-1]
    return None


_EXPLICIT_HEADING = re.compile(
    r"^(?:SECTION\s+\d+|APPENDIX\s+[A-Z0-9]+|\d+(?:\.\d+){1,5})\s*[-\u2013\u2014]\s+\S", re.IGNORECASE
)


def is_heading(block: Block) -> bool:
    if parse_heading(block.text) is None:
        return False
    if block.kind == "heading":
        return True
    # A layout model sometimes labels a printed heading as body text. Accept that only for the
    # explicit "number - Title" form on a short line, so numbered clauses in prose stay body text.
    line = " ".join(block.text.split())
    return block.kind == "paragraph" and len(line) <= 160 and bool(_EXPLICIT_HEADING.match(line))


def split_text(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    pieces: list[str] = []
    current = ""
    for sentence in _SENTENCE_END.split(text):
        if current and len(current) + 1 + len(sentence) > max_chars:
            pieces.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        pieces.append(current)
    return pieces


def build_structure(blocks: Iterable[Block], *, max_chars: int = DEFAULT_MAX_CHARS) -> Structure:
    provisions: list[DraftProvision] = []
    chunks: list[DraftChunk] = []
    seen: dict[tuple, int] = {}
    sibling_counts: dict[int | None, int] = {}
    ordinals: dict[int, int] = {}
    current: int | None = None
    pending: list[Block] = []

    def add_provision(key: tuple, identifier: str, title: str | None) -> int:
        parent = find_parent(key, seen)
        duplicate = key in seen
        unique_identifier = identifier
        if duplicate:
            existing = sum(
                1
                for p in provisions
                if p.full_identifier == identifier or p.full_identifier.startswith(f"{identifier} (duplicate")
            )
            unique_identifier = f"{identifier} (duplicate {existing})"
        order = sibling_counts.get(parent, 0)
        sibling_counts[parent] = order + 1
        provisions.append(DraftProvision(key, unique_identifier, title, parent, order, duplicate))
        index = len(provisions) - 1
        if not duplicate:
            seen[key] = index
        return index

    def emit(provision_index: int, text: str, page_start, page_end, table_md, source: list[Block], flags) -> None:
        ordinal = ordinals.get(provision_index, 0)
        ordinals[provision_index] = ordinal + 1
        merged = list(dict.fromkeys(flags))
        if provisions[provision_index].duplicate:
            merged.append("duplicate_identifier")
        methods = list(dict.fromkeys(b.method for b in source if b.method))
        page_confidence = {b.page: b.confidence for b in source if b.confidence is not None}
        chunks.append(
            DraftChunk(
                provision_index,
                ordinal,
                text,
                page_start,
                page_end,
                table_md,
                merged,
                list(dict.fromkeys(w for b in source for w in b.warnings)),
                " + ".join(methods) or None,
                sum(page_confidence.values()) / len(page_confidence) if page_confidence else None,
            )
        )

    def flush() -> None:
        nonlocal pending
        if not pending or current is None:
            pending = []
            return
        group: list[Block] = []
        size = 0

        def close_group() -> None:
            nonlocal group, size
            if group:
                pages = [b.page for b in group if b.page is not None]
                flags = [flag for b in group for flag in b.flags]
                emit(
                    current,
                    "\n\n".join(b.text for b in group),
                    min(pages, default=None),
                    max(pages, default=None),
                    None,
                    group,
                    flags,
                )
            group, size = [], 0

        for block in pending:
            if block.kind == "table":
                close_group()
                emit(current, block.text, block.page, block.page, block.text, [block], [*block.flags, "table"])
                continue
            for piece in split_text(block.text, max_chars):
                piece_block = replace(block, text=piece)
                if group and size + len(piece) > max_chars:
                    close_group()
                group.append(piece_block)
                size += len(piece) + 2
        close_group()
        pending = []

    for block in blocks:
        text = block.text.strip()
        if not text:
            continue
        block = replace(block, text=text)
        if is_heading(block):
            flush()
            key, identifier, title = parse_heading(block.text)  # type: ignore[misc]
            current = add_provision(key, identifier, title)
            continue
        if current is None:
            current = add_provision((PREAMBLE_IDENTIFIER,), PREAMBLE_IDENTIFIER, None)
        pending.append(block)
    flush()
    return Structure(provisions, chunks)


def provision_path(provisions: list[DraftProvision] | list[tuple[str, str | None, int | None]], index: int) -> str:
    """Breadcrumb used to give each chunk its context when it is embedded."""
    parts: list[str] = []
    cursor: int | None = index
    while cursor is not None:
        item = provisions[cursor]
        identifier, title, parent = (
            (item.full_identifier, item.title, item.parent_index) if isinstance(item, DraftProvision) else item
        )
        parts.append(f"{identifier} {title}".strip() if title else identifier)
        cursor = parent
    return " > ".join(reversed(parts))
