"""Ingestion command: convert a registered manual's PDF, build its structure, load it, embed it.

    python -m app.ingest.run --document-id fire [--document-id drainage ...] [--ocr full-page|auto]
        [--pages 1-5] [--embedder auto|harrier|nomic] [--rebuild] [--reconvert] [--batch-pages 20]
        [--stages convert,load,embed] [--device cuda]

Documents run one after another; a failure is logged and recorded in that document's status.json, the remaining
documents still run, and the exit code is non-zero if any failed. --ocr full-page OCRs every page (scans, print-to-PDF);
auto keeps the PDF's own text layer on pages that have one and OCRs the rest (born-digital manuals).

Artifacts live in data/ingest/<sha256>/ (full run) or data/ingest/<sha256>/partial-<a>-<b>/ (--pages),
with a status.json rewritten atomically after every step, a quality.json summary of per-page extraction warnings and an
ingest.log next to it. Batches converted before warnings existed are refused until --reconvert deletes them.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from app.core.config import BASE_DIR, get_settings
from app.core.manuals import snapshot_file
from app.db.session import create_resources
from app.ingest import convert, load
from app.ingest.embedding import load_embedder
from app.ingest.structure import build_structure

log = logging.getLogger("ingest.run")

LOG_FORMAT = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
INGEST_ROOT = BASE_DIR / "data" / "ingest"
TEST_DATABASE = "civil_chatbot_test"
STAGES = ("convert", "load", "embed")


def parse_pages(value: str) -> tuple[int, int]:
    first, _, last = value.partition("-")
    first_page, last_page = int(first), int(last or first)
    if first_page < 1 or last_page < first_page:
        raise argparse.ArgumentTypeError("--pages must look like 1-5 with 1 <= first <= last")
    return first_page, last_page


def parse_stages(value: str) -> list[str]:
    stages = [part.strip() for part in value.split(",") if part.strip()]
    unknown = [stage for stage in stages if stage not in STAGES]
    if unknown or not stages:
        raise argparse.ArgumentTypeError(f"--stages must be a comma list of {', '.join(STAGES)}")
    return [stage for stage in STAGES if stage in stages]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app.ingest.run", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--document-id",
        action="append",
        required=True,
        help="registered document id; repeat to ingest several documents sequentially",
    )
    parser.add_argument(
        "--ocr",
        choices=convert.OCR_POLICIES,
        default="auto",
        help="full-page: OCR every page; auto: use the PDF text layer where a page has one (>= %s chars), OCR the rest"
        % convert.TEXT_LAYER_MIN_CHARS,
    )
    parser.add_argument("--pages", type=parse_pages, help="inclusive physical page range, e.g. 1-5 (partial run)")
    parser.add_argument(
        "--allow-partial", action="store_true", help="allow --pages against a database other than civil_chatbot_test"
    )
    parser.add_argument("--embedder", choices=("auto", "harrier", "nomic"), default="auto")
    parser.add_argument(
        "--rebuild", action="store_true", help="replace this version's existing provisions, chunks and embeddings"
    )
    parser.add_argument(
        "--reconvert",
        action="store_true",
        help="delete this document's cached converted batches (and page scan) first, then convert again",
    )
    parser.add_argument("--batch-pages", type=int, default=20)
    parser.add_argument("--stages", type=parse_stages, default=list(STAGES))
    parser.add_argument("--device", default="cuda")
    return parser


class Status:
    """status.json, rewritten atomically on every change."""

    def __init__(self, path: Path, initial: dict) -> None:
        self.path = path
        self.data = {**initial, "started_at": _now(), "errors": []}
        self.write()

    def update(self, **fields) -> None:
        self.data.update(fields, updated_at=_now())
        self.write()

    def write(self) -> None:
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.data, indent=2, default=str), encoding="utf-8")
        temporary.replace(self.path)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def configure_console() -> None:
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for noisy in ("httpx", "httpcore", "huggingface_hub", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(LOG_FORMAT)
    root.addHandler(handler)


def attach_log_file(log_file: Path) -> logging.Handler:
    handler = logging.FileHandler(log_file, encoding="utf-8")
    handler.setFormatter(LOG_FORMAT)
    logging.getLogger().addHandler(handler)
    return handler


async def run(args: argparse.Namespace) -> int:
    settings = get_settings()
    if args.pages and settings.postgres_db != TEST_DATABASE and not args.allow_partial:
        print(
            f"refusing --pages against database {settings.postgres_db!r}: partial passages must not enter the live corpus "
            f"(use POSTGRES_DB={TEST_DATABASE} or pass --allow-partial)",
            file=sys.stderr,
        )
        return 2

    configure_console()
    engine, session_factory = create_resources(settings.database_url)
    shared: dict = {}
    failed: list[str] = []
    try:
        for document_id in dict.fromkeys(args.document_id):
            if await run_document(args, settings.postgres_db, session_factory, document_id, shared):
                failed.append(document_id)
    finally:
        await engine.dispose()
    if failed:
        log.error("ingestion failed for: %s", ", ".join(failed))
        return 1
    return 0


async def run_document(args, database: str, session_factory, document_id: str, shared: dict) -> int:
    """Ingest one document; returns 0 on success and 1 on failure (recorded in its status.json when it exists)."""
    status: Status | None = None
    file_handler: logging.Handler | None = None
    try:
        version_id, sha256, snapshot_relative = await load.get_registered_version(session_factory, document_id)
        pdf_path = snapshot_file(snapshot_relative)
        if not pdf_path.is_file():
            raise FileNotFoundError(f"registered snapshot is missing: {pdf_path}")

        total_pages = convert.page_count(pdf_path)
        first_page, last_page = args.pages or (1, total_pages)
        last_page = min(last_page, total_pages)
        out_dir = INGEST_ROOT / sha256
        if args.pages:
            out_dir = out_dir / f"partial-{first_page}-{last_page}"
        out_dir.mkdir(parents=True, exist_ok=True)
        file_handler = attach_log_file(out_dir / "ingest.log")

        status = Status(
            out_dir / "status.json",
            {
                "document_id": document_id,
                "source_version_id": str(version_id),
                "sha256": sha256,
                "database": database,
                "page_range": [first_page, last_page],
                "pages_total": last_page - first_page + 1,
                "pages_done": 0,
                "stage": "starting",
                "stages": args.stages,
                "ocr_policy": args.ocr,
                "counts": {},
                "embedder": None,
            },
        )
        log.info(
            "document=%s version=%s pages=%s-%s ocr=%s database=%s out=%s",
            document_id,
            version_id,
            first_page,
            last_page,
            args.ocr,
            database,
            out_dir,
        )
        if args.reconvert:
            log.info("--reconvert: removed %s cached files from %s", convert.clear_cache(out_dir), out_dir)
        await execute_stages(
            args, status, session_factory, shared, version_id, pdf_path, out_dir, first_page, last_page
        )
        status.update(stage="done")
        log.info("done %s: %s", document_id, status.data["counts"])
        return 0
    except Exception as error:
        log.exception("ingestion of %s failed", document_id)
        if status is not None:
            status.data["errors"].append(f"{type(error).__name__}: {error}")
            status.update(stage="failed")
        return 1
    finally:
        if file_handler is not None:
            logging.getLogger().removeHandler(file_handler)
            file_handler.close()


async def execute_stages(
    args,
    status: Status,
    session_factory,
    shared: dict,
    version_id,
    pdf_path: Path,
    out_dir: Path,
    first_page: int,
    last_page: int,
) -> None:
    if "convert" in args.stages:
        status.update(stage="convert")
        began = time.perf_counter()
        summary = await asyncio.to_thread(
            convert.convert_pdf,
            pdf_path,
            out_dir,
            first_page=first_page,
            last_page=last_page,
            batch_pages=args.batch_pages,
            device=args.device,
            ocr_policy=args.ocr,
            progress=lambda done, _last: status.update(pages_done=done - first_page + 1),
        )
        status.update(
            pages_by_method=summary["pages_by_method"],
            quality_counts=summary["counts"],
            convert_seconds=round(time.perf_counter() - began, 1),
        )

    if "load" in args.stages:
        status.update(stage="load")
        structure = build_structure(convert.read_blocks(out_dir, first_page, last_page))
        written = await load.load_structure(session_factory, version_id, structure, rebuild=args.rebuild)
        flagged = sum(1 for provision in structure.provisions if provision.duplicate)
        warned = sum(1 for chunk in structure.chunks if chunk.warnings)
        status.update(
            counts={
                **status.data["counts"],
                **written,
                "duplicate_identifiers": flagged,
                "chunks_with_warnings": warned,
            }
        )

    if "embed" in args.stages:
        status.update(stage="embed")
        if "embedder" not in shared:
            shared["embedder"] = await asyncio.to_thread(load_embedder, args.embedder, device=args.device)
        embedder = shared["embedder"]
        status.update(
            embedder={"model": embedder.model, "revision": embedder.revision, "dimensions": embedder.dimensions}
        )
        began = time.perf_counter()
        inserted = await load.embed_missing(session_factory, version_id, embedder)
        status.update(embed_seconds=round(time.perf_counter() - began, 1))
        status.update(counts={**status.data["counts"], "embeddings_inserted": inserted})
    status.update(counts={**status.data["counts"], **await load.count_rows(session_factory, version_id)})


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.reconvert and "convert" not in args.stages:
        parser.error("--reconvert needs the convert stage in --stages")
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
