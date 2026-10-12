"""Inspect hybrid retrieval over a document's current source version.

    python -m app.ingest.search "question" --document-id environmental [--k 8] [--embedder auto|harrier|nomic]

The search itself lives in app.rag.retrieval; this only loads the embedder and prints the hits.
"""

from __future__ import annotations

import argparse
import asyncio
import logging

from app.core.config import get_settings
from app.db.session import create_resources
from app.ingest.embedding import load_embedder
from app.rag.retrieval import hybrid_search

SNIPPET_CHARS = 240


def format_pages(start: int | None, end: int | None) -> str:
    if start is None or end is None:
        return "unknown"
    return str(start) if start == end else f"{start}-{end}"


async def run(args: argparse.Namespace) -> int:
    settings = get_settings()
    engine, session_factory = create_resources(settings.database_url)
    try:
        embedder = await asyncio.to_thread(
            load_embedder, args.embedder, device=args.device, ollama_url=settings.ollama_url
        )
        results = await hybrid_search(
            session_factory, embedder, args.question, document_ids=[args.document_id], k=args.k
        )
    finally:
        await engine.dispose()
    print(f"embedder: {embedder.model} revision={embedder.revision[:12]} dims={embedder.dimensions}")
    if not results:
        print("no matching chunks")
    for rank, hit in enumerate(results, start=1):
        snippet = " ".join(hit.text.split())[:SNIPPET_CHARS]
        print(f"{rank}. {hit.provision_identifier} {hit.provision_title or ''}".rstrip())
        print(
            f"   pages {format_pages(hit.page_start, hit.page_end)} | rrf {hit.score:.4f} | from {' + '.join(hit.found_by)}"
        )
        print(f"   {snippet}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m app.ingest.search")
    parser.add_argument("question")
    parser.add_argument("--document-id", required=True)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--embedder", choices=("auto", "harrier", "nomic"), default="auto")
    parser.add_argument("--device", default="cuda")
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    return asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
