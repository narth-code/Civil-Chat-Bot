"""End-to-end check of selected eval/golden.json cases through the production agent (`app.rag.agent.ask`).

Uses the real LangGraph agent, the real Ollama router/extractor model, and the real retrieval stack. The scope is
the case's manual only when ``manual_named`` is true, matching eval/retrieval_eval.py.

Modes:
- default (single turn): the verbatim golden question with empty history. The corpus is Austin-only, so the agent
  asks for the city when the question does not name one. That clarification is recorded as the result.
- ``--city-followup``: when the first turn asks for the city, a second turn replies "Austin, Texas" with the
  prior question and clarification as history, exactly as the chat UI sends it. The second turn is scored.

Scoring:
- provision cases pass when the answer is a KB answer (``source_used == "kb"``) and the expected
  (document_id, provision identifier) appears among its provision citations. Expected facts are also checked
  against those cited quotes and reported separately.
- negative cases pass when the answer is not a KB answer, no provision citation is returned, and no key term appears
  in the answer. Web answers are reported as `web_answer` for review because they count as leaks when a key term appears.

Usage:
    PYTHONPATH=. .venv-ingest/Scripts/python.exe eval/golden_e2e.py --city-followup --ids DRN-14,NEG-01
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import app.rag.agent as agent_module
from app.rag.agent import ask

ROOT = Path(__file__).resolve().parents[1]
GOLDEN_PATH = ROOT / "eval" / "golden.json"
OUT_DIR = ROOT / "data" / "eval"
DEFAULT_IDS = ("DRN-14", "FIR-19", "TRN-08", "NEG-01", "DRN-19", "ENV-14", "TRN-04", "FIR-14")
CITY_REPLY = "Austin, Texas"


def provision_citations(result: dict) -> list[dict]:
    return [c for c in result.get("citations") or [] if c.get("type") == "provision"]


def diverse_ids(cases: list[dict], limit: int) -> list[str]:
    """Round-robin over manuals so a small run covers every document and case kind."""
    buckets: dict[str, list[str]] = {}
    for case in cases:
        document = ((case.get("expected_provisions") or [{}])[0]).get("document_id", "none")
        buckets.setdefault(document, []).append(case["id"])
    ordered = [buckets[key] for key in sorted(buckets)]
    picked: list[str] = []
    for rank in range(max(len(ids) for ids in ordered)):
        for ids in ordered:
            if rank < len(ids) and len(picked) < limit:
                picked.append(ids[rank])
    return picked


def score_provision(case: dict, result: dict) -> dict:
    expected = case["expected_provisions"][0]
    cited = provision_citations(result)
    cited_pairs = [[c.get("document_id"), c.get("identifier")] for c in cited]
    matching = [
        c
        for c in cited
        if c.get("document_id") == expected["document_id"] and c.get("identifier") == expected["provision_identifier"]
    ]
    cited_expected = bool(matching)
    expected_quotes = "\n".join(c.get("quote", "") for c in matching)
    facts = [
        {"pattern": pattern, "hit": bool(re.search(pattern, expected_quotes, re.IGNORECASE))}
        for pattern in case["expected_facts"]
    ]
    return {
        "passed": result.get("source_used") == "kb" and cited_expected,
        "cited_expected": cited_expected,
        "cited": cited_pairs,
        "facts": facts,
    }


def score_negative(case: dict, result: dict) -> dict:
    answer = result.get("answer") or ""
    key_hits = [term for term in case["key_terms"] if re.search(rf"\b{re.escape(term)}\b", answer, re.IGNORECASE)]
    cited = provision_citations(result)
    return {
        "passed": result.get("source_used") != "kb" and not cited and not key_hits,
        "cited": [[c.get("document_id"), c.get("identifier")] for c in cited],
        "key_term_hits_in_answer": key_hits,
        "web_answer": result.get("source_used") == "web",
    }


def turn_record(result: dict) -> dict:
    return {
        "source_used": result.get("source_used"),
        "answer": result.get("answer"),
        "citations": result.get("citations") or [],
        "trace": result.get("trace") or [],
    }


async def run_case(case: dict, city_followup: bool) -> dict:
    scope = [case["manual"]] if case["manual"] and case["manual_named"] else None
    started = time.perf_counter()
    base = {
        "id": case["id"],
        "kind": case["kind"],
        "question": case["question"],
        "scope": scope,
        "manual_named": case["manual_named"],
        "confusable": case["confusable"],
        "city_followup": city_followup,
    }
    try:
        first = await ask(case["question"], None, scope=scope)
        turns = [turn_record(first)]
        final = first
        if (
            city_followup
            and first.get("source_used") == "clarification"
            and first.get("answer") == agent_module.CITY_CLARIFICATION
        ):
            history = [{"role": "user", "content": case["question"]}, {"role": "assistant", "content": first["answer"]}]
            final = await ask(CITY_REPLY, history, scope=scope)
            turns.append(turn_record(final))
            turns[-1]["user_message"] = CITY_REPLY
    except Exception as error:  # per-case failure is recorded, the remaining cases still run
        return {
            **base,
            "elapsed_seconds": round(time.perf_counter() - started, 2),
            "error": f"{type(error).__name__}: {error}",
            "passed": False,
        }
    elapsed = round(time.perf_counter() - started, 2)
    score = score_negative(case, final) if case["expect"] == "absent" else score_provision(case, final)
    return {**base, "elapsed_seconds": elapsed, "turns": turns, **turn_record(final), **score}


def print_report(rows: list[dict]) -> None:
    print(f"\n{'id':<8}{'kind':<12}{'source':<22}{'pass':<6}{'turns':>6}{'sec':>7}  detail")
    for row in rows:
        if row.get("error"):
            detail = row["error"][:90]
        elif row["kind"] == "negative":
            detail = f"cited={row['cited']} key_hits={row['key_term_hits_in_answer']}"
        else:
            facts = sum(1 for f in row["facts"] if f["hit"])
            detail = f"expected_cited={row['cited_expected']} cited={row['cited']} facts={facts}/{len(row['facts'])}"
        source = str(row.get("source_used") or "-")
        turns = len(row.get("turns") or [])
        print(
            f"{row['id']:<8}{row['kind']:<12}{source:<22}{'PASS' if row['passed'] else 'FAIL':<6}"
            f"{turns:>6}{row['elapsed_seconds']:>7.1f}  {detail}"
        )
    passed = sum(1 for row in rows if row["passed"])
    print(f"\n{passed}/{len(rows)} passed")


async def main(args: argparse.Namespace) -> None:
    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    by_id = {case["id"]: case for case in golden["cases"]}
    if args.ids:
        ids = [part.strip() for part in args.ids.split(",")]
    elif args.limit:
        ids = diverse_ids(golden["cases"], args.limit)
    else:
        ids = list(DEFAULT_IDS)
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise SystemExit(f"unknown case IDs: {missing}")
    rows: list[dict] = []
    try:
        for case_id in ids:
            rows.append(await run_case(by_id[case_id], args.city_followup))
    finally:
        if agent_module._engine is not None:
            await agent_module._engine.dispose()

    created = datetime.now(timezone.utc)
    report = {
        "created_at": created.isoformat(),
        "golden_version": golden["version"],
        "city_followup": args.city_followup,
        "case_ids": ids,
        "passed": sum(1 for row in rows if row["passed"]),
        "total": len(rows),
        "results": rows,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"golden-e2e-{created.strftime('%Y%m%dT%H%M%SZ')}.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print_report(rows)
    print(f"\nWrote {out_path.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ids", help="comma-separated case IDs (default: the 8 diverse cases)")
    parser.add_argument("--limit", type=int, help="run N cases round-robin across manuals (overrides default set)")
    parser.add_argument(
        "--city-followup",
        action="store_true",
        help="answer the city clarification with 'Austin, Texas' in a second turn and score that turn",
    )
    asyncio.run(main(parser.parse_args()))
