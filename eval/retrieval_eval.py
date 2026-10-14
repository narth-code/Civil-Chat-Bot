"""Deterministic retrieval-only evaluation (no LLM).

``--source golden`` (default) scores eval/golden.json with the raw question; the scope is the case's manual only when
``manual_named`` is true. ``--source eval`` scores the kb_answer cases of eval/questions.json under three query
constructions: ``raw`` (question), ``agent`` (the extract_context query the agent actually used, read from an agent
eval run) and ``hint`` (question plus the expected manual's title; an oracle upper bound). Every construction uses the
scope the agent recorded for that case, so only the query text varies.

Usage:
    PYTHONPATH=. .venv-ingest/Scripts/python.exe eval/retrieval_eval.py
    PYTHONPATH=. .venv-ingest/Scripts/python.exe eval/retrieval_eval.py --kind numeric --manual fire
    PYTHONPATH=. .venv-ingest/Scripts/python.exe eval/retrieval_eval.py --compare data/eval/retrieval-<UTC>.json
    PYTHONPATH=. .venv-ingest/Scripts/python.exe eval/retrieval_eval.py --source eval --agent-results data/eval/<UTC>.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import sqlalchemy as sa

from app.core.config import get_settings
from app.db.session import create_resources
from app.rag.retrieval import get_embedder, hybrid_search, split_duplicate

ROOT = Path(__file__).resolve().parents[1]
GOLDEN_PATH = ROOT / "eval" / "golden.json"
OUT_DIR = ROOT / "data" / "eval"
EVAL_QUESTIONS_PATH = ROOT / "eval" / "questions.json"
KS = (1, 3, 5, 8)
MANUALS = ("drainage", "environmental", "fire", "transportation")
KINDS = ("numeric", "definition", "table", "regime", "procedure")
CONSTRUCTIONS = ("raw", "agent", "hint")


def select_cases(cases: list[dict], args: argparse.Namespace) -> list[dict]:
    ids = {part.strip() for part in args.ids.split(",")} if args.ids else None
    selected = []
    for case in cases:
        if ids and case["id"] not in ids:
            continue
        if args.manual and case["manual"] != args.manual:
            continue
        if args.kind and case["kind"] != args.kind:
            continue
        selected.append(case)
    return selected


def top_summary(hits) -> list[dict]:
    return [
        {
            "rank": rank,
            "document_id": hit.document_id,
            "provision_identifier": hit.provision_identifier,
            "score": round(hit.score, 6),
            "found_by": list(hit.found_by),
        }
        for rank, hit in enumerate(hits, start=1)
    ]


def score_provision(case: dict, hits) -> dict:
    manual = case["manual"]
    ident = case["expected_provisions"][0]["provision_identifier"]
    # Gold may name a duplicate-marked stored identifier; retrieval reports the base section number.
    base = split_duplicate(ident)[0]
    found_at = next(
        (
            rank
            for rank, hit in enumerate(hits, start=1)
            if hit.document_id == manual and hit.provision_identifier == base
        ),
        None,
    )
    gold_texts = [hit.text for hit in hits if hit.document_id == manual and hit.provision_identifier == base]
    facts = [
        {"pattern": pattern, "hit": any(re.search(pattern, text, re.IGNORECASE) for text in gold_texts)}
        for pattern in case["expected_facts"]
    ]
    return {
        "manual": manual,
        "kind": case["kind"],
        "confusable": case["confusable"],
        "manual_named": case["manual_named"],
        "gold": ident,
        "found_at": found_at,
        "facts": facts,
    }


def score_negative(case: dict, hits) -> dict:
    hit_terms = [
        term
        for term in case["key_terms"]
        if any(re.search(rf"\b{re.escape(term)}\b", hit.text, re.IGNORECASE) for hit in hits)
    ]
    return {"hit_terms": hit_terms, "passed": not hit_terms}


def aggregate(rows: list[dict]) -> dict | None:
    if not rows:
        return None
    n = len(rows)
    result: dict = {"n": n}
    for k in KS:
        result[f"recall@{k}"] = sum(1 for r in rows if r["found_at"] is not None and r["found_at"] <= k) / n
    result["mrr"] = sum(1 / r["found_at"] for r in rows if r["found_at"] is not None) / n
    facts_total = sum(len(r["facts"]) for r in rows)
    facts_hit = sum(sum(1 for f in r["facts"] if f["hit"]) for r in rows)
    result["fact_hit_rate"] = facts_hit / facts_total if facts_total else None
    result["all_facts_rate"] = sum(1 for r in rows if all(f["hit"] for f in r["facts"])) / n
    return result


def build_summary(provision_rows: list[dict], negative_rows: list[dict]) -> dict:
    return {
        "overall": aggregate(provision_rows),
        "by_manual": {m: aggregate([r for r in provision_rows if r["manual"] == m]) for m in MANUALS},
        "by_kind": {k: aggregate([r for r in provision_rows if r["kind"] == k]) for k in KINDS},
        "confusable": aggregate([r for r in provision_rows if r["confusable"]]),
        "negatives": {"n": len(negative_rows), "passed": sum(1 for r in negative_rows if r["passed"])},
    }


def worst_misses(provision_rows: list[dict], limit: int = 10) -> list[dict]:
    ordered = sorted(
        provision_rows,
        key=lambda r: (0 if r["found_at"] is None else 1, -(r["found_at"] or 0), r["id"]),
    )
    return [r for r in ordered if r["found_at"] is None or r["found_at"] > 1][:limit]


def pct(value: float | None) -> str:
    return "  n/a" if value is None else f"{100 * value:5.1f}"


def print_table(title: str, groups: dict[str, dict | None]) -> None:
    print(f"\n{title}")
    print(f"{'group':<16}{'n':>4}{'R@1':>7}{'R@3':>7}{'R@5':>7}{'R@8':>7}{'Fact%':>7}{'AllF%':>7}{'MRR':>7}")
    for name, agg in groups.items():
        if agg is None:
            continue
        print(
            f"{name:<16}{agg['n']:>4}"
            f"{pct(agg['recall@1']):>7}{pct(agg['recall@3']):>7}{pct(agg['recall@5']):>7}{pct(agg['recall@8']):>7}"
            f"{pct(agg['fact_hit_rate']):>7}{pct(agg['all_facts_rate']):>7}"
            f"{100 * agg['mrr']:6.1f}"
        )


def print_compare(previous: dict, current: dict) -> None:
    print("\nDelta vs previous run (percentage points; MRR x100)")
    prev, cur = previous["summary"], current["summary"]
    rows = [("overall", prev["overall"], cur["overall"])]
    rows += [(f"manual:{m}", prev["by_manual"].get(m), cur["by_manual"].get(m)) for m in MANUALS]
    rows += [(f"kind:{k}", prev["by_kind"].get(k), cur["by_kind"].get(k)) for k in KINDS]
    for name, before, after in rows:
        if not before or not after:
            continue
        deltas = [
            (after[key] - before[key]) * 100
            for key in ("recall@1", "recall@3", "recall@5", "recall@8", "fact_hit_rate", "mrr")
            if after.get(key) is not None and before.get(key) is not None
        ]
        formatted = " ".join(f"{d:+6.1f}" for d in deltas)
        print(f"{name:<22} R@1 R@3 R@5 R@8 Fact MRR: {formatted}")


async def run(args: argparse.Namespace) -> None:
    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    cases = select_cases(golden["cases"], args)
    engine, sessions = create_resources(get_settings().database_url)
    embedder = get_embedder()
    provision_rows: list[dict] = []
    negative_rows: list[dict] = []
    try:
        for case in cases:
            is_provision = case["expect"] == "provision"
            scope = [case["manual"]] if is_provision and case["manual_named"] else None
            hits = await hybrid_search(sessions, embedder, case["question"], document_ids=scope, k=args.k)
            base = {"id": case["id"], "question": case["question"], "scope": scope, "top": top_summary(hits)}
            if is_provision:
                provision_rows.append({**base, **score_provision(case, hits)})
            else:
                negative_rows.append({**base, "key_terms": case["key_terms"], **score_negative(case, hits)})
    finally:
        await engine.dispose()

    summary = build_summary(provision_rows, negative_rows)
    misses = worst_misses(provision_rows)
    created = datetime.now(timezone.utc)
    report = {
        "created_at": created.isoformat(),
        "golden_version": golden["version"],
        "case_count": len(cases),
        "filters": {"ids": args.ids, "manual": args.manual, "kind": args.kind, "k": args.k},
        "retrieval": "app.rag.retrieval.hybrid_search; raw question; scope=[manual] only when manual_named",
        "embedder": {
            "model": embedder.model,
            "revision": embedder.revision,
            "dimensions": embedder.dimensions,
        },
        "summary": summary,
        "worst_misses": [r["id"] for r in misses],
        "provision_cases": provision_rows,
        "negative_cases": negative_rows,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"retrieval-{created.strftime('%Y%m%dT%H%M%SZ')}.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"cases={len(cases)} provision={len(provision_rows)} negative={len(negative_rows)} k={args.k}")
    print(f"embedder={embedder.model}@{embedder.revision} dims={embedder.dimensions}")
    print_table(
        "Overall / manual / kind (provision cases)",
        {
            "overall": summary["overall"],
            **{f"manual:{m}": summary["by_manual"][m] for m in MANUALS},
            **{f"kind:{k}": summary["by_kind"][k] for k in KINDS},
            "confusable": summary["confusable"],
        },
    )
    neg = summary["negatives"]
    print(f"\nNegatives (no top-{args.k} chunk contains a key term): {neg['passed']}/{neg['n']} passed")
    print("\nWorst misses (not found first, then lowest rank):")
    for r in misses:
        rank = "miss" if r["found_at"] is None else f"rank {r['found_at']}"
        top1 = r["top"][0]["provision_identifier"] if r["top"] else "-"
        print(f"  {r['id']:<8}{r['manual']:<15}{rank:<9}gold={r['gold']:<16}top1={top1:<14}{r['question'][:80]}")
    print(f"\nWrote {out_path.relative_to(ROOT).as_posix()}")

    if args.compare:
        previous = json.loads(Path(args.compare).read_text(encoding="utf-8"))
        print_compare(previous, report)


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)]


def rank_label(rank: int | None) -> str:
    return "miss" if rank is None else f"r{rank}"


def scope_for(response: dict) -> list[str] | None:
    # Mirrors agent.retrieve_kb: an explicit scope wins, otherwise the extracted manual hint.
    if response["scope"]:
        return list(response["scope"])
    return [response["manual_hint"]] if response["manual_hint"] else None


def load_eval_cases(agent_results: Path, ids: set[str] | None) -> list[dict]:
    # kb_answer cases of eval/questions.json, each joined to the agent's recorded extract_context output.
    questions = json.loads(EVAL_QUESTIONS_PATH.read_text(encoding="utf-8"))
    recorded = {row["id"]: row["response"] for row in json.loads(agent_results.read_text(encoding="utf-8"))["results"]}
    return [
        {**case, "agent": recorded[case["id"]]}
        for case in questions
        if case["expect_outcome"] == "kb_answer" and (ids is None or case["id"] in ids)
    ]


def score_expected(provisions: list[dict], hits) -> int | None:
    # Best rank of any expected provision. Duplicate-marked gold matches its base section number.
    ranks: list[int] = []
    for expected in provisions:
        base = split_duplicate(expected["identifier"])[0]
        ranks.extend(
            rank
            for rank, hit in enumerate(hits, start=1)
            if hit.document_id == expected["document_id"] and hit.provision_identifier == base
        )
    return min(ranks, default=None)


def recall_at(rows: list[dict], construction: str, k: int) -> float:
    found = [row["constructions"][construction]["found_at"] for row in rows]
    return sum(1 for rank in found if rank is not None and rank <= k) / len(found)


async def run_eval_source(args: argparse.Namespace) -> None:
    agent_results = Path(args.agent_results)
    cases = load_eval_cases(agent_results, {part.strip() for part in args.ids.split(",")} if args.ids else None)
    if not cases:
        print("no kb_answer cases selected")
        return
    engine, sessions = create_resources(get_settings().database_url)
    embedder = get_embedder()
    rows: list[dict] = []
    timings: dict[str, list[float]] = {name: [] for name in CONSTRUCTIONS}
    try:
        async with engine.connect() as connection:
            result = await connection.execute(
                sa.text(
                    "SELECT d.id, v.source_title FROM documents d "
                    "JOIN document_versions v ON v.id = d.current_source_version_id"
                )
            )
            titles = {row.id: row.source_title for row in result}
        for case in cases:
            agent = case["agent"]
            scope = scope_for(agent)
            queries = {
                "raw": case["question"],
                "agent": agent["current_query"] or case["question"],
                "hint": f"{case['question']} {titles[case['expected_provisions'][0]['document_id']]}",
            }
            constructions = {}
            for name, text in queries.items():
                started = time.perf_counter()
                hits = await hybrid_search(sessions, embedder, text, document_ids=scope, k=args.k)
                timings[name].append((time.perf_counter() - started) * 1000)
                constructions[name] = {
                    "query": text,
                    "found_at": score_expected(case["expected_provisions"], hits),
                    "top": top_summary(hits),
                }
            rows.append(
                {
                    "id": case["id"],
                    "question": case["question"],
                    "expected": case["expected_provisions"],
                    "scope": scope,
                    "agent_section_lookup": agent["section_ref"] is not None,
                    "constructions": constructions,
                }
            )
    finally:
        await engine.dispose()

    summary = {
        name: {
            **{f"R@{k}": recall_at(rows, name, k) for k in KS},
            "top5_count": round(recall_at(rows, name, 5) * len(rows)),
            "latency_p50_ms": percentile(timings[name], 0.5),
            "latency_p95_ms": percentile(timings[name], 0.95),
        }
        for name in CONSTRUCTIONS
    }
    created = datetime.now(timezone.utc)
    report = {
        "created_at": created.isoformat(),
        "source": "eval/questions.json kb_answer",
        "agent_results": args.agent_results,
        "filters": {"ids": args.ids, "k": args.k},
        "constructions": {
            "raw": "question",
            "agent": "extract_context current_query (question when empty)",
            "hint": "question + expected manual source_title (oracle manual identification; upper bound)",
        },
        "embedder": {"model": embedder.model, "revision": embedder.revision, "dimensions": embedder.dimensions},
        "summary": summary,
        "cases": rows,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"retrieval-eval-{created.strftime('%Y%m%dT%H%M%SZ')}.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"cases={len(rows)} k={args.k} embedder={embedder.model}@{embedder.revision[:12]}")
    header = "".join(f"{'R@' + str(k):>8}" for k in KS)
    print(f"\n{'construction':<14}{header}{'top5':>8}{'p50 ms':>10}{'p95 ms':>10}")
    for name in CONSTRUCTIONS:
        s = summary[name]
        cells = "".join(f"{100 * s[f'R@{k}']:8.1f}" for k in KS)
        print(
            f"{name:<14}{cells}{s['top5_count']:>5}/{len(rows)}{s['latency_p50_ms']:10.1f}{s['latency_p95_ms']:10.1f}"
        )
    print("\nCases with the expected provision outside top 5 (per construction):")
    for row in rows:
        ranks = {name: row["constructions"][name]["found_at"] for name in CONSTRUCTIONS}
        if any(rank is None or rank > 5 for rank in ranks.values()):
            gold = ", ".join(f"{e['document_id']}:{e['identifier']}" for e in row["expected"])
            cells = "  ".join(f"{name}={rank_label(rank)}" for name, rank in ranks.items())
            print(f"  {row['id']:<30}{cells}  gold={gold}")
    print(f"\nWrote {out_path.relative_to(ROOT).as_posix()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--source", choices=("golden", "eval"), default="golden", help="golden.json or eval/questions.json"
    )
    parser.add_argument("--k", type=int, default=get_settings().retrieval_top_k, help="hybrid_search result size")
    parser.add_argument("--kind", choices=KINDS + ("negative",), help="only cases of this kind (golden)")
    parser.add_argument("--manual", choices=MANUALS, help="only cases for this manual (golden)")
    parser.add_argument("--ids", help="comma-separated case IDs")
    parser.add_argument("--compare", help="earlier retrieval-*.json to diff against (golden)")
    parser.add_argument("--agent-results", help="agent eval JSON whose extract_context query is scored (eval)")
    args = parser.parse_args()
    if args.source == "eval":
        if not args.agent_results:
            parser.error("--source eval requires --agent-results")
        if args.kind or args.manual or args.compare:
            parser.error("--kind, --manual and --compare apply to --source golden only")
        asyncio.run(run_eval_source(args))
    else:
        asyncio.run(run(args))


if __name__ == "__main__":
    main()
