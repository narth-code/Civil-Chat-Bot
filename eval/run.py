"""Run the fixed live-stack research evaluations without modifying the source corpus."""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CASES_PATH = Path(__file__).with_name("questions.json")
RESULTS_DIR = Path("data/eval")
OUTCOMES = {
    "kb": "kb_answer",
    "web": "web_answer",
    "clarification": "clarification",
    "blocked": "blocked",
    "insufficient_evidence": "insufficient",
    "direct": "direct",
    "error": "error",
}


def normalized(value: Any) -> str:
    return " ".join(str(value).casefold().split())


def markdown(value: Any) -> str:
    return " ".join(str(value).replace("|", "\\|").split())


def page_numbers(value: Any) -> set[int]:
    if value is None:
        return set()
    if isinstance(value, int):
        return {value}
    pages: set[int] = set()
    for match in re.finditer(r"(?<!\d)(\d+)(?:\s*[-–]\s*(\d+))?(?!\d)", str(value)):
        start = int(match.group(1))
        end = int(match.group(2) or start)
        if 0 < start <= end <= start + 100:
            pages.update(range(start, end + 1))
    return pages


def provenance_matches(case: dict[str, Any], citations: list[dict[str, Any]]) -> bool:
    expected = {(p["document_id"], p["identifier"]) for p in case.get("expected_provisions", [])}
    pages = set(case.get("pdf_pages", []))
    for citation in citations:
        if citation.get("type") != "provision":
            continue
        pair = (citation.get("document_id"), citation.get("identifier"))
        if pair in expected or (
            citation.get("document_id") in {doc for doc, _ in expected} and page_numbers(citation.get("pages")) & pages
        ):
            return True
    return False


def evaluate(case: dict[str, Any], response: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    outcome = OUTCOMES.get(response.get("source_used"), "unknown")
    allowed = case["expect_outcome"]
    allowed = allowed if isinstance(allowed, list) else [allowed]
    if outcome not in allowed:
        failures.append(f"outcome {outcome}, expected {'/'.join(allowed)}")
    answer = normalized(response.get("answer") or "")
    citations = response.get("citations") or []
    trace = response.get("trace") or []
    if not isinstance(citations, list) or not isinstance(trace, list):
        return [*failures, "malformed citations or trace"]
    if outcome in {"kb_answer", "web_answer"}:
        for alternatives in case.get("required_groups", []):
            if not any(normalized(phrase) in answer for phrase in alternatives):
                failures.append(f"missing answer terms: {' / '.join(alternatives)}")
    if outcome == "kb_answer" and not provenance_matches(case, citations):
        failures.append("no expected provision or overlapping PDF page cited")
    if case.get("forbid_web") and (
        any(citation.get("type") == "web" for citation in citations)
        or any(re.search(r"\bweb\b", str(step), re.I) for step in trace)
    ):
        failures.append("used web despite forbid_web")
    if case.get("forbid_kb") and (
        outcome == "kb_answer" or any(citation.get("type") == "provision" for citation in citations)
    ):
        failures.append("used Austin corpus for an outside-authority question")
    if case.get("must_not_block") and outcome == "blocked":
        failures.append("legitimate hazard research blocked")
    clarification = normalized(response.get("clarification") or response.get("answer") or "")
    if outcome == "clarification":
        for phrase in case.get("clarification_must_mention", []):
            if normalized(phrase) not in clarification:
                failures.append(f"clarification omits {phrase}")
        for phrase in case.get("clarification_must_not_contain", []):
            if normalized(phrase) in clarification:
                failures.append(f"clarification prematurely mentions {phrase}")
    for phrase in case.get("must_not_assert", []):
        if normalized(phrase) in answer:
            failures.append(f"asserts prohibited phrase: {phrase}")
    return failures


def list_cases(cases: list[dict[str, Any]]) -> None:
    print("| ID | Research question | Prior turns | Expected outcome | Expected provisions / PDF pages |")
    print("| --- | --- | --- | --- | --- |")
    for case in cases:
        history = case.get("history", [])
        summary = "; ".join(f"{turn['role']}: {turn['content']}" for turn in history) or "—"
        provisions = ", ".join(f"{p['document_id']} {p['identifier']}" for p in case.get("expected_provisions", []))
        pages = ",".join(map(str, case.get("pdf_pages", [])))
        evidence = f"{provisions} (PDF {pages})" if provisions else "—"
        expected = case["expect_outcome"]
        if isinstance(expected, list):
            expected = " / ".join(expected)
        print(
            f"| {markdown(case['id'])} | {markdown(case['question'])} | {markdown(summary)} | "
            f"{markdown(expected)} | {markdown(evidence)} |"
        )


async def run(cases: list[dict[str, Any]]) -> None:
    from app.rag.agent import ask

    started = time.monotonic()
    results: list[dict[str, Any]] = []
    print("| ID | Result | Reason |")
    print("| --- | --- | --- |")
    for case in cases:
        item: dict[str, Any] = {"id": case["id"], "expected_outcome": case["expect_outcome"]}
        try:
            response = await ask(case["question"], case.get("history"))
            item["response"] = response
            failures = evaluate(case, response)
        except Exception as exc:  # A provider failure must not hide results for subsequent cases.
            failures = [f"{type(exc).__name__}: {str(exc).replace(chr(10), ' ')[:160]}"]
        item["passed"] = not failures
        item["failures"] = failures
        results.append(item)
        reason = "; ".join(failures) if failures else "—"
        print(f"| {markdown(case['id'])} | {'PASS' if item['passed'] else 'FAIL'} | {markdown(reason)} |", flush=True)
    elapsed = time.monotonic() - started
    passed = sum(item["passed"] for item in results)
    print(f"Pass rate: {passed}/{len(results)} ({passed / len(results):.1%}); elapsed: {elapsed:.1f}s")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    destination = RESULTS_DIR / f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}.json"
    destination.write_text(
        json.dumps(
            {"elapsed_seconds": elapsed, "passed": passed, "total": len(results), "results": results},
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Results: {destination}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Review or run the fixed civil-source evaluation set")
    parser.add_argument("--ids", help="comma-separated case IDs")
    parser.add_argument("--limit", type=int, help="evaluate only the first N selected cases")
    parser.add_argument("--list", action="store_true", help="print all selected cases as a Markdown table")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    if args.ids:
        ids = [part.strip() for part in args.ids.split(",") if part.strip()]
        unknown = set(ids) - {case["id"] for case in cases}
        if unknown:
            parser.error(f"unknown case IDs: {', '.join(sorted(unknown))}")
        cases = [case for case in cases if case["id"] in ids]
    if args.limit is not None:
        cases = cases[: args.limit]
    if args.list:
        list_cases(cases)
    else:
        asyncio.run(run(cases))


if __name__ == "__main__":
    main()
