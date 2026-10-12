"""Golden retrieval set integrity and the pure scoring functions of the eval harness. No database, no LLM.

Checking that each expected provision exists in the live corpus and that each fact regex matches its text is a
one-time validation against the current source versions (the live DB is never used by tests).
"""

from __future__ import annotations

import importlib.util
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = json.loads((ROOT / "eval" / "golden.json").read_text(encoding="utf-8"))
CASES = GOLDEN["cases"]
PROVISION_CASES = [case for case in CASES if case["expect"] == "provision"]
NEGATIVE_CASES = [case for case in CASES if case["expect"] == "absent"]
MANUALS = ("drainage", "environmental", "fire", "transportation")
KINDS = {"numeric", "definition", "table", "regime", "procedure"}
ID_PREFIX = {"drainage": "DRN", "environmental": "ENV", "fire": "FIR", "transportation": "TRN"}


def load_eval_module(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "eval" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


retrieval_eval = load_eval_module("retrieval_eval")
golden_e2e = load_eval_module("golden_e2e")


@dataclass
class Hit:
    document_id: str
    provision_identifier: str
    text: str
    score: float = 1.0
    found_by: tuple[str, ...] = field(default_factory=tuple)


def test_golden_counts_and_coverage():
    assert GOLDEN["version"] == 1
    assert len(PROVISION_CASES) == 90
    assert len(NEGATIVE_CASES) == 8
    assert Counter(case["manual"] for case in PROVISION_CASES) == {
        "drainage": 20,
        "environmental": 20,
        "fire": 24,
        "transportation": 26,
    }
    assert {case["kind"] for case in PROVISION_CASES} == KINDS
    ids = [case["id"] for case in CASES]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("case", PROVISION_CASES, ids=lambda case: case["id"])
def test_provision_case_is_well_formed(case):
    assert case["manual"] in MANUALS
    assert case["id"].startswith(f"{ID_PREFIX[case['manual']]}-")
    assert case["kind"] in KINDS
    for flag in ("paraphrased", "confusable", "manual_named"):
        assert isinstance(case[flag], bool)
    assert case["question"].strip().endswith("?")
    assert len(case["expected_provisions"]) == 1
    expected = case["expected_provisions"][0]
    assert expected["document_id"] == case["manual"]
    assert expected["provision_identifier"].strip()
    assert case["expected_facts"], "every provision case needs at least one fact regex"
    for pattern in case["expected_facts"]:
        re.compile(pattern)
    pages = case["pages"]
    assert pages is None or (len(pages) == 2 and 1 <= pages[0] <= pages[1])


@pytest.mark.parametrize("case", NEGATIVE_CASES, ids=lambda case: case["id"])
def test_negative_case_is_well_formed(case):
    assert case["manual"] is None
    assert case["kind"] == "negative"
    assert case["expected_provisions"] == []
    assert case["key_terms"] and all(term == term.lower() and term.strip() for term in case["key_terms"])
    assert case["question"].strip()


def test_manual_named_cases_name_their_manual_in_the_question():
    names = {
        "drainage": "Drainage Criteria Manual",
        "environmental": "Environmental Criteria Manual",
        "fire": "Fire Protection Criteria Manual",
        "transportation": "Transportation Criteria Manual",
    }
    for case in PROVISION_CASES:
        if case["manual_named"]:
            assert names[case["manual"]] in case["question"], case["id"]


def test_provision_scoring_reports_rank_and_facts_only_from_the_expected_provision():
    case = {
        "id": "X-1",
        "manual": "fire",
        "kind": "numeric",
        "confusable": False,
        "manual_named": False,
        "expected_provisions": [{"document_id": "fire", "provision_identifier": "2.2.3"}],
        "expected_facts": ["Section 105\\.6", "permit"],
    }
    hits = [
        Hit("fire", "2.2.1", "Section 105.6 and a permit appear elsewhere"),
        Hit("fire", "2.2.3", "Section 105.6 applies"),
    ]
    scored = retrieval_eval.score_provision(case, hits)
    assert scored["found_at"] == 2
    assert [fact["hit"] for fact in scored["facts"]] == [True, False]


def test_provision_scoring_misses_when_the_document_differs():
    case = {
        "id": "X-2",
        "manual": "fire",
        "kind": "numeric",
        "confusable": False,
        "manual_named": False,
        "expected_provisions": [{"document_id": "fire", "provision_identifier": "2.2.3"}],
        "expected_facts": ["Section 105\\.6"],
    }
    assert retrieval_eval.score_provision(case, [Hit("drainage", "2.2.3", "Section 105.6")])["found_at"] is None


def test_aggregate_computes_recall_mrr_and_fact_rates():
    rows = [
        {"found_at": 1, "facts": [{"hit": True}, {"hit": True}]},
        {"found_at": 4, "facts": [{"hit": True}, {"hit": False}]},
        {"found_at": None, "facts": [{"hit": False}, {"hit": False}]},
        {"found_at": 9, "facts": [{"hit": True}, {"hit": True}]},
    ]
    summary = retrieval_eval.aggregate(rows)
    assert summary["n"] == 4
    assert summary["recall@1"] == 0.25
    assert summary["recall@3"] == 0.25
    assert summary["recall@5"] == 0.5
    assert summary["recall@8"] == 0.5
    assert summary["mrr"] == pytest.approx((1 + 1 / 4 + 1 / 9) / 4)
    assert summary["fact_hit_rate"] == pytest.approx(5 / 8)
    assert summary["all_facts_rate"] == 0.5
    assert retrieval_eval.aggregate([]) is None


def test_worst_misses_put_absent_gold_first_then_lowest_rank():
    rows = [
        {"id": "a", "found_at": 2},
        {"id": "b", "found_at": None},
        {"id": "c", "found_at": 1},
        {"id": "d", "found_at": 7},
    ]
    assert [row["id"] for row in retrieval_eval.worst_misses(rows, limit=3)] == ["b", "d", "a"]


def test_negative_scoring_matches_whole_words_only():
    case = {"key_terms": ["brisket"]}
    assert retrieval_eval.score_negative(case, [Hit("fire", "1.0", "Smoked brisket recipe")])["passed"] is False
    assert retrieval_eval.score_negative(case, [Hit("fire", "1.0", "brisketing is not a word here")])["passed"]


def test_e2e_provision_scoring_requires_a_kb_answer_citing_the_expected_provision():
    case = {
        "id": "X-3",
        "expect": "provision",
        "expected_provisions": [{"document_id": "fire", "provision_identifier": "2.2.3"}],
        "expected_facts": ["Section 105\\.6"],
    }
    cited = {"type": "provision", "document_id": "fire", "identifier": "2.2.3", "quote": "Section 105.6 applies"}
    passing = golden_e2e.score_provision(case, {"source_used": "kb", "citations": [cited]})
    assert passing["passed"] is True
    assert passing["facts"] == [{"pattern": "Section 105\\.6", "hit": True}]
    assert golden_e2e.score_provision(case, {"source_used": "clarification", "citations": [cited]})["passed"] is False
    assert golden_e2e.score_provision(case, {"source_used": "kb", "citations": []})["passed"] is False


def test_e2e_negative_scoring_fails_on_provision_citation_or_key_term():
    case = {"id": "NEG-X", "expect": "absent", "key_terms": ["brisket"]}
    assert golden_e2e.score_negative(case, {"answer": "Out of scope.", "citations": []})["passed"] is True
    assert golden_e2e.score_negative(case, {"answer": "Brisket is a cut of beef.", "citations": []})["passed"] is False
    provision = {"type": "provision", "document_id": "fire", "identifier": "2.2.3", "quote": "x"}
    assert golden_e2e.score_negative(case, {"answer": "Out of scope.", "citations": [provision]})["passed"] is False
