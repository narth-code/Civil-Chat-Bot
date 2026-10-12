# Fixed research evaluation

`questions.json` is a frozen, manually grounded set of Austin Source-provision questions plus context, filtering, and Evidence-gap cases. PDF page numbers are physical pages of the captured Sources, not printed page labels. `required_groups` are alternative answer phrasings within each group; every group must match. For the outside-Austin and unavailable-fact cases, either `web_answer` or `insufficient` is acceptable; neither a local Austin citation nor an invented fact is.

Review every case without making provider calls:

```sh
PYTHONPATH=. .venv-ingest/Scripts/python.exe eval/run.py --list
```

Run selected IDs or a small prefix against the **live** PostgreSQL/Ollama stack (no test database or pytest integration):

```sh
PYTHONPATH=. .venv-ingest/Scripts/python.exe eval/run.py --ids fire-water-plan-folding,transport-freeway-driveways
PYTHONPATH=. .venv-ingest/Scripts/python.exe eval/run.py --limit 6
```

Each run prints checks and saves the full response and failures to ignored `data/eval/<UTC timestamp>.json`. A case failure does not stop later cases. Expected provisions and page bounds were checked against `chunks`, `provisions`, and `document_versions` in the local live database; a passing match confirms overlap with cited Evidence, not engineering correctness or Source applicability. The six-case prefix deliberately mixes Source categories and policy outcomes.

## Golden set

`golden.json` (version 1) holds 80 provision-targeted retrieval cases (20 per manual: drainage, environmental, fire, transportation; kinds numeric, definition, table, regime, procedure; 7 confusable, 55 paraphrased, 8 naming their manual) and 8 negatives on topics outside the corpus. Each provision case names one expected `(document_id, identifier)`, optional physical page bounds, and regex facts that must appear in that provision's text. Questions are not reused from `questions.json`. Expected provisions, page bounds, and facts were validated once against the live corpus; the permanent test suite does not touch the database.

Retrieval baseline (no LLM; writes `data/eval/retrieval-<UTC>.json`):

```sh
PYTHONPATH=. .venv-ingest/Scripts/python.exe eval/retrieval_eval.py
PYTHONPATH=. .venv-ingest/Scripts/python.exe eval/retrieval_eval.py --kind numeric --compare data/eval/<previous>.json
```

Flags: `--k`, `--kind`, `--manual`, `--ids`, `--compare`. It calls the production `hybrid_search` with the raw question; it applies no context extraction or query rewrite, so it measures the retriever alone. The agent's retrieval query is different and is measured by the end-to-end check.

End-to-end check through `ask()` (requires Ollama `qwen3:8b`, the Harrier embedder, and the live database):

```sh
PYTHONPATH=. .venv-ingest/Scripts/python.exe eval/golden_e2e.py --city-followup
```

It runs 8 default cases. The corpus is Austin-only, so a question without a city first receives the city clarification; `--city-followup` answers `Austin, Texas` in a second turn and scores that turn. Output: `data/eval/golden-e2e-<UTC>.json`. A provision case passes only when the answer is KB-sourced and cites the expected provision; a negative passes only when no provision is cited and no key term appears.

Permanent integrity tests (no database, no LLM): `.venv/Scripts/python.exe -m pytest tests/test_golden_set.py -q`.
