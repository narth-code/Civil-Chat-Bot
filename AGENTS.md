# Repository Guidelines

## Project Overview
Agentic-RAG chat assistant for civil engineers. An engineer describes a project situation; the assistant retrieves from four ingested **Austin** criteria manuals (drainage, environmental, fire, transportation), may fall back to web search on the allowlisted domain (`library.municode.com`), and returns a cited answer, a precise clarification, a blocked notice, or "not enough information". **It never gives a guessed answer.** Local pilot, no auth. Not a git repo; the path contains a space (`Civil Chatbot`), so quote it in shell commands.

Stack: FastAPI + LangGraph + SQLAlchemy 2 async + Postgres/pgvector (Docker) + Ollama (`qwen3:8b` answers, `qwen2.5:7b` review only) + Harrier embeddings and bge reranker (GPU) + React 19/Vite/TypeScript UI.

## Architecture & Data Flow
One chat turn: `POST /api/conversations/{id}/messages` → `app/api/conversations.py` (thin) → `TurnService.send_message` (`app/services/turns.py`):
1. **Txn 1**: lock conversation, replay/conflict checks by client `message_id`, append user message, insert `routing_runs(status=pending)` with ordered context + scope child rows. History over 12,000 bytes fails `context_limit` with zero model calls (never truncated). Commit.
2. **No transaction held**: `await ask(question, history, scope=...)` = `app.rag.agent.ask` (the LangGraph).
3. **Txn 2**: finalize run (outcome, citations, trace, review, city), insert `run_trace_events`, append the `assistant_answer` message.

Agent graph (`app/rag/agent.py`): `route_question` → conversation=`direct_answer` | invalid=`blocked` | router error=`failed` | retrieval=`extract_context` → no city: `ask_clarification`; non-Austin: `search_web`; Austin: `retrieve_kb` → `grade_kb` → `extract_regimes` → `after_kb` → `generate_from_kb` | `ask_clarification` (regimes) | `retry_kb` (one wider retry) | `search_web` | `insufficient`. Web: `grade_web` → `generate_from_web` | `rewrite_query` (≤ `max_retries`) | `insufficient`. Answers end in `review_answer` (monitoring only; the score never gates and is hidden from engineers).

Retrieval (`app/rag/retrieval.py`, ADR 0006–0009): global (not per-manual) vector ranking over contextual vectors (`#ctx1`), RRF fusion, bge reranker over top 30 with only the top 4 reranked to lead, FTS leg weight 0. KB grade = top rerank score vs `retrieval_relevance_threshold` (calibrated), verdict computed in code. Exact section refs (`Sec 2.3.0`, `Appendix B`) go through `get_provision_chunks` before hybrid search. Provisions named `<id> (duplicate N)` carry real text; retrieval reports the base id plus `is_duplicate`.

Ingestion (GPU env): `app/ingest/run.py` convert (Docling+RapidOCR, resumable) → `structure` (printed ids → Provisions/Chunks) → `load` → `embed_missing`. Artifacts in `data/ingest/<sha256>/`.

Frontend: `src/api.ts` is the only network layer; all state is local `useState`; Vite proxies `/api` to `127.0.0.1:8000`.

## Key Directories
- `app/api/` routes (thin, `response_model=`); `app/services/` turn/conversation/audit/metrics/views; `app/core/` config, errors, logging, manuals manifest; `app/db/` ORM, session, bootstrap, snapshots.
- `app/rag/` agent graph, state/schemas, retrieval, rerank, trace, `trace_report` CLI. `app/ingest/` ingestion pipeline.
- `alembic/versions/` migrations `0001`–`0004` (head `0004_run_trace_events`). `data/manuals.json` shared manifest (backend + `src/manuals.ts`).
- `src/` UI (`ChatWorkspace`, `AdminWorkspace`, `RoutingPanel`, `Citations`, `legacyImport`). `tests/` pytest. `eval/` live-stack evaluation scripts. `docs/adr/` ADRs 0001–0009. `data/eval/` results (git-ignored), `logs/api.log`, `.run/` (launcher state/logs).

## Development Commands
Two interpreters: `.venv` (API deps, pytest, ruff, alembic — no torch) and `.venv-ingest` (GPU: torch, Docling, sentence-transformers, langchain-ollama/tavily; runs the real API, ingestion, agent CLI, eval).
```sh
python start.py [--no-ui] [--quiet]     # Postgres + API + UI, prints test questions; python stop.py [--all] [--force]
.venv-ingest/Scripts/python.exe run.py  # API only (127.0.0.1:8000, single worker); npm run dev | build | preview
.venv/Scripts/python.exe -m alembic upgrade head && .venv/Scripts/python.exe -m app.db.bootstrap
.venv/Scripts/python.exe -m pytest tests -q
.venv/Scripts/python.exe -m ruff format app tests alembic run.py   # then: ruff check app tests alembic run.py (README scope; do not widen)
npx prettier --check src && npm run build                  # build (tsc -b) is the only frontend type-check
PYTHONPATH=. .venv-ingest/Scripts/python.exe -m app.rag.agent "question naming the city"   # one graph run
.venv-ingest/Scripts/python.exe -m app.rag.trace_report --last 3 --verbose                  # stored run traces
.venv-ingest/Scripts/python.exe -m app.ingest.run --document-id drainage --ocr full-page [--rebuild]
docker compose exec postgres psql -U civil_chatbot_owner -d civil_chatbot
```
Docker CLI fallback: `C:/Program Files/Docker/Docker/resources/bin/docker.exe`. Prereqs: Docker Desktop, Ollama with exact tags `qwen3:8b` and `qwen2.5:7b`, `.env` from `.env.example`.

## Code Conventions & Common Patterns
- Python 3.11, `from __future__ import annotations`, module docstring per file, ruff line length 120 (lint selects only F/E9, but `ruff format` must stay clean). Prettier defaults for `src/`.
- **Graph nodes**: plain `node(state: AgentState) -> dict` returning only changed keys; registered via `graph.add_node(name, traced(name, fn))`; routers are `after_*` with `-> Literal[...]` and an explicit mapping dict. State is never mutated: `add_trace(state, msg)` and `add_event / add_llm_event / retrieval_event` (`app/rag/trace.py`) return new lists.
- **Lazy clients** as module globals behind accessors (`llm()`, `reviewer_llm()`, `web_search_tool()`, `sessions()`, `embedder()`); heavy imports happen inside them so `.venv` can import `app.*`. Tests monkeypatch these accessors.
- **Structured output**: `llm().with_structured_output(Schema, method="json_schema")`; schemas in `app/rag/agent_state.py`; `constrain_labels` builds `Literal` enums of real evidence labels. Add any new schema to `app/db/bootstrap.STRUCTURED_SCHEMAS`.
- **Evidence by reference**: evidence is split into labelled units (`[c3.s2]`); the model returns labels and the system attaches verbatim quotes (`resolve_labels`). Never ask a model to copy quotes. Verdicts, route (from `ROUTE_OF_REASON`) and gating are decided in code, not by the model.
- **Failure policy**: failures never become guesses. Router failure → `source_used="error"`, never `invalid`. Model/parse failure → `model_failure()` → `insufficient`. All HTTP failure codes live in the single `FAILURES` table in `app/core/errors.py`; raise `ServiceError("code", ...)`, never hard-code statuses.
- **Fixed strings** (`CITY_CLARIFICATION`, `REGIME_CLARIFICATION_PREFIX`, `BLOCKED_MESSAGE`, …) are matched exactly to detect follow-ups; edit with care.
- **No domain literals** in agent code (no Zone/Murfin/case ids); regime names come from retrieved evidence.
- **Policy version** = `agent-<sha256(app/rag/agent.py)[:12]>`; any edit to `agent.py` creates a new policy row. A schema change under the same hash raises.
- **Async/DB**: `create_resources(url)` → `async_sessionmaker(expire_on_commit=False)`; units of work via `async with sessions.begin()`; row locks `with_for_update()`; UTC via `views.utcnow()`; blocking embed/rerank via `asyncio.to_thread`. DI seam: `create_app(settings, *, ask=None)`; `ask(question, history, *, scope=None) -> dict` (fakes must accept `scope`).
- **Config**: `pydantic_settings.BaseSettings` in `app/core/config.py`, field `ollama_num_ctx` ⇔ env `OLLAMA_NUM_CTX`, `get_settings()` cached, unknown env ignored. Never read or print `.env`.
- **DB**: schema changes only via a new Alembic revision (never edit applied ones; `0001` is frozen explicit DDL); update `app/db/models.py` in lockstep. Audit tables are append-only: `routing_runs` finalized once, `routing_evaluations` append, `run_trace_events` ordered by `(run_id, seq)`; no UPDATE/DELETE of audit rows in app code. Constraint names `ck_/uq_/ix_<table>_…`.
- **Frontend**: wire types stay **snake_case** (no camelCase adapter); merge messages by id and sequence; `send()` captures destination id/scope/content before any `await` and updates via functional setState; keep both workspaces mounted (`hidden`); colors only via CSS variables in `src/styles.css`; flat prefixed class names (`chat-*`, `admin-*`, `routing-*`).
- Vocabulary (GLOSSARY.md): Engineer, Conversation, Research question, Source / Source version, Provision, Evidence, Regime, Clarification, Evidence gap, Research outcome, Blocked message, Routing run. Outcomes: `kb_answer, web_answer, clarification, blocked, insufficient, direct, error`.

## Important Files
`run.py`, `start.py`, `stop.py` (entry/launch) · `app/main.py` (`create_app`, lifespan, preload, orphan-run sweep) · `app/rag/agent.py` (graph, `ask`) · `app/rag/agent_state.py` · `app/rag/retrieval.py` · `app/services/turns.py` · `app/core/config.py`, `errors.py` · `app/schemas.py` (wire DTOs, `Contract` extra=forbid) · `data/manuals.json` · `src/api.ts` · `src/ChatWorkspace.tsx` · `src/RoutingPanel.tsx` · `vite.config.ts` · `compose.yaml` · `GLOSSARY.md` · `docs/adr/`.

**Do not touch**: `.env`, `docs/*.pdf`, `System_design.*`, `create_project.py` (scaffolder; do not run), empty `static/ templates/ uploads/`, `src/analytics.ts` (fixed demo data; never feed live data into it), applied migrations, Docker volume `postgres_data`. Never write to the live `civil_chatbot` DB except via migrations/bootstrap/ingest scripts.

## Runtime/Tooling Preferences
Python 3.11 venvs as above (torch must be installed first from the cu128 index; see header of `requirements-ingest.txt`); Node 22+ with npm (`package-lock.json`); Docker Desktop (WSL2) for Postgres only; Ollama runs natively (do not change its server options). One API worker only (`workers=1`; orphan-run recovery depends on it). Windows launch uses `CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP` (`DETACHED_PROCESS` kills the GPU Python). GPU is 10 GB shared by Ollama, Harrier, and the reranker (knobs: `OLLAMA_NUM_CTX`, `OLLAMA_KEEP_ALIVE`, `REVIEWER_NUM_GPU`, `RERANKER_BATCH_SIZE`; ADR 0009). Postgres regex word boundary is `\y`, not `\b`.

## Testing & QA
- pytest 9 + pytest-asyncio (strict mode: use `pytestmark = pytest.mark.asyncio` / `@pytest_asyncio.fixture`); no conftest. DB tests use `civil_chatbot_test` only (each fixture asserts `postgres_db == "civil_chatbot_test"`). Prepare it once: `CREATE DATABASE civil_chatbot_test`, then `alembic upgrade head` and `app.db.bootstrap` with `POSTGRES_DB=civil_chatbot_test`.
- **Run one pytest process at a time**: concurrent runs share the DB and fail with `database_unavailable: run no longer pending`; wait and rerun once.
- Seams: graph tests use scripted `FakeLLM`/`FakeWeb` + monkeypatched accessors (`tests/test_agent_graph.py`); API tests use `create_app(settings, ask=Agent)` with `httpx.ASGITransport` (`tests/test_routing_api.py`; register new run/conversation-referencing tables in its FK-ordered cleanup); schema tests roll back a transaction.
- Do not test model quality, ranking quality, wiring/mock echoes, or latency in pytest; do not seed fake passages in the live DB. Changing `eval/golden.json` counts requires updating `tests/test_golden_set.py`.
- **Live evaluation (not pytest; needs Ollama + GPU + live DB; run with `PYTHONPATH=.` in `.venv-ingest`)**: `eval/run.py [--list|--ids a,b|--limit N]` (40 agent cases, ~10–25 min full), `eval/retrieval_eval.py [--source golden|eval] [--compare <json>]` (retriever only, ~1 min), `eval/golden_e2e.py [--city-followup --limit 24]`. Results go to `data/eval/*.json` (compare against the newest file there; scores vary run to run at non-zero temperature). Retrieval changes must not regress golden R@8=100 and R@1≥~82; changing retrieval/GPU/routing defaults needs a new ADR.
- Behaviour changes: update `README.md` (GPU table, retrieval numbers, code layout), `.env.example` for new settings, `start.py` instruction block, and `GLOSSARY.md` for new domain terms.
