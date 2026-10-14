# Civil reference workspace

Desktop React + TypeScript interface backed by a local FastAPI service, a local Ollama model, and a local PostgreSQL database. The current milestone **classifies each request and saves the conversation and routing decision**. It does not retrieve passages and does not generate engineering or conversational answers.

## Requirements

- Node.js 22+, Python 3.11, Docker Desktop (WSL2 backend), Ollama with the exact `qwen3:8b` tag.

## Setup

```sh
ollama pull qwen3:8b

python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements-dev.txt
npm install

copy .env.example .env        # then set a strong POSTGRES_PASSWORD; keep existing keys
docker compose up -d --wait   # pgvector/pgvector:0.8.7-pg17 on 127.0.0.1:5432
.venv/Scripts/python.exe -m alembic upgrade head
.venv/Scripts/python.exe -m app.db.bootstrap
```

`bootstrap` registers the three manuals from `data/manuals.json`, captures each PDF from `docs/` into an immutable, SHA-256-addressed copy under `data/source_versions/` (ignored by git), and stores the routing policy. It is idempotent. The original files in `docs/` are never modified.

## Run

```sh
python start.py                      # Postgres + API (GPU environment) + UI, then prints how to try it
python stop.py                       # stop the API and UI (--all also stops Postgres)

# or by hand:
.venv-ingest/Scripts/python.exe run.py   # API, 127.0.0.1:8000, one worker (loads Harrier, so the GPU environment)
npm run dev                              # UI, proxies /api to the backend
npm run build && npm run preview         # built UI, same proxy
```

Open the URL Vite prints. `GET /api/health` reports database and model readiness (503 if either is unavailable).

### GPU memory

The answer path keeps `qwen3:8b` (about 6.3 GB at an 8192-token context), the Harrier embedder, and the bge reranker on the GPU. In the 40-turn stress loop the GPU peaked at 9325 MiB of 10240 MiB. `start.py` warns before starting when the GPU memory available to the app (free memory plus models Ollama already holds) is below that peak.

| Setting (`.env` or `app/core/config.py`) | Default | Effect |
|---|---|---|
| `OLLAMA_NUM_CTX` | 8192 | Context window requested for `qwen3:8b` (was 12288). Ollama truncates prompts longer than this and the app does not detect it. The largest prompt in the stress loop was 7309 tokens. |
| `OLLAMA_KEEP_ALIVE` | -1 | Models stay loaded instead of being unloaded after five idle minutes. |
| `REVIEWER_NUM_GPU` | 0 | The monitoring-only reviewer (`qwen2.5:7b`) runs on the CPU, so it never competes with `qwen3:8b` for VRAM. Trade-off: `review_answer` takes about 10.7 s instead of 6.2 s. |
| `RERANKER_BATCH_SIZE` | 8 | Candidates scored per forward pass; the CUDA cache is emptied afterwards. Golden retrieval unchanged (R@8 100, R@3 96.7). |

Stress loop (10 representative questions, four passes, `ask()`): mean turn 24.6 s before, 13.7 s after; median 19.4 s before, 13.7 s after; zero Ollama errors in both runs. The first answer after a cold model load took 53 s. Neither loop reproduced the earlier CUDA faults, so the VRAM and latency figures are the verified change; a lower crash rate is not demonstrated.

Ollama server options that would lower memory further (`OLLAMA_FLASH_ATTENTION`, KV-cache quantization, `OLLAMA_MAX_LOADED_MODELS`) are not set by this project because they change the user's Ollama service. They are listed here only.

### Looking at the database

Connect any client (DBeaver, pgAdmin) to `127.0.0.1:5432`, database `civil_chatbot`, user `civil_chatbot_owner`, password = `POSTGRES_PASSWORD` in `.env`. Or from a terminal:

```sh
docker compose exec postgres psql -U civil_chatbot_owner -d civil_chatbot
```

Useful queries:

```sql
-- every question: outcome, city, timing, the exact web query and chunks retrieved (trace), citations
select created_at, outcome, city, error_code, elapsed_ms, trace, citations, review
from routing_runs order by created_at desc limit 20;
-- what was ingested
select document_id, count(*) from chunks c join document_versions v on v.id = c.source_version_id group by 1;
```

Logs: `logs/api.log` (application), `.run/api.out.log`, `.run/ui.out.log`. One question without the UI: `.venv-ingest\Scripts\python.exe -m app.rag.agent "question naming the city"`.

### Run traces

`run_trace_events` is append-only, ordered by `(run_id, seq)`, and keeps structured node, model-call, retrieval, web-search, and decision events alongside the existing human-readable `routing_runs.trace`. Inspect one run at `GET /api/routing-runs/{run_id}/trace`, or print recent traces with `.venv-ingest/Scripts/python.exe -m app.rag.trace_report --last 1 --verbose`.

`data` holds event-specific JSON; for example, retrieval chunk ranks can be queried with:

```sql
select e.seq, e.node, e.data->'chunks' as chunks
from run_trace_events e
where e.run_id = '00000000-0000-0000-0000-000000000000'
order by e.seq;
```

## Routing

Each message is classified once by `qwen3:8b` (`temperature 0`, structured JSON output, no tools) through a single-node LangGraph (`START -> route -> END`).

| Route | Reasons | Meaning |
|---|---|---|
| `retrieval` | `code_lookup` | A civil code or reference lookup. Retrieval is **not connected**. |
| `conversation` | `greeting`, `clarification`, `workspace_help` | Not a lookup. A responder is **not connected**. |
| `invalid` | `unsafe_request`, `prompt_injection`, `off_topic` | Refused or out of scope. |

Earlier messages in the conversation are used as context (for example a location given in a prior turn). Historical text, including imported demo replies, is treated as untrusted context. The reply shown in the chat is a fixed status notice, not a generated answer, and never claims a rainfall zone, section, or citation.

A model, network, or output-validation failure is recorded as a **failed** run. It is never converted into `invalid`, never retried, and never routed with another model. If the full conversation exceeds 12,000 UTF-8 bytes of provider request, the run fails with `context_limit` before any model call and the user is asked to start a new focused conversation. Nothing is truncated or summarized.

## Data and audit handling

PostgreSQL owns saved chats. Message text, the exact ordered transcript sent to the model, selected manual scope, model tag and digest, options, policy version, timings, and human labels are stored locally and visible in Admin. There is no authentication; the Engineer/Admin switch is a development control only. Do not treat this as a secure project record system.

Browser `localStorage` keeps the theme and, temporarily, a one-time import of chats saved by earlier versions (`civil-reference-chats-v1`). Import is all-or-nothing, preserves order, timestamps, and manual scope, marks old assistant replies `legacy_demo`, and removes the browser copy only after the backend reload shows the imported chats. Malformed data is left untouched with an error.

### Storage vocabulary

`documents` / `document_versions` (immutable source snapshot) / `provisions` (flexible parent-child hierarchy, exact full identifier) / `chunks` (exact text, page provenance, quality flags) / `chunk_embeddings` (explicit model, revision, dimensions) / `conversations` / `messages` / `routing_policies` / `routing_runs` (+ ordered context and source tables) / `routing_evaluations` (append-only human labels). The provision, chunk, and embedding tables are filled by the ingestion prototype (below). Routing does not read them.

Manual edition, effective dates, and applicability are stored as unknown (`null`). They are not inferred from filenames; confirm the governing edition in the original PDF.

## Admin space

- **Live routing** (local database): UTC 7/30/90-day window, run counts by route and invalid reason, median/P95 routing latency from completed model attempts, run history with exact transcript, scope, model digest, options, and policy. Record a human expected route per run; labels are appended, never overwrite the model decision. Agreement uses the latest label per classified run and shows "Not evaluated" when nothing is reviewed. These figures do not measure retrieval, answer correctness, or safety.
- **Retrieval performance and activity** are labelled demonstration data for the fixed period April 2 to June 30, 2026 and are independent of live runs. Real retrieval and answer metrics are unavailable.

## Tests

```sh
.venv/Scripts/python.exe -m pytest tests -q
```

The suite uses the real compiled graph, real PostgreSQL, and a controlled HTTP transport in place of Ollama. It runs only against a separate `civil_chatbot_test` database, which must exist and be migrated:

```sh
docker compose exec postgres psql -U civil_chatbot_owner -d postgres -c "CREATE DATABASE civil_chatbot_test"
# then run alembic upgrade head and app.db.bootstrap with POSTGRES_DB=civil_chatbot_test
```

Formatting is mechanical and checked, not argued about in review:

```sh
.venv/Scripts/python.exe -m ruff format app tests alembic run.py   # Python, line length 120 (pyproject.toml)
.venv/Scripts/python.exe -m ruff check app tests alembic run.py    # unused and undefined names
npx prettier --check src                                           # frontend
```

## Document ingestion (prototype)

Scanned or text PDFs are processed by a separate GPU environment so the API's pinned dependencies stay untouched. Build it once (commands are at the top of `requirements-ingest.txt`), then run, for example:

```sh
# image-only or untrustworthy text layers: OCR every page
.venv-ingest/Scripts/python.exe -m app.ingest.run --document-id fire --document-id drainage --ocr full-page
# born-digital PDFs: keep the PDF's own text where a page has it, OCR only the rest
.venv-ingest/Scripts/python.exe -m app.ingest.run --document-id transportation --ocr auto
.venv-ingest/Scripts/python.exe -m app.ingest.search "design rainfall south of the Colorado River" --document-id drainage
```

The run converts pages in resumable 20-page batches (Docling with RapidOCR on the GPU), builds Provisions from the printed section identifiers, chunks the text, embeds with Harrier (falling back to `nomic-embed-text` only if Harrier cannot load, and logging it), and loads PostgreSQL. Progress is in `data/ingest/<sha256>/status.json` and `ingest.log`. `--pages` runs are refused against the live database unless `--allow-partial` is given. Re-running never duplicates rows; `--rebuild` replaces only that source version's generated rows, and `--reconvert` discards that document's cached page batches.

Every chunk carries `extraction_warnings`, one string per problem tied to a page it covers: `blurry_image`, `low_ocr_confidence`, `garbled_text`, `empty_page`, `no_text_extracted`, `ocr_text_layer_mismatch` (OCR disagrees with the PDF's own text), or a `docling_error`. Chunks also record how their text was obtained (`extraction_method`) and the mean OCR confidence. Problems that cannot be tied to a page are kept in `data/ingest/<sha256>/quality.json` and the log, not on chunks. Add a manual by adding it to `data/manuals.json`, running `python -m app.db.bootstrap`, then the commands above.

## Retrieval

`app/rag/retrieval.py` implements `hybrid_search`, which the agent's `retrieve_kb` calls. It ranks chunks by cosine distance (`<=>`) over `chunk_embeddings` (Harrier, 1024 dimensions) in one list across all current manuals. Each result carries `chunk_id`, `vector_rank`, `fts_rank` (null when the full-text leg is off), `rrf_score`, and `found_by`. Scoped calls (`document_ids`) restrict candidates to those manuals.

- Ranking: the top 50 vector candidates are fused with reciprocal rank fusion (k = 60). Unscoped searches have no per-manual quota; the highest-scoring chunks win regardless of manual.
- Duplicate-marked provisions (`<section> (duplicate N)`, 130 current provisions) are kept as stored. Retrieval reports them under the base section number and sets `is_duplicate`. Nothing was rebuilt or re-OCR'd.
- `hybrid_search(..., expand_neighbors=True)` appends the ordinal-adjacent chunks of the same provision after the primary hits (`neighbor_of` set, `found_by` empty). Off by default; the primary `k` is unchanged.
- Full-text leg: `RETRIEVAL_FTS_WEIGHT` (default 0.0). A positive value adds an AND query over the question's alphanumeric terms. On the golden set every positive weight lowered R@1 and MRR, so the default is vector-only.
- Default path (ADR 0008): contextual embedding text (`RETRIEVAL_EMBEDDING_VARIANT=ctx1`, stored under `<harrier-revision>#ctx1`) and a reranker (`RETRIEVAL_RERANK=true`, `BAAI/bge-reranker-v2-m3`) over the fused top 30. The reranker's top 4 lead the result; the remaining candidates keep their fused order. The reranker runs on the GPU, is preloaded at startup, and falls back to fused order if it cannot run. Setting `RETRIEVAL_EMBEDDING_VARIANT=` and `RETRIEVAL_RERANK=false` restores the ADR 0007 baseline.
- Opt-in (off by default; measured per ADR 0007): `RETRIEVAL_EMBEDDING_VARIANT=ctx2` (no measured difference from ctx1), `RETRIEVAL_EXPAND_ABBREVIATIONS=true` (golden R@1 below the gate). A title index was tried and removed.
- Eval-source (25 kb_answer cases), default path: raw R@1 92, R@3 92, R@5 96, R@8 96, top-5 24/25 (ADR 0007 baseline 21/25); agent-constructed R@1 88, top-5 23/25 (`data/eval/retrieval-eval-20261010T032823Z.json`). The 25/25 top-5 target is not met.

Evaluation: `eval/retrieval_eval.py` runs `eval/golden.json` (90 provision cases, 8 negatives) with k = 8 on the raw question, and `--compare` diffs against a previous JSON. Latest run (`data/eval/retrieval-20261010T032656Z.json`): all 90 cases R@1 88.9, R@3 96.7, R@5 98.9, R@8 100.0, fact-hit 99.0, MRR 93.4. All 8 negatives absent.

```sh
PYTHONPATH="." .venv-ingest/Scripts/python.exe eval/retrieval_eval.py --compare data/eval/retrieval-20261010T010541Z.json
```

Latency: the reranker scores the top 30 candidates on the GPU for each query. The eval-source run `data/eval/retrieval-eval-20261010T032823Z.json` measured raw p50 692 ms, p95 865 ms (the pre-ADR-0008 baseline path was p50 85 ms). The reranker weights take about 1.1 GiB of GPU memory.

## Code layout

Backend (`app/`), one job per folder:

- `core/`: `config.py` settings; `errors.py` the single failure vocabulary (code, HTTP status, message); `manuals.py` the manual catalog and snapshot paths.
- `schemas.py`: wire contracts (snake_case) shared with the UI.
- `db/`: `models.py` tables and constraints, `session.py` engine factory, `snapshots.py` immutable PDF capture, `bootstrap.py` registers manuals and the routing policy. Migrations are in `alembic/`.
- `rag/`: `policy.py` the versioned routing prompt, `router.py` the Ollama adapter, `graph.py` the single-node LangGraph.
- `services/`: `turns.py` one user turn (pending run, graph call, audit), `conversations.py` conversations and import, `audit.py` routing history and human evaluations, `metrics.py` the aggregate queries behind the admin summary, `views.py` shared projections.
- `api/`: HTTP routes (`conversations.py`, `routing.py`, `manuals.py`, `health.py`), `deps.py`, `errors.py`. `main.py` only wires them together (`create_app(settings)`).
- `ingest/`: `convert.py` OCR to blocks, `structure.py` Provisions and chunks, `embedding.py`, `load.py`, `run.py`, `search.py`.

Frontend (`src/`): `api.ts` typed client; `ChatWorkspace.tsx` chat; `legacyImport.ts` the one-time browser-history import (deletable once old browser data is gone); `sampleConversation.ts` the labelled demo; `AdminWorkspace.tsx`, `RoutingPanel.tsx`, `analytics.ts` (demo data). `data/manuals.json` is the manifest shared by backend and frontend.
