# 0008. Contextual vectors and a reranked head are the default retrieval path

Status: accepted. Supersedes the default decision of ADR 0007 (ADR 0006 still applies to global vector ranking).

## Context

ADR 0007 kept the baseline path (bare Harrier vectors, no reranker) because no single lever met the gate. The gate was golden R@8 = 100, golden R@1 >= 85, and 25/25 expected provisions in the top 5 on the eval-question source (25 kb cases). Under the baseline the eval source reached raw top-5 21/25.

Two levers help different parts of the problem:
- Contextual embedding text (`ctx1`) raises golden R@1 to 87.8 and eval top-5 to 23/25, but drops TRN-25 out of the golden top 8 (R@8 98.9).
- The cross-encoder reranker (`BAAI/bge-reranker-v2-m3` over the top 30) raises golden R@1 to 88.9 and eval top-5 to 24/25, but drops FIR-05 and TRN-11 out of the golden top 8 (R@8 97.8). Row 8 of `eval/retrieval-experiments.md`.

Reranking all 30 candidates and taking its order replaces the fused order wholesale, which loses those long-tail cases.

The user's priority is accurate retrieval of the provisions a generator relies on, and the generator consumes the top 4 to 8 provisions. A small loss at the far tail of the golden set is acceptable only if it does not remove any golden case from the top 8.

## Decision

The default retrieval path is:

1. Contextual vectors: `RETRIEVAL_EMBEDDING_VARIANT=ctx1` (default). Stored vectors are tagged `<harrier-revision>#ctx1`.
2. Candidate pool: the top `RETRIEVAL_RERANK_TOP_N` (30) fused vector candidates, in contextual vector order.
3. Reranking: `RETRIEVAL_RERANK=true` (default). The reranker scores the pool. Its top `RERANK_HEAD` (4) candidates lead the result, and every other pool candidate follows in its existing fused order (`rerank_head_order` in `app/rag/retrieval.py`). The result is then truncated to k.
4. Unchanged defaults: `RETRIEVAL_FTS_WEIGHT=0.0` (full-text leg off), `RETRIEVAL_EXPAND_ABBREVIATIONS=false`, `RETRIEVAL_TOP_K=8`.

Measured offline on the cached production vectors, chunks, and full-text output. The sweep reproduces the logged baseline, ctx1, and reranker rows (rows 0, 6a, 8) exactly.

| Configuration | Golden R@1 | R@3 | R@5 | R@8 | Fact-hit | Eval raw top-5 |
|---|---|---|---|---|---|---|
| Baseline (ADR 0006/0007 default) | 82.2 | 94.4 | 97.8 | 100.0 | 99.0 | 21/25 |
| ctx1 only | 87.8 | 95.6 | 98.9 | 98.9 | 98.1 | 23/25 |
| Reranker over fused top 30 (row 8) | 88.9 | 96.7 | 96.7 | 97.8 | 96.1 | 24/25 |
| Chosen: ctx1 + reranked head of 4 | 88.9 | 96.7 | 98.9 | 100.0 | 99.0 | 24/25 |

The chosen configuration keeps golden R@8 at 100 and fact-hit at 99.0, and raises eval raw top-5 from 21/25 to 24/25.

Rejected in the same sweep:
- Reciprocal-rank fusion of the reranker and vector lists, k=2: golden R@1 91.1 but R@5 97.8 (chosen: 98.9), eval agent top-5 23/25 (no gain). k=3: golden R@1 85.6.
- Reranking the whole pool without the head (row 8 order): drops FIR-05 and TRN-11 out of the golden top 8.

## Consequences

- Retrieval loads a reranker model on the GPU at startup (`app/main.py` background preload). The reranker weights take about 1.1 GiB of VRAM and share the GPU with Ollama. The production eval run (row 10) measured p50 692 ms and p95 865 ms per raw query. Row 8 measured 3.2 s p50 for 30 passages, and its eval run measured 3860 ms raw p50. The cause of the gap is not isolated; the production golden metrics match the reranker-on sweep exactly, so the reranker ran.
- The embedding revision used for search is `<harrier-revision>#ctx1`. Vectors for the bare and `#ctx2` revisions remain in the database but are not searched by default.
- The result order inside the top 4 comes from the reranker; positions 5 to k keep the fused contextual order.
- The 25/25 eval target is not met (24/25). Remaining misses are listed in `eval/retrieval-experiments.md` row 10.
- Changing the default again requires a new ADR that supersedes this one and measures against the same gates.
