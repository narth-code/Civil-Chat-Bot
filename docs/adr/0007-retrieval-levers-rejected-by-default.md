# 0007. Retrieval levers measured one at a time; none kept as default

Status: accepted; the default decision is superseded in part by ADR 0008 (contextual vectors and a reranked head are now the default)

## Context

ADR 0006 ranks chunks globally on vector similarity, with the golden set (90 cases) at R@1 82.2, R@3 94.4, R@8 100.
The kb_answer eval cases (25, from `data/eval/20261010T012841442673Z.json`) reach only raw top-5 21/25 and R@8 92.
The target was 25/25 expected provisions in the top 5 on the eval source, with golden R@1 >= 85 and R@8 = 100.

Four levers were measured, each alone against the same baseline, on both sources (`eval/retrieval-experiments.md`, rows 6a to 9):

1. Contextual embedding text (`ctx1`: manual label > provision chain > chunk text; `ctx2`: plus `(unnumbered text)`
   chunks attached to the nearest preceding numbered provision by physical page).
2. Provision-title embedding index fused into RRF as a third leg (weights 1.0 and 0.3).
3. Cross-encoder reranker `BAAI/bge-reranker-v2-m3` over the fused top 30.
4. Abbreviation expansion from the manuals' own `X (LONG FORM)` definitions (224 derived).

## Decision

No lever becomes a default. The default retrieval path is unchanged: baseline Harrier vectors, FTS weight 0, no title leg, no reranker, no expansion.

- Lever 1 (ctx1, ctx2): golden R@1 87.8, R@3 95.6, R@5 98.9, R@8 98.9. TRN-25 drops out of the top 8, so the R@8 gate fails by one case. Eval raw top-5 improves to 23/25. ctx2 is identical to ctx1 on every metric, so the `(unnumbered text)` attachment changes no measured case. Vectors are stored under `<harrier-revision>#ctx1` and `#ctx2`, and the revision tag is selected by `RETRIEVAL_EMBEDDING_VARIANT`. Old vectors remain.
- Lever 2 (title index): golden R@1 67.8 at weight 1.0 and 75.6 at weight 0.3; eval raw top-5 11/25 and 15/25. Provision-level heading matches push chunks of generic headings above the precise chunk. Code removed.
- Lever 3 (reranker): golden R@1 88.9, R@3 96.7, R@8 97.8 (FIR-05 drops out). Eval raw R@1 92, top-5 24/25. The cost is about 3.2 s per query for 30 passages at 512 tokens (p50 3193.7 ms, p95 3282.8 ms). Peak allocated VRAM 1.39 GiB (reserved 1.59 GiB; weights 1.06 GiB after load). That is too slow for a default path.
- Lever 4 (abbreviations): golden R@1 81.1, below the 82 gate. Eval top-5 21/25 unchanged.
- Combination ctx1 plus reranker: golden R@1 88.9, R@3 96.7, R@8 97.8. Identical to reranker alone, so the R@8 gate still fails.

## Consequences

- The default configuration is unchanged: eval raw top-5 21/25, R@8 92; golden R@1 82.2, R@8 100. The 25/25 target is not met.
- Default path latency on the eval source is raw p50 81.8 ms, p95 97.5 ms (`data/eval/retrieval-eval-20261010T024901Z.json`).
- The opt-in toggles (`RETRIEVAL_EMBEDDING_VARIANT`, `RETRIEVAL_RERANK`, `RETRIEVAL_EXPAND_ABBREVIATIONS`) remain for measurement only. They are not validated for production use.
- A reranker over a smaller pool, or a retrieval change that recovers TRN-25 and FIR-05, must be measured against the same gate before it can become a default.
- No migration is required: no retrieval table was kept.
