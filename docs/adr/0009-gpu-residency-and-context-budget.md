# 0009. Keep answer models resident, cap the context, and keep the reviewer on the CPU

Status: accepted. Changes generation-runtime defaults only. It does not change retrieval (ADR 0006 to 0008), the agent graph, or the models.

## Context

On the 10 GB RTX 3080 the answer model (`qwen3:8b`), the monitoring reviewer (`qwen2.5:7b`), the Harrier embedder, and the bge reranker share one GPU. Ollama swapped the two chat models in and out of VRAM, and every swap reloaded weights. The per-node means showed the cost: `route_question` averaged 6.0 s and `grade_kb` 5.9 s, most of it reload time.

Ollama logs from earlier sessions (`server.log`) contain a CUDA out-of-memory failure, a CUDA illegal memory access, and two `llama-server terminated` events. The answer model ran with a 12288-token context at that time.

Hypothesis, not proven: a 12288-token qwen3 context plus model swaps and reranker transient allocations led Ollama's free-VRAM estimate astray and produced those faults.

## Decision

1. `OLLAMA_KEEP_ALIVE=-1` (default). Ollama keeps models loaded instead of unloading them after five idle minutes.
2. `OLLAMA_NUM_CTX=8192` (was 12288). Requested context window for `qwen3:8b` answer calls.
3. `REVIEWER_NUM_GPU=0` (default). The monitoring-only reviewer runs on the CPU and never competes with `qwen3:8b` for VRAM. Setting it empty restores Ollama's placement.
4. `RERANKER_BATCH_SIZE=8`. The reranker scores the candidate pool in chunks under `torch.no_grad()` and empties the CUDA cache after scoring. Score order is preserved.
5. Ollama server tuning (`OLLAMA_FLASH_ATTENTION`, KV-cache quantization, `OLLAMA_MAX_LOADED_MODELS`) is documented in the README only. It changes the user's Ollama service, so it is not applied.

Unchanged: the models, prompts, retrieval configuration, and graph topology.

## Evidence

Same 40-turn loop before and after: 10 representative questions, four passes, through `ask()`. Route sources matched in 40/40 turns (kb 28, clarification 4, insufficient_evidence 4, blocked 4), so the latency difference is not caused by different routes.

| Measure | Before | After |
|---|---|---|
| Exceptions / crash lines / llama-server terminations | 0 / 0 / 0 | 0 / 0 / 0 |
| Mean turn | 24.6 s | 13.7 s |
| Median turn | 19.4 s | 13.7 s |
| P90 turn | 37.3 s | 25.5 s |
| Slowest turn | 139.8 s | 53.2 s |
| Peak GPU memory (`nvidia-smi`) | 9981 MiB | 9325 MiB |
| Largest prompt (tokens) | 5563 | 7309 |

The before run used `num_ctx` 12288, default keep-alive, the reviewer on the GPU, and no reranker batching, and it started with models already loaded. The after run used the configuration above and was a clean restart. Its slowest turn (53.2 s) is the first answer after a cold qwen3 load.

Per-node mean duration, in seconds, averaged over node invocations (the probe's `node_mean_s`):

| Node | Before | After |
|---|---|---|
| `route_question` | 6.03 | 0.86 |
| `retrieve_kb` | 1.68 | 0.79 |
| `extract_context` | 2.74 | 0.65 |
| `extract_regimes` | 0.001 | 0.001 |
| `grade_kb` | 5.89 | 2.34 |
| `generate_from_kb` | 4.96 | 2.33 |
| `review_answer` | 6.16 | 10.69 |
| `retry_kb` | 3.69 | 0.25 |

After-run invocation counts: `route_question` 40, `retrieve_kb` 40, `grade_kb` 40, `extract_context` 36, `generate_from_kb` 28, `review_answer` 28, `retry_kb` 8.

Server log for the after window: no matches for `truncated = 1`, `truncating input prompt`, `CUDA error`, `llama-server terminated`, `illegal memory`, or `out of memory`.

Golden retrieval after the reranker batching change (`data/eval/retrieval-20261010T041847Z.json`): R@1 88.9, R@3 96.7, R@5 98.9, R@8 100.0, fact-hit 99.0, MRR 93.4, negatives 8/8. These match the ADR 0008 chosen row, so batching did not change ranking.

Limits of this evidence:
- Neither 40-turn loop reproduced a crash (0 of 40 turns in each). The crash-rate reduction is **not demonstrated**. What is verified is the peak-VRAM reduction (656 MiB) and the latency reduction.
- The after window was checked for isolation only at its start (`ollama ps`, process list). Another GPU job was running later, so the window is not guaranteed to be free of other GPU load.

## Rejected or deferred

- Keeping `num_ctx` at 12288: it was the configuration in the historical faults and held the most VRAM. Rejected.
- Reviewer on the GPU: it forces model swaps against `qwen3:8b`. Rejected. Its cost is accepted, see Consequences.
- Ollama flash attention, KV-cache quantization, `OLLAMA_MAX_LOADED_MODELS`: deferred. They change the user's Ollama service. Documented in the README only.
- A token guard that refuses or trims prompts above `OLLAMA_NUM_CTX`: not implemented. It belongs in the agent prompt-building code, which another worker owns.
- Taking the review step off the answer path: a product decision, deferred to the user.

## Consequences

- **Truncation risk.** Ollama truncates prompts longer than `num_ctx`, and the app does not detect it. The largest measured prompt was 7309 tokens, leaving about 900 tokens of headroom. A long conversation or many provisions could exceed the window. Options are a token-count guard in agent code, or a larger context at a VRAM cost.
- **Residency.** `keep_alive=-1` keeps `qwen3:8b` (about 6.3 GB, GPU) and `qwen2.5:7b` (about 5.1 GB, CPU RAM) loaded until Ollama restarts or `ollama stop` runs.
- **Review latency.** `review_answer` is now the largest node at 10.7 s, because the reviewer runs on the CPU. The reviewer is monitoring-only and never gates an answer.
- **VRAM headroom.** The app peaked at 9325 MiB of 10240. `start.py` warns when GPU memory available to the app (free plus models Ollama already holds) falls below that peak.
- Changing these defaults again requires a new ADR that supersedes this one and measures against the same loop and gates.
