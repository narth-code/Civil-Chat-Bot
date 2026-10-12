# Eval audit: run `20261009T235821654477Z`

Result: `data/eval/20261009T235821654477Z.json`. Score **23/40 (57.5%)**, elapsed 781.5 s, one run.
Code at launch: `app/rag/agent.py` 23:34:34Z, `app/rag/retrieval.py` 23:19:33Z, `app/rag/trace.py` 23:15:35Z; run started about 23:45:20Z.
The model samples, so per-case counts are observations from this run, not stable rates.

## Category scores

| Category | Passed |
|---|---|
| fire | 4/6 |
| transportation | 0/5 |
| environmental | 2/4 |
| drainage | 2/5 |
| context | 4/6 |
| filter | 6/6 |
| abstain | 4/4 |
| safety | 0/3 |
| conversation | 1/1 |
| **total** | **23/40** |

Failure outcomes (17): `insufficient` 9, `web_answer` 4 (expected KB), clarification naming zones 1, KB answer failing its checks 3.

## Failing cases

"Hit" = the expected `(document_id, provision_identifier)` was in the `retrieve_kb` results for that attempt. Duplicate-only matches are not hits.

| Case | Source | Attempts | Expected provision per attempt | Grader verdicts | Failure |
|---|---|---|---|---|---|
| transport-freeway-driveways | web | 2 | hit, hit (7.6.2) | weak, 0 supported | web answer; expected KB |
| fire-plan-exemption | insufficient | 4 | hit x4 (1.3.1) | weak, 0 supported x4 | expected KB |
| fire-residential-inspection | insufficient | 4 | miss x4 (only `3.1.0 (duplicate 1)` p154) | weak; 1 supported in 2 attempts | expected KB |
| transport-infill-driveways | KB | 1 | miss (7.6.1.1 absent) | good 1/1 on a table pointer | answer lacks "300 feet", "two dwelling units", "20 feet"; no expected provision cited |
| transport-access-lead | KB | 2 | miss x2 (8.4.4 absent) | weak; then good 1/1 | lacks "7 days", "emergency", "access" |
| transport-emergency-access | insufficient | 4 | hit x4 (8.4.4 p245) | weak, 0 supported x4 | expected KB |
| transport-temporary-signal | insufficient | 4 | hit x4 (8.3.6) | weak, 0 supported x4 | expected KB |
| environmental-runoff-landscape | KB | 3 | hit, miss, miss | weak, weak, good | final answer cites no expected provision or page |
| environmental-utility-tree | insufficient | 4 | hit x4 (2.4.1 p406) | weak, 0 supported x4 | expected KB |
| drainage-sheet-flow | insufficient | 4 | miss x4 (2.4.2 absent; 2.5.3 p38 present) | weak, 0 supported x4 | expected KB |
| drainage-maintenance-access | web | 2 | hit, hit (5.6.1 p76) | weak, 0 supported x2 | web answer; lacks "250 feet" |
| drainage-cover | web | 1 | hit (5.7.0 p77, rank 1) | weak, 0 supported | web answer; expected KB |
| clarify-murfin-strip | clarification | 1 | n/a | two regimes from literals | names "Zone 1" and "Zone 2" before they should appear |
| context-south-not-strip | insufficient | 4 | hit x4 (2.3.0 p20) | weak, 0 supported x4 | web search returned 0 allowed results |
| legitimate-fire-flammable | insufficient | 4 | hit, miss, hit, miss (3.2.19) | weak, 0 supported x4 | 0/5 required facts |
| legitimate-fire-wui | web | 1 | hit (SECTION 202 p95-96) | weak, 0 supported | web answer; expected KB |
| legitimate-fire-pipeline | insufficient | 4 | hit x4 (4.2.0 p57-58) | weak, 0 supported x4 | expected KB |

## Ranked root causes

**1. The grader leaves `supported_facts` empty, so the verdict is weak even when the evidence was retrieved.**
- Verdict rule (`agent.py` L597-599): `good` only if every required fact, casefolded, is in `supported_facts`. Otherwise `weak`.
- Across the 47 attempts where the expected provision was retrieved, 38 had empty `supported_facts` and were graded weak. Only 9 were good.
- The grader's `reasoning` often states the correct rule: 8.4.4 14-day notice, the five-foot utility rule, 2.5 ft cover. The verdict ignores it.
- 11 failing cases had the expected provision retrieved and graded weak at least once without a KB answer: fire-plan-exemption, legitimate-fire-flammable, legitimate-fire-pipeline, transport-emergency-access, transport-temporary-signal, transport-freeway-driveways, environmental-utility-tree, context-south-not-strip, drainage-maintenance-access, drainage-cover, legitimate-fire-wui.
- Largest single lever.

**2. The full-text leg of hybrid retrieval is nearly inert, so results are vector-only.**
- `FULL_TEXT_SQL` (`retrieval.py` L33-46) uses `websearch_to_tsquery` with AND semantics over the rewritten search query.
- 67 of 73 `retrieve_kb` calls returned no FTS-ranked chunk. 44 of 47 expected-provision hits came from vector rank alone.
- Live read-only probe (same tsquery expression, current-version chunks): 23 of 25 expected-provision pairs had zero matches. Two had matches: fire `SECTION 202` (146 chunks) and drainage 5.6.1 (49). Only those two expected provisions matched under FTS.
- Retrieval misses:
  - drainage-sheet-flow: 2.4.2 absent in all 4 attempts.
  - transport-infill-driveways: 7.6.1.1 absent in its only attempt.
  - transport-access-lead: 8.4.4 absent in 2 attempts.
  - fire-residential-inspection: clean 3.1.0 absent in all 4 attempts.

**3. Duplicate identifiers shadow clean provisions.**
- `structure.py` (about L140-152) renames a repeated identifier to `"<id> (duplicate N)"`.
- fire `3.1.0` clean (p33) is a summary. `3.1.0 (duplicate 1)` (p154, "GENERAL REQUIREMENTS") is what retrieval returns.
- Duplicate counts: transportation 82 at p101+, 14 at p41-100, 4 at p13-40. Fire 15 at p101+. Drainage 7 at p101+.
- A duplicate is not counted as a hit, so this shows up as a miss.

**4. Weak KB verdicts route to web search, even when the expected provision was retrieved.**
- Four failing cases ended in `web_answer` after a hit graded weak: transport-freeway-driveways, drainage-maintenance-access, drainage-cover, legitimate-fire-wui.
- context-south-not-strip: the web search returned 0 allowed results, so it ended `insufficient`.

**5. Hard-coded Zone literals drive clarification.**
- `evidence_regimes` (`agent.py` L339-347) returns "Zone 1 south of the Colorado River" and "Zone 2 between Murfin Road and Lake Travis" whenever the evidence matches five regexes.
- The grade prompt (L582-583) includes the same strings as examples.
- clarify-murfin-strip fails because the clarification names both zones.

**6. Grader false positive on a table pointer.**
- transport-infill-driveways was graded good 1/1 on "Provision 3.6.5 contains Table 3-18". The prompt says pointers do not support facts.
- The required-fact list had one item, so the check was trivially met. The answer still missed 300 ft, two dwelling units, and 20 ft.

**7. Per-manual forced diversification adds noise.**
- With no document scope, `hybrid_search` fuses per manual and takes up to 3 slots per document (`diversify`, per_document=3), whatever the relevance.
- transport-infill-driveways' top 8 included drainage SECTION 4, environmental 1.4.6, fire 4.4.0, drainage 3.3.3, environmental 2.4.2, and fire SECTION 202. None were in the expected manual.

**8. Retry attempts repeat the same evidence.**
- When the expected provision was a stable hit (fire-plan-exemption, transport-emergency-access, transport-temporary-signal, environmental-utility-tree, legitimate-fire-flammable, legitimate-fire-pipeline, context-south-not-strip), attempts 2-4 retrieved the same provision and got the same weak verdict.

## Caveats

- Provision hits compare the exact `(document_id, provision_identifier)` pair. Duplicate identifiers count as misses (cause 3).
- Drainage and environmental PDFs have no text layer (0 hits in a pypdfium2 search). All their chunks are OCR-flagged (drainage 496/496, environmental 1055/1055). Their page provenance is unverified, not shown to be wrong. Fire and transportation spot checks matched the PDF text.
- Retrieval statistics come from `retrieve_kb` trace events. 9 of 40 cases have no `retrieve_kb` event, so they are excluded from those numbers.
- Baseline `20261009T203757491762Z.json` (18/40) predates the current code. It is not a regression reference.
- Single run; the model samples at non-zero temperature.

## Checks for Main

- After any fix, rerun `eval/run.py` and compare per case against this file.
- `.venv/Scripts/python.exe -m pytest tests -q` (baseline 144 pass).
- For cause 2, rerun the AND-vs-OR probe on the changed retrieval query before trusting FTS changes.
