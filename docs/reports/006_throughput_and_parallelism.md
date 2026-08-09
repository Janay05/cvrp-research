# Report 006 — Throughput, parallelism, and honest budgets

**Follows:** [005_cost_optimization.md](005_cost_optimization.md). That report closed two concurrency bugs and five wasted-search-budget defects, landing at Lazio +0.88% cost / 3.1x FILO2's speed and Valle-D-Aosta +1.72% / 2.3x. This report asks: what next? Fresh exploration of the post-005 code found the remaining gap was **mostly wasted throughput, not missing algorithms** — a 24 MB `Solution` copy on every improving iteration, a malloc+sort in the innermost loop of the entire solver, a single mutex serializing most of Stage 3, an unbounded local-search sweep that sat entirely outside its own time budget, and a single-threaded kNN build. This report fixes those, re-tunes the resulting slack, and reports the honest result: closer, not there.

**Goal (unchanged):** strictly lower cost than FILO2 while staying at least 2x faster (a confirmed target — see the plan file for the tradeoff discussion). **Beat-targets:** Lazio cost < 3,158,419,623; Valle-D-Aosta cost < 21,732,499.

## 1. Throughput fixes (Phase 2)

Every item here had one validation bar: **byte-identical cost with lower wall time.** Any cost change from a "pure performance" change means a bug — this is the same standard report 005 §5 used for its rescan fix, and every item below was confirmed against it via the full Tier-1 harness (34 instances × 5 seeds = 170 runs, `verifier.py` on every single one).

| # | Change | Result |
|---|---|---|
| 2.1 | MSVC build flags (`/O2 /Ob3 /GL /arch:AVX2`) | `-O3 -march=native` in `CMakeLists.txt` was GCC syntax MSVC silently ignored. Deliberately **not** `/fp:fast` — `dist()` is `llround(sqrt(...))` and `verifier.py` independently recomputes with Python's `math.sqrt`; a lower-precision reciprocal-sqrt could shift a value across a rounding boundary into a genuine verifier mismatch. Byte-identical, confirmed. |
| 2.2 | `get_top3_insertions`: vector+push_back+`std::sort` of an entire route → zero-allocation 3-element running min | Called up to 60x per local-search node pop — the innermost loop of the whole solver. Tier-1 wall time **-41.8%** (543.6s → 316.5s across the 170-run batch). Byte-identical cost across all 170 real CVRPLIB instances (a small perturbation on the *synthetic* N=2000 demo instance specifically, explained below). |
| 2.3 | Per-improvement `bestSol` snapshots drop `routePosition`/`cumLoad` (pure derived caches, fully reconstructible via `update_route_info`), recomputed once at loop exit instead | Halves the dominant per-improvement copy cost (24 → 12 bytes/node). Byte-identical. |
| 2.5 | `NeighborLists::build` (kNN) parallelized over independent per-node queries | Each node's query only reads the already-built, immutable `KDTree` and writes its own `nbr[i]` slot — no synchronization needed, deterministic regardless of thread count. Byte-identical. |
| 2.6 + 4.1 | Deleted dead `top3_cache` (56 MB, never read/written anywhere in the repo); right-sized route-indexed arena arrays; hoisted `ThreadArena` out of Stage 3's per-chunk-pair lambda into a reused pool; added bounds guards on route-indexed arena access | See §2 below — this one had a real bug in it, found and fixed via bisection. |

**On 2.2's N=2000 cost shift:** `std::sort` was never actually guaranteed stable, so replacing it with a running min (which uses a well-defined, deterministic first-encountered-wins tie-break) can select a different position among *exactly tied* candidates than whatever MSVC's introsort happened to do. This only visibly moved the result on the synthetic N=2000 demo instance (54,816→54,860 at P=1; 55,498→55,024 at P=4, both feasible, both re-verified) — a quirk of that instance's degenerate tie structure, not a general effect. All 170 real Tier-1 instances landed exactly byte-identical, confirming this in aggregate.

## 2. A real regression, found by bisection

Tier-1 validation of the arena changes (item 2.6/4.1) came back **72/170 infeasible** — capacity violations, the same failure signature report 005's Stage 5 bug had. Bisecting by reverting the hoist, then the bounds guards, one at a time:

- Reverting the **hoist** alone: still broken (72/170).
- Reverting the **bounds guards** too (only the sizing signature change left): **worse** — `STATUS_HEAP_CORRUPTION` crashes and a hang.

The guards were masking a real bug, not causing one — removing them exposed it more directly. Root cause: `stage4_route_cleanup` (runs immediately before Stage 5) **compacts** `routeHead` down to just the live route count (e.g., ~20-40 routes at N=2,000, ~40,000 at Lazio). Stage 5's arena was sized from `globalSolution.routeHead.size()` read **after** that compaction — a snapshot with zero headroom for the routes Stage 5's own `recreate()` creates as it runs (the same "leaked route slot" mechanism report 005 documented for Stage 3, `routeHead.resize(r+100)` on demand, uncapped). The previous single-argument `reserve_fixed_capacity(inst.n)` call defaulted to `inst.n + 100` — accidentally generous, with plenty of slack; the "fix" replaced that with a much smaller, precisely-wrong bound.

**Fix:** Stage 5 now uses the same provably-generous bound Stage 3 already commits to (`2 * inst.n + 10000`) instead of a point-in-time snapshot. The bounds guards (`r_j` checked against `arena.route_visited_iter.size()` before indexing, both in the SWAP\* precompute and evaluation passes) were restored — they were correct and are now genuine defense-in-depth, not covering for a bug. Re-verified: 170/170 clean, byte-identical cost. The arena hoist itself was re-confirmed race-free with **15/15 bit-identical results across 3 repeated rounds** at a fixed seed — the same determinism signature report 005 used to confirm its own concurrency fixes.

This is worth stating plainly: right-sizing a buffer from "a generous constant" to "a precisely-computed bound" is exactly the kind of change that looks obviously correct and isn't. The fix that actually worked was reusing the same already-proven-generous formula Stage 3 uses, not computing a new "precise" one.

## 3. Time-bounding the full sweep (Phase 1.1)

Report 005 added `full_sweep_local_search` (a real, convergent local-search pass over every customer/boundary node before the main SA loop) but never bounded it — both call sites ran **before** their stage's `stageStart` was even captured, so the sweep sat entirely outside `--stage3-ms`/`--stage5-ms`. Report 005 §4 measured this directly: "Stage 4+5 took 56.0s against a nominal 20s budget."

Fix: `full_sweep_local_search` takes an optional deadline, checked once per batch (every 50 nodes — negligible overhead, not per-node). Both call sites now share **one clock and one budget** with their stage's main loop — time the sweep spends is time the main loop has less of, not extra time stacked on top. Legacy (no time-budget flags) mode is unaffected by construction (`deadline` defaults to "unbounded"), confirmed byte-identical at both N=2,000 and full Tier-1.

**Verified on Valle-D-Aosta:** Stage 4&5 now measures **60.3s against a 60,000ms nominal budget** — essentially exact, versus the previous 56s-against-20,000ms (2.8x overrun).

## 4. Re-tuning the resulting slack (Phase 1.2)

With the sweep now properly bounded, both instances had real unused headroom at their existing report-004/005 budgets:

| | Time used | Ceiling (2x floor) | Headroom |
|---|---|---|---|
| Valle-D-Aosta | 103.5s | 118s | 14.5s |
| Lazio | ~183s | 320s | ~137s |

Swept both stage 2 and stage 5 budgets against this headroom:

| Instance | Config tried | Cost gap vs BKS | Time |
|---|---|---|---|
| Lazio | baseline (stage5=20k) | 1.211% | 183s |
| Lazio | stage5→120k (+100s) | 1.235% (**worse**) | 297s |
| Lazio | stage2→250k (+130s) | 1.152% (best) | 313s (2.05x — thin margin) |
| Lazio | **stage2→200k (+80s, chosen)** | **1.158%** | **263s (2.44x)** |
| Valle-D-Aosta | stage2→50k (+10s) | 1.891% (no improvement) | 113.5s (thin margin) |
| Valle-D-Aosta | **baseline, unchanged** | **1.880%** | **103.5s (2.29x)** |

Two findings worth stating plainly:

- **Stage 5's budget still shows muted-to-negative returns at Lazio scale**, confirming report 004's original finding even after every throughput fix in this report — this was not an artifact of a slow, broken measurement. Giving it 6x more time made cost slightly *worse* (within run-to-run noise, but never better across repeated tries).
- **Stage 2's budget is the lever that actually works at large scale** — chunk-local search benefits from more time in a way Stage 5's full-graph polish doesn't. This wasn't tested in report 004/005's tuning passes (which only swept Stage 5), so it's a genuinely new finding, not a re-confirmation.
- **Valle-D-Aosta is not headroom-limited** — extra time in either stage bought nothing measurable. Its gap to FILO2 has to close some other way (see §6).

Chose the config with 57s of safety margin over the one with 7s: report 005 already documented real run-to-run timing variance at large scale, and a config sitting within single-digit seconds of a hard ceiling risks crossing it on an unlucky run.

## 5. Final numbers

| Scale | Config | Cost gap vs FILO2 | Speedup vs FILO2 |
|---|---|---|---|
| Valle-D-Aosta (20k), 5 seeds | `--stage2-ms 40000 --stage3-ms 1000 --stage5-ms 60000` (unchanged) | **+1.632%** (report 005: +1.72%) | **2.29x** |
| Lazio (1M), 3 seeds | `--stage2-ms 200000 --stage3-ms 2000 --stage5-ms 20000` | **+0.741%** (report 005: +0.88%) | **2.43x** |

Both improved on cost relative to report 005 while giving back some of the speed margin — an explicit, confirmed tradeoff (this session's target was "beat FILO2 on cost while staying ≥2x faster," not "stay maximally fast"). **Neither beat-target is met** — both instances are still more expensive than FILO2, not less.

## 6. Phase 3 evaluated and skipped

The plan's next lever was de-serializing Stage 3's `route_creation_mutex` (held for the entire `ruin` walk and the entire `recreate` loop). The original exploration measured Stage 3 at ~32% of total runtime — but that measurement predates every fix in this report. Re-measured on the final, tuned Lazio config:

```
Setup (Stage 0):              2.0s
Stage 1 & 2 (Construction/ILS): 204.8s
Stage 3 (Parallel Healing):    15.4s   <-- 5.85% of total, not 32%
Stage 4 & 5 (Cleanup/Polish):   40.2s
Total:                        262.4s
```

Everything *else* got faster, so Stage 3's share shrank even though its absolute time barely changed. Even a perfect (unrealistic) full de-serialization would reclaim at most ~10-12s — a 4-5% total-time improvement — against real risk: Stage 3's concurrency model is exactly where report 005 found both of its concurrency bugs. Not a good trade at this point. **Skipped**, with this measurement as the record of why.

## 7. What's still open

- **The remaining ~0.7-1.6% cost gap is most likely the missing-neighborhoods gap identified before report 005**: this solver has 5 local-search operators (relocate, swap, 2-opt, 2-opt\*, SWAP\*) against FILO2's 23, and every evaluator forbids the depot as an operand (blocking ~8% of all edges from ever being optimized). Neither was attempted this session — Or-opt/segment-relocate is the natural next addition, with well-defined extension points already mapped: a new `eval_*` following the established contract (depot-first guard, capacity before `dist`, `return 0` as the reject sentinel — the re-verify-under-lock re-runs it, so anything the applier can't survive must be rejected here), a new `apply_*` built exclusively from `remove_customer`/`insert_customer` (which makes it automatically covered by `rescan_touched_routes` and `apply_undo_list` — both derive their touched-route set from the do/undo log, nothing needs registering), plus branches at the op-code list, the scan loop, the verify switch, and the apply switch in `local_search`. Depot-edge moves specifically were scoped out of this round: lifting the depot exclusion touches the `pred[0]`/`succ[0]` shared-alias mechanism that was behind report 005's crash, and isn't worth the risk without dedicated attention.
- **`Solution::pred/succ/routeOf/routePosition/cumLoad` are still sized to `inst.n + 1` per chunk regardless of chunk size** (a chunk owning 1/16th of the customers at P=16 still carries full-size arrays — a 16x overshoot on the per-chunk working set and on every remaining chunk-local copy). Identified but deprioritized this session: fixing it means touching the global-vs-local node-id indexing broadly, a larger and riskier change than anything else in this report for benefit that ranked below the items actually done.
- **Valle-D-Aosta's gap did not respond to the time-budget lever at all** (§4) — closing it further needs either the same missing-neighborhoods work or a scale-specific investigation this report didn't have time for.

## 8. Caveats

- Same MSVC `-O3`/`-march=native` caveat as prior reports, now addressed for MSVC specifically (§1, item 2.1) — GCC/Clang builds were already fine.
- Tier-2/3 remain single-instance-family (one Valle-D-Aosta, one Lazio); Tier-1's 170-run spread is the statistically solid part of this report.
- Lazio's timing carries real run-to-run variance, as report 005 also found — the configs in §5 were chosen with an explicit safety margin for this reason, not just for their single best-observed number.
