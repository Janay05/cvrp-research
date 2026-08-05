# Report 002 — Capacity bug fix + iteration-budget rebalance

**Follows:** [001_p1_p4_filo2_baseline.md](001_p1_p4_filo2_baseline.md). That report flagged two things left open: a capacity-violation bug found under a stress-test iteration count, and the suggestion that P=4's default iteration budget (`chunkSize*50`) throws away most of its parallelism dividend on pure speed instead of search. This report does two things about that: (1) finds and fixes the actual root cause of the capacity bug, and (2) rebalances the iteration budget now that it's safe to increase it — then re-measures against FILO2 and quantifies what gap remains.

**Instance/build:** same as report 001 (`test_2000.vrp`, N=2000, MSVC Release, `-O3`/`-march=native` still silently ignored by MSVC).

---

## 1. Root-causing the capacity bug

Report 001 found that forcing each P=4 thread to run 100,000 iterations produced routes exceeding the 100-unit capacity (up to 170 units), independently confirmed by `verifier.py`, with the solver's own `routeLoad` bookkeeping matching the actual (invalid) route contents exactly — ruling out a simple load-tracking arithmetic error.

**Method:** rather than keep auditing the four move-evaluation functions by eye, a temporary debug check (scanning every route for `load > Q` after each pipeline stage) was inserted at four checkpoints: after Stage 1&2 (per-chunk, in `Worker.cpp`), after Stage 3 healing, after Stage 4 cleanup, and after Stage 5 polish (`main.cpp`). Re-running the same stress config showed the violation appearing **only after Stage 5**:

```
[CAPCHECK after-stage5-polish] Route 0 OVER CAPACITY: load=144 Q=100
[CAPCHECK after-stage5-polish] Route 9 OVER CAPACITY: load=170 Q=100
[CAPCHECK after-stage5-polish] Route 11 OVER CAPACITY: load=167 Q=100
[CAPCHECK after-stage5-polish] Route 19 OVER CAPACITY: load=154 Q=100
```

**Root cause:** `stage5_serial_polish` (`Stage2_ILS.cpp`) runs the same `ruin` → `recreate` → `local_search` sequence as `stage2_ils` and `stage3_healing_ils_pass`, but was missing one step both of those have: a full `update_route_info()` rescan of every route between `recreate()` and `local_search()`.

`recreate()` inserts customers via `insert_customer()`, which updates `routeLoad` and the linked-list pointers immediately and correctly — but only calls `update_route_info()` (which recomputes `cumLoad[]`/`routePosition[]`) for its new-empty-route fallback path, *not* for the normal case of inserting into an existing route. `eval_2opt_star`'s capacity check reads `cumLoad[]` directly:
```cpp
Cost load_tail_i = sol.routeLoad[r_i] - sol.cumLoad[i];
...
if (sol.routeLoad[r_i] - load_tail_i + load_tail_j > inst.Q) return 0;
```
Without the rescan, a route that `recreate()` just inserted a customer into has a stale (too-low) `cumLoad[]`, so this check can underestimate the route's true load and pass a 2-opt\* segment exchange that actually pushes the route over capacity. `stage2_ils` (`Stage2_ILS.cpp:704-706`) and `stage3_healing_ils_pass` (`Stage2_ILS.cpp:786-790`) both already do this rescan; `stage5_serial_polish` was the one caller missing it — which lines up exactly with the bisection result (violations only after Stage 5).

**Fix** (`Stage2_ILS.cpp`, `stage5_serial_polish`):
```cpp
recreate(globalSolution, arena, cache, inst, neighborLists, nullptr);

for (int r = 0; r < globalSolution.numRoutes; ++r) {
    update_route_info(globalSolution, r, inst);
}

bool local_search_improved = true;
while(local_search_improved) { ... }
```

**Verification:** re-ran the identical stress config (100,000 iterations/chunk) — zero `CAPCHECK` warnings, and `verifier.py` confirmed the resulting solution feasible (cost 55,715 vs. the previous invalid run's 53,616 — the "improvement" report 001 saw was partly the bug letting routes overflow for free; the honest number is higher). Debug instrumentation was removed after confirming the fix; it isn't part of the shipped binary.

---

## 2. Rebalancing the iteration budget

**Change** (`Stage2_ILS.cpp`, `stage2_ils`):
```cpp
// was: int max_iterations = chunkSize * 50;
int max_iterations = inst.n * 50;
```
Every thread now runs the same absolute 100,000-iteration budget P=1 spends on the whole graph, regardless of its chunk's size — spending the parallelism dividend (chunked iterations are ~6-8x cheaper per report 001) on more search instead of banking it all as idle time. This is a no-op for P=1 (`chunkSize == inst.n` when there's only one chunk), so P=1's numbers are unaffected and serve as a consistency check.

**A second, exploratory data point** was also collected (2x this budget, `inst.n * 100`, not shipped as the default) to see whether pushing further continues to pay off — see §4.

---

## 3. Results after both changes

| Config | Iterations/thread | Wall time | Cost | vs FILO2 | Feasible? |
|---|---|---|---|---|---|
| FILO2 (unchanged from report 001) | 100,000 | 114.0 s | 51,878 | — | ✅ |
| P=1 (unaffected by either change) | 100,000 | ~77–110 s | 54,589 | +5.2% | ✅ |
| P=4, old default (`chunkSize*50`, report 001) | 25,000 | 7.9 s | 56,131 | +8.2% | ✅ (but see report 001's other bugs) |
| **P=4, new default (fixed + rebalanced)** | **100,000** | **11.7 s** | **55,715** | **+7.4%** | ✅ |

The new default is both faster in absolute terms than the naive intuition would suggest it should be (100k iterations/thread on N=500 in 11.7s vs. P=1's 100k iterations on N=2000 in ~77-110s — an ~7-9x speedup for 4x more *aggregate* search effort, not less) and closes part of the quality gap versus the old default (56,131 → 55,715, -0.7%). It's still slower per-thread-iteration-count than the old default (11.7s vs 7.9s), which is expected and intentional — that's the parallelism dividend being spent on search now.

### Stage-wise breakdown, P=4 (fixed + rebalanced)

| Stage | Time | % of total |
|---|---|---|
| Stage 0 — Partitioning & k-NN setup | 7.6 ms | 0.07% |
| Stage 1 & 2 — Parallel construction + ILS (100,000 iter/thread) | 10,918.7 ms | 93.5% |
| Stage 3 — Parallel boundary healing | 597.8 ms | 5.1% |
| Stage 4 & 5 — Cleanup + serial polish | 152.3 ms | 1.3% |
| **Total** | **11,676.4 ms** | 100% |

Per-thread Stage 2 detail:

| Worker | Chunk size | Iterations | Stage 2 time | Distance evals |
|---|---|---|---|---|
| 0 | 500 | 100,000 | 10,909 ms | 1,237,330,210 |
| 1 | 500 | 100,000 | 9,669 ms | 1,057,944,737 |
| 2 | 500 | 100,000 | 7,437 ms | 841,417,665 |
| 3 | 500 | 100,000 | 5,923 ms | 666,621,477 |

Compare to report 001's P=4 default: total distance evaluations went from 1.12B to 3.80B (4x more absolute search, matching the 4x iteration increase) — Stage 1&2 wall time went from 6.0s to 10.9s (only ~1.8x, not 4x), because the extra iterations are still running on the same small, cheap 500-node sub-problems.

---

## 4. Is there more headroom? (exploratory)

A second run at `inst.n * 100` (200,000 iterations/thread, double the new default) was tested but **not shipped**:

| Config | Wall time | Cost | Δ cost vs. 1x |
|---|---|---|---|
| P=4, 1x budget (shipped default) | 11.7 s | 55,715 | — |
| P=4, 2x budget (exploratory) | 22.5 s | 55,432 | -0.5% |

Doubling the iteration budget (doubling wall time) only closed **0.5%** more of the gap to FILO2. This is a diminishing-returns signal: naively cranking the iteration count further is not an efficient way to close the remaining ~7% gap, even though there's still large wall-clock slack available (22.5s is still ~5x under FILO2's 114s). The bottleneck has shifted from "not enough search" to something structural.

---

## 5. What's actually causing the remaining ~7% gap, and what to do about it

1. **Chunk-boundary blindness, only partially recovered.** Stage 2's search on each chunk cannot see or make moves across chunk boundaries at all — only Stage 3's healing pass (restricted to boundary-adjacent customers, graph-colored chunk pairs) and Stage 5's full-graph polish get a chance to fix that, and both currently run for comparatively little time (Stage 3: 598ms; Stage 5 is folded into the 152ms Stage 4&5 combined figure) relative to the ~10.9s spent on chunk-local Stage 2 search. Diminishing returns on chunk-local iterations (§4) plus a comparatively tiny cross-boundary healing budget points at this as the most likely explanation for where the residual gap lives.
2. **Recommended next step: extend Stage 5, not Stage 2.** Stage 5's SA loop currently runs a fixed, small budget (`max_iterations = 500`, `stagnation_limit = 150`, both hardcoded in `Stage2_ILS.cpp`) regardless of how much wall-clock slack is available. Since Stage 5 operates on the *entire* un-chunked graph (no boundary restriction), it's the one stage that can actually make the cross-boundary moves Stage 2 structurally can't — and given P=4 currently finishes in ~12s against FILO2's 114s, there's roughly 10x wall-clock budget sitting unused that could go there instead of (or in addition to) more chunk-local Stage 2 iterations.
3. **Also worth trying: give Stage 3 healing a larger budget.** Its iteration cap is `min(1000, boundaryList.size() * 50)` (`Stage2_ILS.cpp`) — capped low regardless of available time. It uses the same move portfolio as Stage 2 (same `ruin`/`recreate`/`local_search` functions), so more iterations there should behave similarly to Stage 2's, but concentrated exactly on the boundary-adjacent customers most likely to still be sub-optimally placed.
4. **Longer-term, more invasive option: overlapping chunks or periodic re-partitioning**, so the chunk boundaries themselves aren't fixed for the entire run — this would reduce reliance on Stage 3/5 fixing boundary artifacts after the fact, at the cost of real implementation complexity.
5. **Not yet investigated:** whether FILO2's specific move portfolio/parameters (adaptive `gamma`/`omega` shaking, its particular RVND move ordering) give it an edge independent of iteration count or search architecture — worth a controlled comparison if the above structural changes don't close the gap.

Recommended order: try (2) first — it's a small, low-risk parameter change (just increase Stage 5's iteration/stagnation caps) that directly targets the mechanism (§5.1) most likely responsible, and there's already measured evidence (§4) that more chunk-local search isn't the efficient lever anymore.

---

## 6. Caveats (carried over from report 001, still apply)

- Wall-clock timings vary run-to-run on this machine (background load) — read as order-of-magnitude, not to the second. Cost values are the more reliable reproducibility signal (bit-identical across repeated runs with the same code, given seeded RNG).
- MSVC on this Windows build still silently ignores `-O3 -march=native`; both our solver and FILO2's prebuilt binary's actual optimization flags are unverified to match, so the absolute time comparison (not the internal P=1-vs-P=4 comparison) carries that caveat.
- Single 2000-node synthetic instance, uniform demand=1 — results may not generalize without re-running on other instances.
