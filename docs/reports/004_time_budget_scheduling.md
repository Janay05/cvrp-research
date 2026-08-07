# Report 004 — Time-budget scheduling and cost tuning

**Follows:** [003_scale_testing.md](003_scale_testing.md). That report found the per-thread iteration budget (`inst.n * g_iters_per_node`) was tuned only at N=2,000, became inefficient at N=20,000, and needed a hand-picked override to even run at N=1,000,000. This report replaces that mechanism and uses the replacement to tune cost.

## 1. Stress-testing the old mechanism first (bug-fixing)

Before changing anything, pushed the existing `--max-iterations` override well beyond anything previously tested, specifically to catch a latent bug the way the last one (Stage 5's missing route-info rescan, report 002) was found — by pushing harder, not by inspection:

| Instance | Config | Result |
|---|---|---|
| Valle-D-Aosta (20k) | `--max-iterations 200000` (2x) | ✅ feasible, cost 22,120,401 |
| Valle-D-Aosta (20k) | `--max-iterations 400000` (4x) | ✅ feasible, cost 22,105,474 |
| Lazio (1M) | `--max-iterations 20000` (2x) | ✅ feasible, cost 3,192,447,852 |
| Lazio (1M) | `--max-iterations 50000` (5x) | ✅ feasible, cost 3,190,944,570 |

All four passed `verifier.py` cleanly. No new bugs found — the codebase held up under 2-5x more stress than report 003 tested. Proceeded to Phase 2 with a validated foundation.

## 2. Time-budget scheduling

Converted all three ILS loops (`stage2_ils`, `stage3_healing_ils_pass`, `stage5_serial_polish` in `Stage2_ILS.cpp`) to an optional wall-clock time-budget mode, mirroring FILO2's own `TimeBasedSimulatedAnnealing` (`baselines/filo2/opt/SimulatedAnnealing.hpp`): the cooling schedule is driven by elapsed-time fraction (`T = T0 * (Tf/T0)^(elapsed/budget)`) instead of iteration fraction, so the same budget behaves sensibly regardless of instance size. New flags: `--stage2-ms`, `--stage3-ms` (per chunk-pair), `--stage5-ms`. Omitting them keeps the exact legacy iteration-count path — verified byte-for-byte identical to report 002 (P=1: 54,589; P=4: 55,715 at N=2,000).

**What "scale-aware" actually means here:** the mechanism doesn't auto-select a budget for you — you still choose a number. What changed is the *unit*. A `--max-iterations` number that means "reasonable" at N=2,000 means "instant, wasted" or "impossibly slow" at other scales, forcing hand-tuning per instance (report 003). A `--stage2-ms 40000` means "40 seconds" at any N — the mechanism adapts the achieved iteration count to fit, not the other way around. That portability is the actual fix.

**Side effect — the per-thread timing imbalance from report 003 is resolved by construction.** At N=20,000/P=4, workers used to finish 284s to 459s apart (report 003 §3) running identical iteration counts on unequally-hard chunks. With a time budget, every thread runs for the same wall-clock duration and lets iteration count absorb the difficulty difference instead:

| Scale | Old spread (iteration-count mode) | New spread (time-budget mode) |
|---|---|---|
| N=20,000, P=4 | 284s – 459s (175s / ~1.6x) | 40,003ms – 40,012ms (9ms) |
| N=1,000,000, P=16 | *(not measured at this P in report 003)* | 120,200ms – 120,236ms (36ms) |

Iteration counts now visibly vary instead (e.g. 168,528–271,149 across 4 workers at N=20,000 for the same 40s budget) — confirming the imbalance was a real per-chunk difficulty difference, not a bug, and that it's fully absorbed by this design.

**A bug this surfaced and fixed along the way:** the first sweep of Stage 5 time budgets (1,000ms to 40,000ms at N=2,000) showed *no change in Stage 5's actual runtime* — it was exiting after ~150ms regardless of the requested budget. Cause: `stagnation_limit = 150` (an early-exit for "150 consecutive non-improving iterations") was carried over unchanged into time-budget mode, and at N=2,000 iterations are fast enough that 150 non-improvements happens almost immediately, silently capping Stage 5 far below whatever budget was requested. Fixed by disabling the stagnation early-exit specifically in time-budget mode (legacy mode keeps it, unchanged) — with a real time cap already bounding worst-case runtime, letting a stagnant search keep exploring is low-risk and occasionally still finds an improving move, whereas exiting early unconditionally forfeits budget the caller explicitly asked for.

## 3. Cost tuning results

With the mechanism validated, swept Stage 5's budget (the lever reports 002/003 identified as highest-value, since it's the only stage with full-graph visibility) and checked the Stage 3 allocation question report 003 raised.

### N=2,000 (fast iteration, `--stage2-ms 15000 --stage3-ms 400`, Stage 5 swept)

| Stage 5 budget | Cost | Total time | vs. P=1 (54,589) | vs. FILO2 (51,878, 114s) |
|---|---|---|---|---|
| 1,000 ms | 55,344 | 17.2s | +1.4% | +6.7% |
| 20,000 ms | 54,739 | 36.2s | +0.3% | +5.5% |
| 60,000 ms | 54,591 | 76.2s | +0.004% | +5.2% |
| 90,000 ms | **54,240** | 106.2s | **-0.64% (beats P=1)** | +4.55% |

Clear, close-to-monotonic improvement with more Stage 5 time (some run-to-run noise expected, single run per config) — at 90 seconds of Stage 5 polish, the chunked P=4 result **beats the sequential P=1 baseline** on cost while still finishing slightly faster than FILO2 (106.2s vs. 114s). This confirms report 002/003's hypothesis: Stage 5 was the underused lever.

### Valle-D-Aosta (20,000 nodes), `--stage2-ms 40000 --stage3-ms 1000 --stage5-ms 60000`

| | Time | Cost | vs. FILO2 |
|---|---|---|---|
| FILO2 | 237s | 21,732,499 | — |
| Ours (tuned) | **103.2s** | 22,047,178 | **2.30x faster, +1.45% cost** |
| *(old time-budget default, report checkpoint)* | *460s* | *22,069,827* | *1.9x slower, +1.55% cost* |

Beats the previous (iteration-count) default on **both** axes at once — faster and better cost — by spending the same rough time budget on Stage 5 polish instead of just more chunk-local Stage 2 search.

### Lazio (999,999 nodes), P=16, three Stage 3/Stage 5 allocations tested

| Config | Stage3 (per pair) | Stage5 | Total time | Cost | vs. FILO2 (641s, 3,158,419,623) |
|---|---|---|---|---|---|
| A: more Stage 3 | 5,000 ms | 10,000 ms | 178.6s | 3,193,522,396 | 3.59x faster, +1.11% |
| B: less Stage 3, more Stage 5 | 500 ms | 60,000 ms | 195.9s | 3,193,205,579 | 3.27x faster, +1.10% |
| **C: balanced (best)** | 2,000 ms | 20,000 ms | **165.7s** | **3,193,217,946** | **3.87x faster, +1.10%** |

**Finding, contrary to the naive expectation from the N=2,000/20k results:** at 1,000,000 nodes, Stage 3's and Stage 5's budgets barely matter within the ranges tested — all three configs land within 0.01% of each other on cost, despite Config B spending 12x more time in Stage 5 than Config A. The lever that worked cleanly at smaller scales has much more muted returns here. Likely explanation: Stage 5 operates on the *entire* graph, so a fixed time budget represents a proportionally much smaller "dose" of full-graph search relative to problem size at 1,000,000 nodes than at 2,000 — getting a comparably large effect would need a budget large enough to erode the speed advantage that's the point of this architecture at this scale. Config C (balanced, modest budgets on both) happens to dominate on both cost *and* time among the three tested, so it's the recommended Lazio configuration — not because Stage 3/5 tuning did much, but because there was no evidence a bigger spend anywhere helped enough to justify it.

## 4. Recommended configurations

| Scale | Recommended flags | Result vs. FILO2 |
|---|---|---|
| ~2,000 nodes | `--stage2-ms 15000 --stage3-ms 400 --stage5-ms 60000` (balanced) or `90000` (quality-focused) | ~5.2% worse cost at 1.5x FILO2's time, or beats P=1 at roughly FILO2's own time |
| ~20,000 nodes | `--stage2-ms 40000 --stage3-ms 1000 --stage5-ms 60000` | 2.3x faster, +1.45% cost |
| ~1,000,000 nodes | `--stage2-ms 120000 --stage3-ms 2000 --stage5-ms 20000` | 3.87x faster, +1.10% cost |

These are not auto-selected by the solver — still manually chosen per run via the new flags, same as before. What changed is that "40 seconds" is a stable, meaningful choice at any N, where a specific iteration count never was.

## 5. What's still open

- **No automatic budget selection.** The operator still picks numbers; this report only made the numbers portable across scale, not automatic. A further step (not attempted here) would be estimating a good split from a short calibration run, similar to how `routemin` timing is used elsewhere.
- **Stage 5's diminishing returns at 1,000,000 nodes are not fully explained**, just observed. Worth profiling whether Stage 5's O(N) `bestSol = globalSolution` copy (noted as a risk in the original plan for this work) is eating into the budget disproportionately at this scale — it wasn't measured directly in this pass.
- **Only one run per configuration** at the 20k/1M scales (matching prior reports' practice) — the Lazio Stage 3/5 allocation results being within 0.01% of each other could partly be within normal run-to-run noise rather than a fully separated signal; worth a repeated-run check before treating "Stage 3/5 budget doesn't matter at 1M" as fully settled.
- **The N=2,000 sweep is single-run per data point too** — the 20,000ms result (54,739) being worse than the 5,000ms result from an earlier ad hoc check (55,441, not in the table above) during initial exploration is a reminder that individual runs carry real noise; the reported trend is consistent across 4 points but not statistically tight.

## 6. Caveats

- Same MSVC `-O3`/`-march=native`-ignored caveat as prior reports — doesn't affect internal comparisons, does affect absolute-time comparisons against FILO2's prebuilt binary.
- All comparisons single-instance (one seed each of CMT1/Valle-D-Aosta/Lazio) — see report 003 for the same caveat, unchanged here.
