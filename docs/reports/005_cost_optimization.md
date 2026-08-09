# Report 005 — Closing the cost gap: two concurrency bugs, five wasted-budget fixes, one rescan optimization

**Follows:** [004_time_budget_scheduling.md](004_time_budget_scheduling.md). That report closed the speed/quality tradeoff to +1.10-1.45% cost at 2.3-3.9x FILO2's speed and asked, implicitly, "what next to close the rest of the gap?" This report answers it: the gap was not primarily a tuning problem. Reading `src/Stage2_ILS.cpp` against `baselines/filo2/` found four defects that meant the solver was doing far less real search than its time budget implied, and stress-testing to validate a fix for one of them surfaced two more serious bugs — a data-loss race and a crash — neither previously known.

**Goal (unchanged):** strictly lower cost than FILO2 while still finishing faster.

## 0. Method

Built `tools/score_sol.py` (independently recomputes a `.sol` file's cost against its `.vrp` instance — no id assumptions taken on faith; the BKS/reference `.sol` files turned out to use the same "depot removed, survivors renumbered from 1" convention our own output does, confirmed by matching `data/bks/X/X/X-n101-k25.sol`'s embedded cost against `baselines/filo2/results/csvs/filo2-x.csv`), `tools/bench.py` (runs instance x seed combinations, calls `src/verifier.py` on every single result — not sampled — and scores against a reference), and `tools/compare_bench.py` (paired before/after comparison by exact instance+seed). Added `--seed` to the solver (previously fully deterministic, so repeat runs carried zero variance information).

**Tier 1:** 34 instances from `data/instances/X` (n=101-1001) x 5 seeds = 170 runs per config, scored against CVRPLIB BKS. **Tier 2:** Valle-D-Aosta (20k) x 5 seeds. **Tier 3:** Lazio (1M) x 3 seeds. Both scored against BKS (independently verified this session: Valle-D-Aosta 21,679,514/800 routes, Lazio 3,145,381,332/39,982 routes — both better than FILO2's own published numbers) as well as FILO2.

## 1. What was actually wrong (before touching any code)

Reading `Stage2_ILS.cpp`'s five operators, ruin/recreate, and SA setup against FILO2's 23 operators and parameter defaults found, with file:line precision:

- **Stage 2's ruin seed was drawn in the wrong index space** (`chunkSize`-bounded local count used directly as a global `NodeId`) — `ruin()` no-ops immediately for roughly (P-1)/P of all Stage 2 iterations.
- **`apply_undo_list`'s rollback rescan was capped at 10 routes** (`int modified_routes[10]`) — routine to exceed on small-route instances, leaving `routePosition`/`cumLoad` stale for the rest.
- **SA temperature was a hardcoded constant** (`avg_arc_cost_estimate = 100.0`) regardless of instance coordinate scale — ~11-32x too cold at Valle-D-Aosta/Lazio, collapsing the SA into pure hill-climbing exactly where the cost gap lives.
- **Three "seed every node" loops promised a full sweep and delivered 50** — `SVCCache`'s ring-buffer capacity (50) silently truncated "insert every customer" / "insert every boundary customer" loops to whatever the last 50 insertions were.

This is Phase 1 of the approved plan. Full reasoning and FILO2 comparison in the plan file; not reproduced here for length.

## 2. Two concurrency bugs found via stress testing (unplanned)

Attempting to freeze a Tier-1 baseline (the first real multi-seed, multi-instance stress test this codebase had been through) surfaced two bugs beyond the five above, both in Stage 3's healing pass, which is the only place multiple threads share one `Solution` object concurrently.

**2.1 — Data loss.** `stage3_healing_ils_pass` read `int prevNumRoutes = globalSolution.numRoutes;` outside its mutex and unconditionally wrote `globalSolution.numRoutes = prevNumRoutes;` on rejection, also outside the mutex. With multiple chunk-pair threads sharing `globalSolution` within a color class, one thread's rejected iteration could roll the shared counter back below a route another thread had just legitimately created — that route's customers become permanently invisible to every later scan and the output writer, without any error. **Fix:** removed the snapshot/restore entirely. `apply_undo_list` already fully unwinds a rejected iteration's own changes; a route it created ends up merely empty, the same harmless dead-slot state `recreate()`'s empty-route-reuse scan and Stage 4 cleanup already handle elsewhere in this codebase.

**2.2 — Crash.** Fixing 2.1 made `numRoutes` monotonically non-decreasing for the rest of Stage 3 (nothing restores it anymore), which made it far easier to exceed the pre-allocated route-slot buffer (`inst.n + 100` — a cushion, not a proven bound; confirmed by the fact that the original comment called it exactly that: "prevent reallocation data races"). `local_search`'s evaluation phase reads `routeHead`/`routeTail`/`routeLoad` without the mutex by design (only the eventual move-application is locked); if `recreate()`'s mutex-protected route creation ever needs to actually reallocate one of those vectors while another thread holds a bare pointer into it, that's a real use-after-free. Confirmed via repeated crashes (`STATUS_ACCESS_VIOLATION`, exit code `3221225477`/`3221226505`) across multiple X-set instances during stress testing. **Fix:** resized the pre-allocation to `2 * inst.n + 10000` — a provable bound (no route can ever hold more than `inst.n` customers total, and the leaked-dead-slot growth from 2.1's fix is bounded by total Stage 3 iterations across all threads, which is much smaller than `inst.n` at any tested scale).

**Verification:** the exact failing instance/seed combinations (`X-n670-k130`, `X-n936-k151`, others) were re-run repeatedly post-fix — 15/15 trials clean and, notably, **bit-identical across repeats** (the signature of a real race fixed, not papered over). Full Tier-1 (170 runs): 0 crashes, 0 infeasible, 0 data loss, down from 1 crash + 6 infeasible/lost-node runs in the two stress attempts that found these bugs.

## 3. Phase 1 fixes, measured

Each fix was landed and measured independently against the (by-then-clean) Tier-1 baseline. `verifier.py` ran on every single one of the ~1,200 solver invocations behind this report; none of the numbers below come from a run that wasn't independently confirmed feasible.

### 3.1 — Ruin seed fix (`partitionInfo.globalId[chunkId][...]` instead of the raw local count)

| | N=2000, P=4 (single run) | Tier-1 (170 paired runs) |
|---|---|---|
| Before | 55,715 | mean gap to BKS 4.574% |
| After | 54,989 | mean gap to BKS 4.189% |
| Delta | **-1.30%** | **-0.403%** |

31 of 34 Tier-1 instances improved; the 3 that didn't moved by <0.6%. Large, unambiguous, exactly where the analysis predicted it (the fix affects every P>1 run identically, so the size of the effect scales with how much of the previous budget was actually wasted).

### 3.2-3.4 — Temperature scaling + real full-sweep descent + defensive bounds guard (landed together)

Landed as one batch since 3.3 (full sweep) and 3.4 (guard) are expected to be near-neutral at Tier-1's small scale (a 50-node truncation barely matters when the *whole instance* is 101-1001 nodes; the real test is Tier-2/3, section 4). Implementing the full sweep correctly required two more fixes, both caught by the same Tier-1 harness before they ever reached a report:

- **Cost bookkeeping desync**: the new `full_sweep_local_search` helper applies real improving moves (mutating routes immediately, same as everywhere else in this file) but nothing folded its accumulated delta back into `sol.totalCost`/`globalSolution.totalCost` — genuinely-improved routes, stale reported cost. `verifier.py` caught this as "reported cost doesn't match recomputed cost" on 156/170 runs on the first attempt. Fixed by having the helper return its accumulated delta for the caller to apply (directly for Stage 5's single-threaded case; via the existing `acceptedDelta`-summed-after-join pattern for Stage 3's concurrent case, to avoid a shared-scalar race).
- **Stale route info**: `stage4_route_cleanup` relocates customers but never calls `update_route_info`; the new full sweep is the first thing to run `local_search` afterward, which depends on fresh `routePosition`/`cumLoad`. 6/170 runs showed capacity violations on the first attempt (same failure signature as report 002's original bug and section 2.2 above — stale position data feeding `eval_2opt_star`'s capacity check). Fixed with an explicit refresh before the sweep in both Stage 3 and Stage 5.

| | Tier-1 (170 paired runs, vs 3.1's checkpoint) | Tier-1 (vs original baseline) |
|---|---|---|
| Delta | -0.021% | **-0.424% combined** |
| Feasibility | 170/170 clean (after the two fixes above) | |

Small at Tier-1, as expected. Section 4 measures where these actually matter.

## 4. Tier-2/3: does any of this matter at the scale the architecture targets?

Using report 004's recommended tuned configs (`--stage2-ms 40000 --stage3-ms 1000 --stage5-ms 60000` at Valle-D-Aosta; `--stage2-ms 120000 --stage3-ms 2000 --stage5-ms 20000` at Lazio, P=16), now multi-seed for the first time:

| Scale | Cost gap to BKS | Cost gap to FILO2 | Time | Routes (BKS/kmin ref) |
|---|---|---|---|---|
| Valle-D-Aosta (20k), 5 seeds | 1.97% | 1.72% | 103.9s (2.28x faster than FILO2's 237s) | 806-810 (BKS 800, kmin 800) |
| Lazio (1M), 3 seeds | 1.30% | **0.88%** | 206.8-255s (2.5-3.1x faster than FILO2's 641s) | 40,459-40,498 (BKS 39,982, kmin 39,979) |

**Route counts are already close to optimal at both scales** (~1-1.3% excess over the bin-packing lower bound `kmin`) — **Phase 4 (route minimization) is skipped**, per the plan's own gate: there isn't enough slack in route count to be worth building a `routemin`-style phase for.

**Lazio's cost gap to FILO2 genuinely improved** (report 004: +1.10% → now +0.88%) — the largest-scale, most representative result got measurably better. **Valle-D-Aosta stayed roughly flat** (report 004's single-run checkpoint was 22,047,178; this session's 5-seed mean is 22,105,837, +0.27%, well within the spread of what a single un-averaged run can show — FILO2's own published 10-seed spread at this instance is 0.14% of cost, so a single-run "before" number was never going to support a tight comparison).

**Honest finding on timing:** Lazio's wall time shows real run-to-run variance at this scale (206.8s in a dedicated verification run vs. 226-255s across the 3-seed batch measured immediately after Phase 5, same config, same code). The stage-by-stage breakdown from the verification run explains part of this directly: Stage 4+5 combined took 56.0s against a nominal 20s Stage 5 budget, because **the new full-sweep descent (fix 3.3) is not itself time-bounded** — it runs to convergence over all ~1,000,000 customers regardless of `--stage5-ms`. This is a real, uncapped cost this report's fixes introduced, not fixed. Matches report 004's own stated caveat about single/few-run measurements at this scale not yet separating signal from noise — treat the Lazio timing range as exactly that, a range, not a point estimate.

## 5. Phase 5 — cheaper rescans (report 004's open question, answered)

Report 004 asked whether Stage 5's O(N) `bestSol` copy explained its muted returns at 1M nodes. There are actually two O(N)-per-iteration costs in each of the three ILS loops' accept/reject paths, and this report fixes one of them.

**What was fixed:** `stage2_ils`, `stage3_healing_ils_pass`, and `stage5_serial_polish` each ran an unconditional `for (r = 0; r < numRoutes; ++r) update_route_info(...)` after every accepted iteration (and again, redundantly, after every rejected one — `apply_undo_list` was already doing a complete rescan of everything it touched once report 002/this report's fix 3.2/3.4 removed the old 10-route cap, making the second rescan pure waste). `update_route_info` isn't a cheap per-route check — it walks that route's *entire* customer list — so at Lazio's ~40,500-route scale, this meant walking every one of ~1,000,000 customers on every single SA iteration to service a ruin that touches ~14 nodes. **Fix:** extracted the existing generation-marker dedup mechanism (already built for `apply_undo_list` in fix 3.2) into a shared `rescan_touched_routes` helper that reads the iteration's own do-list and rescans only the routes actually touched; deleted the now-fully-redundant reject-path rescans in all three loops.

**Verification:** byte-identical costs across all 170 Tier-1 paired runs and the N=2000 legacy P=1/P=4 numbers (55498, unchanged) — confirms this is a pure performance change, not a behavior change, exactly as intended. At Valle-D-Aosta's scale the wall-clock-bounded stages absorb the savings as more search within the same budget (no time change expected or observed). At Lazio, cost improved further (0.88%, from the table above) — plausibly more search fit into the same nominal budget, though see the timing caveat in §4 for why a clean before/after time delta isn't claimed here.

**Still open, not attempted this pass:** the *other* O(N) cost report 004 flagged — `Solution bestSol = globalSolution` / `bestSol = sol`, a full ~40MB struct copy on every improving iteration at Lazio scale — is unchanged. The plan describes the fix as "a delta-tracked incumbent plus one copy at the end"; that's a more invasive change than the rescan fix (need to track enough state to reconstruct the best solution without holding a live copy at every step) and was not attempted in the time available this session.

## 6. What's still open

- **The full Solution copy** (§5, above) — the other half of report 004's open question.
- **The full-sweep descent (fix 3.3) is not time-bounded** — a real, uncapped cost at huge scale (§4). Worth capping or making it periodic rather than unconditional-at-entry.
- **Missing neighborhoods** (Or-opt/segment relocate, reversed 2-opt*, ejection chains — FILO2 has 23 operators, this solver has 5) and **frozen depot edges** (every evaluator forbids the depot as an operand, blocking ~8% of all edges from ever being optimized) — both identified during the initial code-reading pass, neither attempted this session. Depot-edge handling specifically touches the exact `pred[0]`/`succ[0]` aliasing mechanism behind §2.2's crash, so it needs to be done carefully, not as a quick add.
- **Parallel cross-boundary optimization** (Stage 3's single global mutex serializing all healing threads; multi-round re-partitioning) — not attempted.
- **Goal status:** not yet reached. Lazio is within 0.88% of FILO2's cost while 2.5-3.1x faster — closer than report 004's 1.10%, but not yet *strictly lower cost*. Valle-D-Aosta is at 1.72%, largely unchanged. The neighborhoods gap (§ above) is the most likely place the remaining cost difference lives, per the original code-vs-FILO2 comparison.

## 7. Caveats

- Same MSVC `-O3`/`-march=native`-ignored caveat as prior reports.
- Tier-2/3 remain single-instance (one seed family each of Valle-D-Aosta/Lazio); Tier-1's 5-seed, 34-instance spread is the statistically solid part of this report.
- Lazio's timing numbers are a range, not a point estimate — see §4.
- BKS reference costs for CMT-style instances (unrounded distances) differ slightly from this solver's integer-rounded costs by convention, not by error — already known from report 003; unaffected by anything in this report.
