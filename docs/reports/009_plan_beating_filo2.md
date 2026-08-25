# Plan 009 — What it would actually take to beat FILO2 (time *and* cost)

Date: 2026-08-24. Status: proposal, nothing implemented.

---

## 0. Where we actually stand

Verified numbers (WSL, g++ `-O3 -march=native`, 5 seeds):

| Instance | Ours | FILO2 | Cost gap | Routes (ours / FILO2) |
|---|---|---|---|---|
| Valle-D-Aosta (n≈20k) | 22,002,022 @ ~102 s wall, P=2 | 21,738,306 @ ~140 s, 1 core | **+1.21%** | 806.6 / **800.0 every seed** |
| Lazio (n≈360k) | ~3.18e9 @ ~263 s | 3,157,466,737 @ ~340 s | **+0.71%** | ~40,372 / ~40,092 |

We look faster, but that comparison is not honest yet: we spend P cores against
FILO2's one, and FILO2's runtime is set by a fixed 100,000-iteration budget, not
by a time budget. FILO2 ships a `TIMELIMIT` build (`--optimization-seconds`,
default 100) — so **cost-at-equal-wall-clock has never been measured.** Every
claim below is meant to be judged on that scoreboard.

### The single root cause

From `results/run_log.txt`, VDA P=2:

```
Worker 0 chunk 0 actual_iterations_completed=68863
Worker 0 Stage 2: 38770.6 ms | dist_calls=7099964522
```

**~103,000 `sqrt` evaluations per ILS iteration.** ~183 M dist calls/sec/core,
14.2 billion total in 38.8 s. FILO2 does the equivalent work in the *low
hundreds* per iteration. That is a 2–3 order-of-magnitude gap in search
efficiency per unit of progress, and it is why we cannot afford enough
iterations to close the cost gap — the cost gap and the time gap are the same gap.

Our `local_search` (`src/Stage2_ILS.cpp:945`) pops a node, evaluates
**9 operators × 30 neighbors = 270 full delta evaluations**, applies exactly one
move, and throws the other 269 away. FILO2 keeps every candidate move alive in a
heap and recomputes only the handful invalidated by the applied move.

### Correction to something I said earlier

I previously described "let a vehicle serve a slightly different set of
customers" as something we would have to build. That was wrong — **ruin-and-recreate
already exists**, `Stage2_ILS.cpp:294-434` (SISR-style random-walk ruin, greedy
recreate). The genuinely missing form of that idea is FILO2's **ROUTEMIN**, which
is a different animal: it destroys *two whole routes* and tolerates temporarily
unserved customers. That is T3 below, and it is the highest-value item per unit
of effort.

---

## 1. The technique menu, with impact estimates

Confidence is my own; "cost" means gap-to-BKS reduction on VDA, "time" means
wall-clock at equal cost.

### Stage 1 — Free wins (2–4 days total)

| # | Technique | Expected gain | Confidence | Effort |
|---|---|---|---|---|
| **T0.1** | Remove the unconditional `thread_dist_calls++` from `dist()` (`Types.hpp:21`); guard behind `#ifdef PROFILE_DIST` | **3–8% time** | high | hours |
| **T0.2** | Symmetrize neighbour lists by union — `KDTree.cpp:101` builds an asymmetric kNN, so `j ∈ nbr[i]` does not imply `i ∈ nbr[j]`, and we silently never see half the candidate moves | **0.05–0.15% cost** | medium | hours |
| **T0.3** | Randomize recreate insertion order across FILO2's 4 rules instead of always descending-demand (`Stage2_ILS.cpp:356`) | 0.02–0.08% cost | medium | hours |
| **T0.4** | **Build FILO2 with `-DTIMELIMIT` and re-benchmark at our wall-clock.** No gain — but without it no claim we make is defensible | — | — | 1 day |

T0.4 is not optional. It gates the whole program: it may reveal the real cost gap
at equal time is 0.6% rather than 1.21%, which changes what is worth doing.

### Stage 2 — Cheap structural wins (1–1.5 weeks)

| # | Technique | Expected gain | Confidence | Effort |
|---|---|---|---|---|
| **T1** | Cache `costToPred[v]` in `Solution` (FILO2's `c_prev_curr`) — every operator currently recomputes arc costs it already knew | **15–30% time** | med-high | 3–5 days |
| **T4.2** | Adaptive per-vertex ruin size ω (init `ceil(ln n)`, ±1 on too-destructive / too-timid, bounds from mean arc cost × 0.375 / 0.85) — **independent of everything else, can start immediately** | **0.1–0.25% cost** | medium | 2 days |

### Stage 3 — ROUTEMIN done FILO2's way (~1 week)

**T3.** This is the prototype of the "different set of customers per vehicle"
idea, and it directly redeems the twice-failed work. Our v1/v2 failed because
they differed from FILO2 on *every single axis*:

| Axis | Our failed attempts | FILO2 |
|---|---|---|
| When | last, after Stage 4, before Stage 5 polish | **early** — right after construction, before core opt |
| Unit of destruction | try to empty *one* route | destroy **two whole routes** |
| Unserved customers | never allowed | allowed, probability `t` annealed 1.0 → 0.01 |
| Target | "one fewer route" | bin-packing FFD lower bound `kmin` |
| Neighbourhood | granular (γ=0.25) | γ = **1.0**, full neighbourhoods |
| Acceptance | any feasible | only **complete** solutions |

The failure is a "right idea, wrong placement and wrong mechanism" result, not a
refutation. FILO2 hits exactly 800 routes on every VDA seed; we sit at 806.6.

- **Expected gain: 0.3–0.5% cost.** Confidence med-high.
- Requires partial-solution support (an unserved-customer pool) — the main real work.

### Stage 4 — The inner-loop rewrite (2–4 weeks)

**T2. Static Move Descriptors + intrusive min-heap + precomputed `edge_costs` +
`Cache12`/`Cache1`/`Cache2` partial evaluation.**

Four stacked mechanisms, all of which we lack:

1. An SMD `(i, j, delta, heap_index)` per candidate pair, kept in a heap keyed on
   delta. After applying a move, only the *affected* vertices' generators are
   recomputed (~|affected| × γk ≈ 31 deltas) instead of 270 per node.
2. Precomputed `edge_costs` for every candidate pair (half-length array, twin
   packing via `index ^ 1`).
3. `c_prev_curr` arc caching (= T1, which is why T1 lands first).
4. Delta factored into an i-side and a j-side part, so the i-side is computed once
   per vertex and reused across all ~γk of its generators.

Plus `apply_rough_best_improvement`: scan the heap *array* linearly via `spy()`
rather than popping repeatedly — the array is only partially ordered, hence
"rough", and that is fine.

- **Expected gain: 3–10× fewer distance evaluations per applied move →
  0.3–0.7% cost at equal time, or 2–5× time reduction at equal cost.**
  Confidence medium on magnitude, high on direction.
- **Cost:** rewrites `local_search` entirely and imposes a contract on every
  `apply_*` — each must report its affected vertices and update bits. This is the
  invasive one. It is also the only item that attacks the root cause.

Once T2 exists, **T4.1 adaptive per-vertex γ** (base 0.25 of k=25, doubled on
repeated failure, reset on new best) becomes nearly free and buys another
**1.5–3× fewer evaluations**. It is meaningless without T2.

### Stage 5 — More search power (1.5–2 weeks, cheap *after* T2)

| # | Technique | Expected gain | Confidence |
|---|---|---|---|
| **T5.1** | Ejection chain (depth ~25), isolated in its own VND tier | 0.2–0.4% cost | med-high |
| **T5.2** | Missing segment–segment operators: E21/E22/E31/E32/E33 + Rev variants (we have E10/E20/E30 +rev and E11) | 0.1–0.2% total, diminishing | medium |
| **T6** | Clarke & Wright construction replacing randomized MST + DFS | 0–0.15% cost directly; **real value is giving ROUTEMIN a sane starting route count** | medium |

### Stage 6 — Parallel architecture (the fork in the road)

- **(A) Keep spatial chunking, just fix the inner loop.** Lowest risk. Ceiling: we
  stay behind on cost, win comfortably on time.
- **(B) FILO2-style *global* localized search, parallelized by optimistic
  concurrency** — per-route locks, abort/retry on conflict. Each iteration touches
  ~50 vertices and 2–5 routes, so conflicts are genuinely rare. **This is the only
  architecture that converts a fixed inner loop into a win on both axes.**
  Concurrency correctness is hard; budget 3–4 weeks and expect subtle bugs.
- **(C) Parallel multi-start portfolio** with periodic best-solution sharing.
  Trivially correct, embarrassingly parallel. Gets best-of-T, not T×-speed. Our VDA
  seed spread is 21.958M–22.061M, so **best-of-5 ≈ 0.2% under the mean** for free.

**Recommendation: C as the safe play (1 week, ~0.2% nearly guaranteed), B as the
ambitious one.** Note the P-sweep already ruled chunking out as the *dominant* cost
driver — P=1, the best case for zero boundary damage, was the *worst* point in the
sweep — so B's value is throughput, not boundary quality.

---

## 2. Honest overall assessment

The estimates sum naively to ~1.3% against a 1.21% gap. **They will not add
linearly** — they overlap heavily (T2 and T4.1 fight over the same evaluations;
T3 and T6 over the same route count).

Realistically:

- **Beating FILO2 on time: plausible, likely achievable.** T0.1 + T1 + T2 alone is
  a 2–5× inner-loop speedup before counting P cores.
- **Beating FILO2 on cost: genuinely hard.** T0+T1+T2+T3+T4 realistically closes
  **50–70% of the gap**, landing ~0.4–0.6% behind on cost at substantially lower
  time. Beating on cost *outright* additionally needs Stage 6-B or 6-C.
- **Total program: 2–3 months.**

If the honest end state is "we match FILO2 on cost at 3× less wall-clock on P
cores", that is a defensible, reportable result — arguably a better one than a
marginal cost win.

---

## 3. Suggested order, with decision gates

```
Stage 1  T0.1 T0.2 T0.3 T0.4     2-4 days   GATE: what is the real gap at equal time?
Stage 2  T1, T4.2                1-1.5 wk   GATE: did time drop >=15%?
Stage 3  T3 (ROUTEMIN, properly) 1 wk       GATE: did route count reach 800?
Stage 4  T2 (+T4.1)              2-4 wk     GATE: dist_calls/iter down >=3x?
Stage 5  T5.1, T5.2, T6          1.5-2 wk
Stage 6  C first, then B if time
```

Each gate is a real stop-or-continue point. Stages 1–3 are worth doing regardless
of how far the program ultimately goes; Stage 4 is the commitment point.

## 4. Immediate next actions

1. **T0.1** — delete the profiling counter from the hot path, re-benchmark VDA.
   One afternoon, measurable.
2. **T0.4** — build `filo2` with `-DTIMELIMIT`, run at 94 s solver-time on VDA,
   5 seeds. This tells us what we are actually chasing.
3. **T4.2** — adaptive ω; independent of everything, 2 days.

---

## 5. Progress log

**Stage 1 (T0.1–T0.4): done.** T0.1 (profiling counter compiled out), T0.2
(kNN symmetrization), T0.3 (randomized recreate order) implemented and
verified (determinism, feasibility on small/medium/Lazio). VDA cost flat vs.
baseline within noise (~22.0M, seed spread ~0.47%).

T0.4 (FILO2 `-DTIMELIMIT` build, 5 seeds @ 102s = our actual VDA solver time):
mean cost **21,740,517**, ~800 routes every seed — barely different from
FILO2's own 140s/100k-iteration number (21,738,306). **FILO2 converges almost
immediately; extra time buys it nothing.** The gap is real at equal time
(~1.2–1.23%), not an artifact of an unfair comparison. This is the number
everything below is measured against.

**T1 (arc-cost caching): done, real throughput win.** Added `costToPred[v]`
to `Solution`, maintained incrementally in `remove_customer`/`insert_customer`
/`apply_undo_list` by reusing dist() values callers already compute for the
delta (not recomputing) — cut `remove_customer` from 3 dist() calls to 1,
`insert_customer` from 3 to 2. Rewired all 9 local-search operators'
eval_* functions (the ~270-evaluation hot loop the plan's root-cause section
identified) to read `costToPred` for "current edge" terms instead of calling
dist(). **Isolated result: +19–23% iteration throughput on VDA** (94,845/
137,989 vs. 79,709/112,249 baseline, per-worker, same 40s budget), determinism
and feasibility verified (X-n1001-k43, test_2000, Lazio — byte-identical cost
across repeats, verifier.py cost matches on all three).

Found and fixed a real bug along the way: `stage4_route_cleanup`
(`Stage4_5_CleanupPolish.cpp`) mutates the route linked-list directly,
bypassing `remove_customer`/`insert_customer` — it was leaving `costToPred`
stale for Stage 5, and this actually manifested (Lazio: reported cost
3,183,135,074 vs. independently-verified 3,183,224,511, a ~0.003% desync)
before being fixed.

**T4.2 (adaptive ruin-walk ω): implemented, then disabled — net negative.**
Measured cost effect was consistently within noise (best case ~0.04% better
than baseline, worst case ~0.03% worse, across omega caps from 2x to 4x
base), but the throughput cost was not: 35–47% fewer iterations than T1
alone, because a larger ruin walk means proportionally more recreate +
local_search work per iteration. This fails the plan's own Stage 2 gate ("did
time drop ≥15%?") outright when combined with T1. Left in the code behind a
`constexpr bool kEnableOmegaAdaptation = false` rather than deleted.

**Net Stage 1+2 result on VDA (T1 only, T4.2 off): mean cost 22,003,614,
gap-to-FILO2-at-equal-time ~1.21%** — essentially unchanged from the T0.4
baseline (as expected: T1 is a pure speed lever, not a cost lever by itself).
The throughput win it bought is what T3 (ROUTEMIN) and T2 (SMD rewrite) are
meant to spend.

**T3 (ROUTEMIN done FILO2's way): implemented correctly, disabled — net
negative given our current local_search.** Ported per-chunk (runs once per
chunk right after construction, before that chunk's stage2_ils), matching
FILO2's mechanics exactly: FFD bin-packing `kmin` target, destroy-two-routes
ruin, annealed-probability partial-solution tolerance, cost-primary accept
rule with a route-count tiebreak. Found and fixed a real ordering bug during
implementation: `rescan_touched_routes` (which refreshes `cumLoad`) must run
*before* `local_search`, not after — calling it after left `local_search`'s
`eval_2opt_star` reading stale `cumLoad` for just-reinserted routes, which
produced a genuine capacity-check false-pass (a `[FATAL] insert_customer
load > Q` crash, caught on X-n1001-k43). Fixed and re-verified: deterministic,
feasible on X-n1001-k43/test_2000/VDA/Lazio (Lazio at `-p 4`, 40,438 routes,
cost matches independently-recomputed).

Measured on VDA (5 seeds): mean cost **22,038,711 vs. 22,003,614 without
T3 (+0.16% worse)**, and route count moved the wrong way (806.4 avg vs. 805
baseline, target was 800). Root cause is almost certainly architectural, not
a bug: FILO2's ROUTEMIN leans on its much richer local-search move set (22
operator types — E30/E31/E32/E33, RE-variants, SPLIT, TAILS, etc., all at
γ=1.0) to make whole-route destruction pay off; our 9 operators can't
compensate as well, so the cost-primary accept rule keeps drifting toward
cheaper-but-more-numerous route configurations instead of consolidating.
This matches T2's own note that adaptive γ "is meaningless without T2" — T3
may have the same dependency. Disabled via `g_routemin_iterations = 0`
default (CLI: `--routemin-iters`), code left in place and verified-safe for
when T2 exists.

**Net Stage 1-3 result on VDA: unchanged from T1-only (mean cost 22,003,614,
gap ~1.21%)** — T3 contributed nothing net-positive in this architecture.

**T2 ("T2-lite"): implemented, verified correctness-neutral, disabled —
targeted the wrong hot spot.** Full FILO2-faithful SMD (separate heap per
operator, i-side/j-side algebraic delta factoring, cache persisted across the
whole SA run with do/undo-list integration) was scoped down, by agreement, to
a smaller, lower-risk subset: a per-(i,j)-candidate-pair delta cache
(`PairCacheEntry` in `ThreadArena.hpp`) for `local_search`'s Step 2 (the
9-operator × k-neighbor eval sweep the plan's root-cause section identified),
scoped to a single `local_search`-to-convergence call so it never has to
interact with `apply_undo_list`'s rollback. Generation-stamped (an established
idiom already used by `rescan_touched_routes` in this file) for O(1)
invalidation on a new call, plus a reverse-index-driven per-entry invalidation
(`NeighborLists::reverseIdx`) when a touched vertex is any node's cached
candidate. A real correctness gap was caught and fixed *before* it shipped:
capacity checks depend on `routeLoad[r_i]`/`routeLoad[r_j]`, which can change
from a move that touches neither `i` nor `j` (some other customer added to
the same route elsewhere) — closed by snapshotting both route loads in the
cache entry and re-checking them on every hit.

Correctness: verified exactly — byte-identical final cost vs. the uncached
baseline on repeat runs at a fixed seed (74832 on X-n1001-k43, 56421 on
test_2000), feasible at Lazio scale (`-p 4`, 40,446 routes, cost matches
independently-recomputed). This is strong evidence the caching logic itself
is sound.

Performance: **net negative**, for a different reason than T3/T4.2. A
dist()-call-count comparison (`-DPROFILE_DIST`) showed cached vs. uncached
gave essentially the same dist() calls/iteration (~85k/67k either way) — the
cache wasn't reducing work at all. Root cause: Step 1 (`get_top3_insertions`,
the SWAP* precompute that walks a candidate route's *entire* customer list)
is the actual dominant cost, not the Step 2 evals this cache targets, and
Step 1 was never cached by this design. With no work actually saved, the
caching overhead (generation checks, routeLoad snapshot comparisons,
reverse-index invalidation walks on every applied move) was pure loss: VDA
iteration throughput dropped to 68,837/96,331 vs. 93,928/142,751 without it
(same 40s budget, T3 already disabled in both). Disabled via `constexpr bool
kEnablePairCache = false` in `stage2_ils` (`Stage2_ILS.cpp`), implementation
left in place and Lazio-verified-safe for a follow-up.

**A real full T2 would need to target Step 1, not Step 2** — a
`(customer × route)`-shaped cache (not `(customer × candidate-index)`),
substantially larger and differently invalidated, closer in spirit to
FILO2's actual per-operator SMD heaps than what was tried here. That's a
bigger, separate undertaking, not a small follow-up to this attempt.

**Net Stage 1-4 result on VDA: unchanged from T1-only (mean cost 22,003,614,
gap ~1.21%)** — three of four attempted levers (T3, T4.2, T2-lite) were
implemented correctly, verified safe, and found net-negative or neutral in
this specific architecture; only T1 (throughput) and T0.1-T0.3 delivered a
measurable, kept win. The honest state of the program: cost gap to FILO2 is
essentially where T0.4 found it, and every attempted lever beyond raw
throughput has run into the same wall — our 9-operator local_search is
meaningfully weaker than FILO2's ~22-operator one, and cost-side techniques
(ROUTEMIN, adaptive shaking, pair caching) all depend on search quality/speed
this architecture doesn't have yet. The remaining honest options are: (a) a
real Step-1-targeted T2 (large, high-risk), (b) T5.2 (add the missing
operator types outright, cheaper than a full SMD rewrite and directly
addresses the "weaker local_search" root cause), or (c) accept the
throughput-only result as the deliverable and pursue Stage 6's parallel
architecture options for a time-side win instead.
