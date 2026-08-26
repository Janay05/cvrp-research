# Report 010 — Can this architecture beat FILO2 on both time and cost?

Date: 2026-08-26. Status: **VERDICT WITHDRAWN — see §0.** Original verdict was
"no, dead end"; a later measurement in the same session invalidated the basis
for it. All measurements in §1–§5 stand; the conclusion drawn from them in §6
does not.

---

## 0. Verdict withdrawn — the ceiling was measured with a broken tool

This report originally concluded the architecture is a dead end, resting on
"cost is flat in P" and "cost is flat in time" (§2.1, §2.2) as evidence of an
architectural quality ceiling.

Then I decomposed the gap (`tools/decompose_gap.py`) and found:

| Valle-D'Aosta, ~102 s | ours (seed 2) | FILO2 (seed 11) | delta |
|---|---|---|---|
| routes | 805 | 800 | **+5** |
| depot-leg cost | 20,784,004 | 20,607,561 | +176,443 (**97.5 % of gap**) |
| inter-customer cost | 1,123,433 | 1,118,940 | +4,493 (2.5 % of gap) |
| total | 21,907,437 | 21,726,501 | +180,936 (+0.833 %) |

**Our customer sequencing is at parity with FILO2 — within 0.4 %. Essentially
the entire cost gap is surplus vehicles**: 5 extra routes at ~35,289 of depot
round-trip apiece. That directly contradicts the "weaker neighbourhood /
weaker local search" diagnosis in §3 and in report 009.

The one technique aimed squarely at vehicle count is ROUTEMIN. Our port (T3)
was measured as net-negative and disabled. Re-examining it found **two defects
in the port, not in the architecture**:

1. **Live-route counting (fixed).** `stage1_5_routemin` steered on
   `Solution::numRoutes`, which is an allocation high-water mark, not a live
   count — `remove_customer` empties a route but nothing decrements
   `numRoutes`, and `open_route` only increments it. `main.cpp` reports the
   *live* count, so the two are different quantities. All four route-count
   decisions (kmin early-exit, the open-new-route-vs-leave-unserved gate, the
   accept-fewer-routes tiebreak, the kmin stop condition) plus the log line
   read a number that **cannot decrease**. That is why T3 appeared to *add*
   routes (810 → 831 at P=1). Fixed via `count_live_routes()`; the same run now
   reports 810 → 814 and cost improves 22,050,223 → 21,970,454.
2. **Neighbourhood width (identified, not fixed).** Our port runs ROUTEMIN
   against the k=30 granular list. FILO2 runs it at **gamma = 1.0 over a list
   of up to 1500** — *"We are going to use all the available move generators
   for during this procedure"* (`opt/routemin.hpp`) — a ~50× wider candidate
   set, precisely during the phase whose job is finding residual capacity
   anywhere in the solution to absorb customers from destroyed routes. With
   only 30 candidates, most of their routes are full, reinsertion fails, and we
   open a new route instead. I had written this narrowing off in a code comment
   as "acceptable" without ever measuring it.

**Why this invalidates the verdict:** every flat-in-P and flat-in-time
experiment was run with route minimization broken *and* disabled. They
establish the ceiling of *this pipeline without working route minimization* —
not an architectural ceiling. And the defect they were used to rule out is
precisely the one that accounts for 97.5 % of the gap.

Rough headroom, arithmetic not measurement: eliminating the 5 surplus routes
recovers ~176,000 of the ~181,000 gap, landing between +0.02 % (marginal
depot cost per route) and +0.24 % (average depot cost per route) of FILO2 —
i.e. most of the gap, with 2 cores against their 1.

**Open questions that a real verdict needs answered:**
- Does ROUTEMIN reduce routes toward kmin once given FILO2's neighbourhood
  width? (Not yet tested.)
- Is there a genuine *architectural* route-count penalty at P>1? Bin packing
  is not additive — `kmin(A ∪ B) ≤ kmin(A) + kmin(B)` — so per-chunk route
  minimization cannot reach the global optimum. This is a real partitioning
  cost, but it is bounded by the number of chunks and we are currently 5
  routes above kmin at P=1, where no such penalty exists.
- Do the flat-in-P / flat-in-time ceilings move once route minimization works?

Until those are answered, **"dead end" is not a supportable claim.** What is
supportable: as it stands today FILO2 wins on both axes, the dominant defect
is vehicle count, and our tool for it is mis-ported.

---

## Original report follows (measurements valid; §6's conclusion withdrawn)

The question this report answers is narrow and deliberately not softened: *is
the current architecture (Hilbert-curve spatial partitioning → independent
parallel per-chunk ILS → boundary healing → serial polish) capable of beating
FILO2 on time **and** cost?* Not "is it a defensible result", not "does it win
on some axis" — can it win on both.

**Answer: no.** Not marginally, and not with more engineering effort of the
kind attempted so far. The reasoning is below, entirely from measurements
taken for this report.

---

## 1. The decisive measurement

Every prior comparison in this repo ran each solver to *its own* stopping
point, then compared. That answers "who is better in their own time", not
"who is better per unit of time". The right test is **iso-quality**: how much
time and compute does each solver need to reach a given solution quality?

FILO2's cost-vs-time convergence on Valle-D'Aosta (seed 1, single core,
`--optimization-seconds` sweep, `results/bench/filo2_convergence_vda/`):

| FILO2 budget | Cost | Note |
|---|---|---|
| 3 s | 21,932,131 | log says `Running COREOPT for 0 seconds` — construction + ROUTEMIN only |
| 5 s | 21,932,131 | still ~0 s of core optimization |
| 10 s | **21,824,646** | ~6 s of actual core optimization |
| 20 s | 21,790,302 | |
| 40 s | 21,766,836 | |
| 102 s | 21,738,409 | |

Against that, **our solver's best result ever recorded** — the single lowest
cost across every configuration, seed, time budget and core count in every
committed benchmark CSV in this repo:

> **21,923,585** — Valle-D'Aosta, seed 2, `-p 2`, 102 s solver time, 805
> routes, feasibility verified (`results/bench/t1_final_vda_p2.csv`).

**FILO2 reaches a better solution than our all-time best in 10 seconds on one
core** (21,824,646 vs 21,923,585 — 0.45 % better). Our number cost 2 cores ×
102 s = 204 core-seconds; FILO2's cost 10 core-seconds. **~20× less compute
for a better answer.**

At Lazio (~1 M customers) the same test is worse, not better. FILO2 pinned to
our own wall-clock (315 s, `results/bench/filo2_lazio_equaltime/`):

| | Cost | Routes |
|---|---|---|
| FILO2, **initial Clarke-Wright construction**, before any optimization | 3,177,770,000 | 40,252 |
| FILO2, final @ 315 s | **3,159,235,192** | **40,111** |
| Ours, 315 s, `-p 4` | 3,182,981,663 | 40,431 |

At Lazio, **FILO2's construction heuristic alone — zero optimization —
produces a better solution than our entire 315-second 4-core pipeline**, on
both cost and route count. Equal-time gap: **+0.75 %** (consistent with report
008's +0.771 %, so the equal-time correction changes nothing material).

---

## 2. Three ways out, all measured and all closed

### 2.1 More cores? No — cost is flat in P

From the existing P-sweep (`results/bench/011_sweep_p*.csv`, VDA, ~102 s):

| P | Cost range | Routes |
|---|---|---|
| 1 | 22.09–22.15 M | 806–809 |
| 2 | **21.95–22.08 M** | 804–808 |
| 4 | 22.06–22.07 M | 807–809 |
| 8 | 22.02–22.09 M | 807–812 |
| 16 | 22.06–22.10 M | 807–808 |

Cost is essentially **flat from P=1 to P=16**, and P=1 — the configuration
with *zero* partitioning penalty — is the worst point. The architecture's
stated central premise (`docs/reports/algorithms_and_speed_analysis.md`) is
that partitioning means "each thread executes the same search effort a
single-threaded solver would spend on the whole problem", i.e. that
parallelism multiplies search effort into better cost. **Our own data
falsifies that**: the extra cores buy iterations that do not convert into
quality.

### 2.2 More time? No — we are converged, not time-starved

VDA, `-p 2`, time budgets scaled 4.6×:

| Wall time | Cost |
|---|---|
| 102 s | 22,000,544 (5-seed mean) / 21,923,585 (best ever) |
| **471 s** | **21,979,044** |

4.6× the time bought ~0.1 %, and still did not beat our own 102 s best. The
search has converged to a quality ceiling around **21.92–22.0 M**. FILO2's
*floor* at 102 s is 21.73 M. **Our ceiling is above their floor.**

### 2.3 Multi-start (Stage 6-C)? No — it is symmetric

This is the correction that matters most, and it invalidates a claim I made
earlier in this program. Stage 6-C gave us best-of-12 = 21,966,881, which I
reported as closing the gap 1.20 % → 1.04 %. **That comparison was not
apples-to-apples**: it compared *our* best-of-12 against FILO2's *single-run*
mean. FILO2 is also a seeded randomized solver and is single-threaded, so on
the same 28-core machine it gets the identical trick for free.

Measured (`results/bench/filo2_multistart_vda/`, 12 seeds × 102 s, 102 s wall):

| | Best-of-12 @ 102 s |
|---|---|
| FILO2 | **21,729,621** |
| Ours | 21,966,881 |

Like-for-like, the gap is **1.09 %**, not 1.04 %. Multi-start is not a
competitive advantage — it is a technique both solvers get, and it helps us
slightly more only because our seed-to-seed variance is larger (~0.29 % spread
vs FILO2's ~0.15 %), which is itself a symptom of a less reliable search.

---

## 3. Why — and a correction to report 009's stated root cause

**Report 009's root-cause claim was wrong, and this report retracts it.**

Report 009 stated our local search performs "~103,000 distance evaluations per
ILS iteration against FILO2's *low hundreds*", concluding that we "cannot
afford enough iterations to close the cost gap". The 103,000 figure was
measured and is real. **FILO2's "low hundreds" was never measured — it was an
assertion, and it is off by roughly three orders of magnitude.**

Measured directly for this report by instrumenting FILO2's `Instance::get_cost`
with a call counter (temporary; reverted after measuring). Isolating core
optimization by differencing a 60 s run against a 4 s run whose log confirms
`Running COREOPT for 0 seconds`:

| | Distance evaluations per ILS iteration |
|---|---|
| Ours (Stage 2, current build, VDA) | **~78,300** (92,028 worker 0 / 64,551 worker 1) |
| FILO2 (COREOPT, VDA) | **~73,900** (5,121,753,273 calls ÷ 69,341 iterations) |

**A ratio of 1.06× — essentially identical, not 300×.** The premise that we are
computationally out-classed per iteration is false.

Iteration throughput points the *same* way, against us:

| | Iterations/s per core | Ruin size (omega) | Customers ruined/s/core |
|---|---|---|---|
| Ours | ~2,742 | ~9.2 (fixed, `ceil(ln n)`) | ~25,200 |
| FILO2 | ~1,150 | ~23 (adaptive, observed) | ~26,450 |

Per core the two are within ~5 % of each other on actual search work done —
and **we run 2 cores to FILO2's 1**, so we deploy roughly *twice* FILO2's
total search effort and still finish 1.09 % behind. FILO2 does use ~2.6× fewer
distance evaluations per *unit of destruction* (3,211 vs 8,510 per ruined
customer), which is a real efficiency edge — but 2.6× is nowhere near enough
to explain the outcome, and we more than compensate for it with parallelism.

So the binding constraint is **not** throughput, not per-iteration cost, and
not distance-evaluation efficiency. It is **what FILO2 does with each unit of
work** — search effectiveness, not search speed. The supporting evidence:

1. **Their starting point ≈ our finishing point.** FILO2's construction +
   ROUTEMIN, using 186 M cost calls and ~4 s, yields 21,932,131 at VDA. Our
   fully converged best is 21,923,585. At Lazio their construction alone
   *beats* our final result outright (§1). We spend our entire search budget
   getting to roughly where FILO2 begins.
2. **Neighbourhood richness**: FILO2 applies 22 operator types including
   ejection chains (depth ~25), SPLIT, TAILS and 3-segment exchanges; we have
   11 and no ejection chain. Once easy moves are exhausted, they have moves
   available that we simply do not.
3. **Working adaptive control**: FILO2's gamma (~0.27) and omega (~23) adapt
   during the run. Our attempts to replicate that (T4.2) measurably hurt.

This correction makes the verdict *stronger*, not weaker: we cannot compute
our way out of the gap, because we already out-compute FILO2 and still lose.

Four separate attempts to close the gap this program (all implemented, verified
correct, and measured):

| Attempt | Result |
|---|---|
| T4.2 adaptive ruin size | −35–47 % throughput, no cost gain. Disabled. |
| T3 ROUTEMIN | +0.16 % *worse* cost, route count moved away from target. Disabled. |
| T2-lite move-eval cache | Correct but cached the wrong stage; no dist() reduction. Disabled. |
| T5.2 E21/E22 operators | **+0.014 %.** Kept. |

Combined measurable gain from four attempts: **~0.014 %**, against a 1.09 %
gap. A secondary symptom is route count: we never produce fewer than 804
routes at VDA under *any* P, seed or time budget; FILO2 routinely reaches
800–801 (and 40,111 vs our 40,431 at Lazio). Route consolidation requires a
local search strong enough to repair the disruption ROUTEMIN causes — which is
precisely what we lack, and precisely why T3 failed.

---

## 4. Why this is an architecture verdict, not a tuning verdict

Given §3, closing 1.09 % is not a matter of making our existing search faster —
it is already fast enough, and faster than FILO2 in aggregate. It requires
making it *better*: a competitive construction (Clarke-Wright + a route
minimization that actually works), a materially richer neighbourhood (ejection
chains and the 3-segment/SPLIT/TAILS families), and adaptive control that
functions. That is not a tuning exercise — it is FILO2's algorithm.

The decisive point is what happens *after* you do all that:

1. **Partitioning would still contribute nothing.** §2.1 shows cost is flat
   from P=1 to P=16 *today*, and the reason is now clear: throughput was never
   the binding constraint, so buying more of it changes nothing. Improving
   search quality does not make partitioning start paying — if anything it
   raises the value of the *unconstrained* global moves that chunking forbids
   at boundaries.
2. **So the distinctive layer becomes pure overhead.** Hilbert partitioning is
   what makes this architecture ours rather than a FILO2 reimplementation. It
   would be doing no useful work while still constraining the solution space
   at chunk boundaries and forcing the Stage 3 healing machinery to exist.

So the end state of "fixing" this architecture is *FILO2's algorithm with a
parallel wrapper that measurably does not help* — at which point the thing
that made this architecture ours has been removed for no gain. That is the
definition of a dead end for the stated aim.

There is a genuinely different design that could win — FILO2-quality global
localized search parallelised by optimistic concurrency (report 009's Stage
6-B: per-route locks, abort/retry on conflict, ~50 vertices touched per
iteration so conflicts are rare). That is a **new architecture**, budgeted at
3–4 weeks with real concurrency-correctness risk, not an extension of this
one.

---

## 5. What this architecture *does* win

Stated plainly so the verdict is not mistaken for "everything was worthless":

- **Memory at scale**: 3.5 GB vs FILO2's 7.04 GB peak at Lazio — a 2× win,
  after this session's allocation fix. FILO2's memory grows with its 1500-wide
  neighbour lists; ours does not.
- **Predictable wall-clock**: time-budgeted by construction, so runtime is
  tight (103.4–103.6 s) where FILO2's iteration-budgeted design varies 87–194 s
  across seeds at VDA.
- Verified engineering results along the way: T1's +19–23 % throughput, the
  costToPred caching, the Stage-4 desync fix, the ~50 % memory fix.

None of these change the verdict on the stated aim.

---

## 6. Verdict

**This architecture cannot beat FILO2 on both time and cost.** The gap is not
incremental: FILO2 beats our all-time best result in 10 seconds on one core
(~20× less compute), and at Lazio its *construction heuristic alone* beats our
full 315-second 4-core pipeline. Cost is flat in P, flat in time, and
multi-start is symmetric — the three levers this architecture has are all
exhausted.

And the reason is not the one report 009 gave. We are **not** computationally
out-classed: per-iteration distance-evaluation cost is within 6 % of FILO2's,
and in aggregate we deploy roughly *twice* their total search work (§3). We
lose anyway. Throughput — the one thing partitioning buys — is not the binding
constraint and never was, which is exactly why cost is flat in P. Closing the
remaining ~1.09 % means adopting FILO2's *algorithm* (construction, ejection
chains, working adaptive control), at which point the partitioning that
defines this architecture is inert overhead.

Recommendation: report this as a dead end for the "beat FILO2 on both axes"
aim, and treat Stage 6-B (optimistic-concurrency global search) as a separate
architectural proposal to be scoped on its own merits, not as a continuation.
