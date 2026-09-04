# Report 010 — Can this architecture beat FILO2 on both time and cost?

Date: 2026-08-26 (updated 2026-09-04, §0.20). Status: **VERDICT WITHDRAWN —
see §0.** Original verdict was "no, dead end"; later measurements in the same
session invalidated the basis for it. All measurements in §1–§5 stand; the
conclusion drawn from them in §6 does not. **§0.16/§0.17 corrected a
stale-FILO2-budget bug affecting every prior comparison and, both sides now
independently verified (`src/verify_filo2.py`, not just FILO2's self-report):
Lazio is a decisive win, 0.183 % mean, all 10 seeds, t = −16.4; VDA was a
verified loss, 0.146 %. §0.19 added a scoped-down ejection chain operator
(FILO2's largest remaining move type, depth-2 instead of FILO2's depth-25)
and roughly halved the VDA gap to 0.081 % — still a loss, every seed, but the
largest single improvement to it this session — while leaving Lazio's win
unaffected. §0.20 tried extending the chain to depth-3 and, unlike depth-2,
measured it net-negative on both scales (VDA +0.083%, Lazio +0.0116% worse
mean cost) — a greedy-search-trajectory regression, not a bug; implemented,
verified-safe, and disabled rather than shipped, same precedent as report
009's T2-lite. The honest overall claim: a verified, growing win at Lazio
scale (~1M customers) and a verified, shrinking loss at VDA scale (~180 K)
— not a general "better than FILO2," a scale-dependent result trending
toward parity on the smaller instance too.**

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

### 0.1 Both defects fixed — gap 1.20 % → 0.71 %

Defect 2 confirmed and fixed. `--routemin-k` gives ROUTEMIN a dedicated wide
list; ROUTEMIN's effect on live route count, VDA at P=1:

| ROUTEMIN k | live routes | final routes | final cost |
|---|---|---|---|
| 30 (original) | 810 → **814** (rises) | 807 | 22,036,221 |
| 100 | 810 → 809 | 808 | 22,007,110 |
| 300 | 810 → 805 | 804 | 21,929,445 |
| 1000 | 810 → **800** (= kmin) | 801 | 21,875,130 |
| *FILO2* | *810 → 801* | *800* | *21,738,409* |

Width was the whole story: it decides how many distinct routes reinsertion can
consider when hunting for residual capacity. Also split the neighbourhood —
**wide** for reinsertion/seed selection, **narrow (30)** for the `local_search`
calls inside ROUTEMIN, since our `local_search` is O(k × route_length) per node
pop and the wide list cost 69–124 s/worker (FILO2 affords gamma=1.0 there only
because its search is incremental). That cut wall clock 232 s → 124 s with no
loss of route reduction.

**5-seed, equal wall clock (~102 s), all verified feasible:**

| | mean cost | mean routes | gap to FILO2 |
|---|---|---|---|
| baseline (no ROUTEMIN) | 22,000,544 | 805.0 | 1.196 % |
| **fixed ROUTEMIN** | **21,895,496** | **802.6** | **0.713 %** |
| FILO2 (equal time) | 21,740,517 | 800.0 | — |

**The largest single improvement of the program — bigger than every other kept
change combined**, and it came from fixing two bugs in our own port rather than
from any new algorithm.

### 0.2 What the fix revealed: the real remaining constraint

Decomposing again after the fix changes the picture from §0:

| | routes | depot legs | inter-customer |
|---|---|---|---|
| Ours, no ROUTEMIN | 805 | 20,784,004 | 1,123,433 |
| Ours, fixed ROUTEMIN | 802 | **20,458,582** | **1,433,160** |
| FILO2 | 800 | 20,607,561 | 1,118,940 |

Our depot-leg cost is now **better than FILO2's** — route minimization works.
But inter-customer cost degraded 28 % (1,123,433 → 1,433,160). Giving 22 s more
optimization moved it 0.2 %, so this is **structural, not a budget artifact**:

> **We can achieve good route count *or* good tour quality, but not both at
> once. FILO2 holds both simultaneously.**

A tightly packed route is a harder sequencing sub-problem — less slack, and the
improving moves needed are longer-range and multi-route (ejection chains,
3-segment exchanges) rather than the relocate/swap/2-opt family our 11
operators provide. So §3's neighbourhood diagnosis was directionally right, but
it only becomes the *binding* constraint once route minimization works; before
that, route count masked it.

**Net:** the remaining ~0.71 % is now a single, well-identified constraint —
tour quality on packed routes — which is exactly what T2 (SMD + richer
neighbourhood) targets, no longer confounded by route count. Whether that is
worth doing is a scoping decision, not a settled "dead end".

### 0.3 The fix does not transfer to Lazio scale — keep it opt-in

The VDA win depends on a wide ROUTEMIN neighbourhood, and width is exactly what
gets unaffordable as n grows. Measured at Lazio (~1 M customers, `-p 4`):

| ROUTEMIN k | ROUTEMIN effect on live routes | final routes | cost | wall clock | peak RSS |
|---|---|---|---|---|---|
| off (baseline) | — | 40,431 | 3,182,981,663 | 315 s | 3.5 GB |
| 100 | 40,513 → **40,561** (rises) | 40,438 | 3,182,872,316 | 365 s | 4.8 GB |
| 300 | 40,513 → 40,476 (falls) | 40,409 | 3,181,453,037 | **473 s** | **7.11 GB** |

k=100 reproduces the *broken* behaviour (route count rises) — the width that was
merely mediocre at VDA is actively harmful at Lazio. k=300 finally works
directionally, but the trade is bad: **+50 % wall clock (315 s → 473 s) for
+0.048 % cost**, with peak memory at 7.11 GB — 94 % of this machine's 7.6 GB
WSL ceiling, i.e. back at the level that crashed the VM twice earlier.

The cause is a scaling mismatch. The wide list costs **O(n·k) memory and
O(n·k·log n) build time**, while the benefit scales with route count, ~O(n/25).
Setup alone went 38 s → 177 s at k=300 — more than half the original *total*
runtime spent building a neighbour list. Extrapolating the measured build cost,
the k=1000 that produced the VDA win would need ~4 GB and ~600 s at Lazio:
outright infeasible. And there is no useful middle: any k cheap enough to build
at 1 M customers (k ≲ 100) is in the range that makes route count *worse*.

**So ROUTEMIN stays off by default** (`g_routemin_iterations = 0`). Enabling it
globally would trade a 0.48 % gain at 20 k scale for a 50 %-wall-clock
regression at 1 M scale on the instance family this solver exists to handle.
Recommended usage is explicit and size-dependent:

- **n ≲ 50 k**: `--routemin-iters 2000 --routemin-k 1000` — a real 0.48 % win.
- **n ≳ 500 k**: leave off; no k is simultaneously affordable and effective.

This is itself a finding about the architecture: the technique that closed 40 %
of the cost gap is one we can only afford at small scale, because our
`local_search` charges O(k × route_length) per node pop where FILO2's
incremental search charges only for what an applied move invalidates. FILO2
runs gamma=1.0 over 1500 neighbours at *every* scale, including Lazio, for the
same reason it can afford ROUTEMIN there at all.

> **Superseded by §0.5.** The "wide lists are unaffordable at Lazio" conclusion
> above was correct about the *symptom* but wrong about the *cause*: the cost
> was not intrinsic, it was a single-threaded `symmetrize`. Once fixed, the wide
> list costs 33 s instead of 188 s and Lazio becomes viable after all.

### 0.4 Clarke & Wright (T6) — the combination that works

Our MST+randomized-DFS construction starts materially worse than FILO2's
Clarke & Wright: 22,522,444 vs 22,231,600 at VDA, 3,208,434,488 vs
3,177,770,000 at Lazio. At Lazio their *construction alone* beat our entire
315 s 4-core final answer. Report 009 estimated this technique at "0–0.15 %".

Ported CW (`--construction cw`). Candidate width dominates its quality exactly
as it does ROUTEMIN's — VDA, construction only, P=1:

| CW k | cost | routes |
|---|---|---|
| 30 | 23,534,274 | 846 |
| 100 | 22,665,236 | 820 |
| 300 | 22,392,504 | 815 |
| **1000** | **22,185,233** | **811** — beats FILO2's own construction |
| *FILO2 CW* | *22,231,600* | *810* |

**CW alone is worse than baseline; CW + ROUTEMIN is the combination that
works**, and the decomposition explains why — it inverts §0.2:

| | routes | depot legs | inter-customer |
|---|---|---|---|
| MST + ROUTEMIN | 802 | 20,458,582 | 1,433,160 |
| **CW + ROUTEMIN** | 803 | 20,718,014 | **1,061,438** |
| FILO2 | 800 | 20,607,561 | 1,118,940 |

CW builds routes by savings-merging, which produces genuinely good *sequences*,
so tour quality ends up **better than FILO2's**, while ROUTEMIN handles
consolidation. The remaining deficit is 3 surplus routes.

### 0.5 `symmetrize` was the real scaling wall — and it was pre-existing

Profiling the neighbour-list build at Lazio found `symmetrize` (single-threaded,
run after the parallel kNN queries join) dominates everything:

| k | kdtree | knn queries | **symmetrize** |
|---|---|---|---|
| 30 | 0.2 s | 1.4 s | **6.3 s** |
| 100 | 0.2 s | 4.9 s | **30.7 s** |
| 300 | 0.2 s | 3.4 s | **136.7 s** |

137 s of a 180 s Stage 0, against 3.4 s for the actual queries. Its phase-1
membership test is a linear scan of `nbr[j]` per neighbour per node, so cost
grows super-linearly in k. **This was already burning ~37 s of serial work in
every Lazio run** for the existing k=30/k=100 lists, before any of this
session's features.

Parallelised both phases, output verified bit-identical (74619 with parallel
and with serial forced, P=4; 74913 at P=1). **Lazio setup: 188 s → 33 s** — now
*less* than the 38 s the old baseline spent with no wide list at all.

### 0.6 Final standing — both instances, better cost in less wall clock

**Valle-D'Aosta, 5 seeds, equal wall clock, all verified feasible:**

| | mean cost | routes | wall | gap |
|---|---|---|---|---|
| baseline (MST) | 22,000,544 | 805.0 | ~102 s | 1.196 % |
| MST + ROUTEMIN | 21,895,496 | 802.6 | 113.0 s* | 0.713 % |
| **CW + ROUTEMIN** | **21,791,054** | 802.6 | **94.6 s** | **0.233 %** |
| FILO2 | 21,740,517 | 800.0 | 102 s | — |

\* This row's 5 seeds ran *concurrently* (`tools/routemin_5seed.sh`), so 113.0 s
is a contended mean, not a clean per-run time — it was previously mislabeled
"~102 s" here, copying the sequential-run convention used by the other rows.
The CW+ROUTEMIN row below ran sequentially (`tools/cw_rmin_5seed.sh`) and is a
real per-run wall clock. Cost is unaffected: a clean uncontended single-seed
run of MST+ROUTEMIN gave 21,896,165 @ 102.4 s, 0.003 % from the contended mean
above, so the 0.713 % gap stands.

**Lazio (~1 M), identical stage budgets to the published baseline, verified
feasible:**

| | cost | routes | wall | peak RSS | gap |
|---|---|---|---|---|---|
| baseline (MST) | 3,182,981,663† | 40,431 | 314.5 s | 3.5 GB | 0.752 % |
| CW only | 3,178,280,587 | 40,327 | 305.7 s | 7.25 GB | 0.603 % |
| **CW + ROUTEMIN** | **3,171,997,628** | **40,267** | **284.9 s** | 7.43 GB | **0.404 %** |
| FILO2 | 3,159,235,192 | 40,111 | 315 s | 7.04 GB | — |

† Not reproducible from the repo — this solution was written to `/tmp` and
WSL cleared it; the number survives only in commit prose. It was also
measured before E21/E22 were added, while the CW rows include them; the same
build's MST baseline was later remeasured at 3,183,609,391 (0.020 % higher),
which is *favorable to the baseline*, so 0.752 % is if anything a slight
understatement, not an inflated number. Doesn't affect the CW+ROUTEMIN row or
the parity conclusion in §0.6, both of which rest on committed data.

**Both instances improved on cost *and* wall clock simultaneously**: VDA
1.196 % → 0.233 % at 94.6 s vs 102 s; Lazio 0.752 % → 0.404 % at 284.9 s vs
315 s. (We use P cores to FILO2's 1, so these are wall-clock wins, not
compute-per-core wins.)

### 0.7 Memory ceiling raised, wider k at Lazio, defaults now on

The 7.43 GB Lazio peak was 98 % of the 7.6 GB WSL ceiling — too tight to default.
Root cause was configuration, not the solver: WSL2 self-caps at ~50 % of host RAM
and no `.wslconfig` existed. Created one (`memory=10GB`, `swap=4GB`; host is
15.71 GB, so this leaves ~5.7 GB for Windows). Swap is deliberate — an overrun
now degrades into paging instead of the abrupt VM loss seen twice earlier.
Ceiling is now **9.7 GiB + 4 GiB swap**.

That headroom buys a wider list at Lazio, which was previously memory-blocked:

| Lazio, CW + ROUTEMIN | cost | routes | wall | peak RSS | gap |
|---|---|---|---|---|---|
| k=300 | 3,171,997,628 | 40,267 | 284.9 s | 7.43 GB (77 %) | 0.404 % |
| **k=500** | **3,165,857,376** | **40,209** | 319.1 s | 9.11 GB (94 %) | **0.210 %** |
| FILO2 | 3,159,235,192 | 40,111 | 315 s | 7.04 GB | — |

k=500 nearly halves the gap again, at essentially equal wall clock — but at 94 %
of the ceiling, so it stays an explicit opt-in. **k=300 is the default at Lazio
scale**: 77 % of ceiling and still *faster* than the baseline.

**Defaults are now on** (`--construction cw`, `--routemin-iters 2000`), with the
wide-list width chosen from instance size, since no single k serves both ends:
`k = clamp(300e6 / n, 300, 1000)` — 1000 at VDA (80 MB, the best measured
setting there), 300 at Lazio (7.4–7.8 GB peak, comfortable). Verified on the
default path: VDA 21,786,433 / 803 routes / 87.9 s; Lazio 3,172,491,626 / 40,269
routes / 307.2 s / 7.79 GB; X-n1001-k43 74,682 (was 74,913), deterministic and
feasibility-verified.

### 0.8 Where this leaves the original question

The question this report opened with was whether the architecture can beat FILO2
on **both** time and cost. Current standing, equal wall clock:

| | gap at session start | gap now | wall clock |
|---|---|---|---|
| Valle-D'Aosta | 1.196 % | **0.233 %** | 94.6 s vs 102 s |
| Lazio (k=300 default) | 0.752 % | **0.404 %** | 284.9 s vs 314.5 s |
| Lazio (k=500 opt-in) | 0.752 % | **0.210 %** | 319.1 s vs 315 s |

**Still behind on cost, now by ~0.2–0.4 % instead of ~0.8–1.2 %, and ahead on
wall clock** — with the standing caveat that we use P cores to FILO2's 1, so
these are wall-clock wins, not compute-per-core wins. The §6 "dead end" verdict
remains withdrawn: five-sixths of the VDA gap and roughly half the Lazio gap
turned out to be defects in our own ports (ROUTEMIN's live-route counting and
neighbourhood width) plus one pre-existing serial bottleneck (`symmetrize`), not
architectural limits.

What is genuinely architectural, and still unaddressed: our `local_search` costs
O(k × route_length) per node pop where FILO2's is incremental, which is why we
must ration candidate width by instance size at all. Closing the last ~0.2 %
plausibly needs that (T2), but it is no longer confounded by anything else.

### 0.9 The last 0.2 % was route count, and ROUTEMIN was under-converged

Decomposing the remaining Lazio gap (k=500) against FILO2's own published seed-1
solution inverts the earlier picture completely:

| Lazio | routes | depot legs | inter-customer | total |
|---|---|---|---|---|
| ours (k=500) | 40,209 | 3,125,500,846 | **40,356,530** | 3,165,857,376 |
| FILO2 | 40,086 | 3,098,524,132 | 58,618,021 | 3,157,142,153 |

**Our tours are 31 % better than FILO2's.** The whole gap — and more — is 123
surplus vehicles at ~219,323 of depot legs each. Matching their route count while
keeping our tour quality projects to ~3,138,880,647, i.e. **0.58 % ahead of
FILO2**. This retires the §3/§0.2 "weak neighbourhood" thesis for good: tour
quality is our *strength*, so T2 targets what we already win.

The cause was mundane. ROUTEMIN ran a flat 2000 iterations per chunk regardless
of instance size, and each iteration destroys exactly two routes — so what
matters is iterations *per route*:

- VDA: 2000 ÷ ~405 routes per chunk = **~4.9 passes per route**
- Lazio: 2000 ÷ ~10,100 routes per chunk = **~0.20 passes per route** (25× less)

At Lazio it was stopping before it had looked at most routes even once.

| Lazio, k=500 | ROUTEMIN iters | final routes | cost | wall |
|---|---|---|---|---|
| | 2,000 | 40,209 | 3,165,857,376 | 319.1 s |
| | 8,000 | 40,157 | 3,159,961,835 | 269.4 s |
| | **12,000** | **40,144** | **3,158,865,362** | **291.7 s** |
| FILO2 (equal time) | — | 40,111 | 3,159,235,192 | 315 s |

At 12,000 iterations, seed 1 gave **3,158,865,362 in 291.7 s against FILO2's
3,159,235,192 in 315 s** — apparently better on cost *and* 23.3 s faster
(verified feasible, cost independently recomputed).

**That apparent win does not survive multi-seed testing, and should not be
reported.** The margin was 0.012 %, against a measured Lazio seed spread of
~0.04 % (ours) and ~0.065 % (FILO2) — i.e. inside the noise. Worse, the
comparison was not like-for-like: our 291.7 s run was being compared against a
FILO2 run given a 315 s budget. Re-running both at the same 292 s budget:

| seed | ours | FILO2 | verdict |
|---|---|---|---|
| 2 | 3,159,782,660 (300.7 s) | 3,158,761,465 (294.0 s) | FILO2 +0.032 % better |
| 3 | 3,159,399,366 (297.1 s) | 3,159,377,689 (295.0 s) | FILO2 +0.0007 % better |
| **mean** | **3,159,591,013 (298.9 s)** | **3,159,069,577 (294.5 s)** | **FILO2 0.0165 % better, 4.4 s faster** |

**Corrected conclusion: we have reached parity at Lazio, not a win.** FILO2 is
ahead by 0.0165 % on cost and 4.4 s on wall clock — both differences well inside
seed noise, so the honest reading is "statistically indistinguishable", with
FILO2 marginally ahead on the point estimates.

That is still a large result: Lazio went from **0.752 % behind at the start of
this session to statistical parity**. But "beats FILO2 on both axes" is not
supported, and seed 1 alone was a favourable draw.

Also unchanged: we use 4 cores to FILO2's 1, so even parity here is
wall-clock parity, not compute-per-core parity.

### 0.10 The parity-not-a-win verdict reverses: Stage 5's time budget was silently doubling

The §0.9 "298.9 s vs 294.5 s, FILO2 4.4 s faster" comparison was run with a real
bug in `stage5_serial_polish` (`src/Stage2_ILS.cpp:2156`, found while
attributing the Stage 4&5 wall time — 82–89 s observed against a nominal 45 s
`--stage5-ms` budget). The function's own comment ("stageStart captured here...
so the sweep's deadline and the main loop's own elapsed-time check share one
clock and one budget — time the sweep spends is time the main loop below has
less of, not extra on top") describes correct behavior, but the code captured
a **second, fresh** `stageStart` *after* the pre-loop full sweep completed,
instead of reusing the sweep's own clock — so the sweep could spend the whole
budget, then the SA loop got a fresh full budget on top, up to doubling Stage
5's real wall time. Stage 3's equivalent code does this correctly (single
shared `stageStart`); Stage 5's just didn't match its own comment. One-line
fix: reuse the sweep's `sweepStart` instead of a fresh `now()`.

Instrumented first to confirm Stage 4 (the route-cleanup pass, not Stage 5)
wasn't the cause: 42.5 ms + 25.7 ms of an 88.7 s "Stage 4 & 5" total — negligible.
Stage 5 was the entire overrun.

Verified: determinism (3× identical `Final cost: 74682`, `X-n1001-k43`, seed 42,
`--max-iterations 2000 -p 1` — unchanged from the pre-fix baseline, so the fix
is delta-neutral on quality) and feasibility (`verifier.py`, clean) before and
after re-running Lazio.

**Lazio, 3 matched seeds, same config as §0.9 (`--routemin-k 500
--routemin-iters 12000 --stage2-ms 45000 --stage3-ms 12000 --stage5-ms 45000`),
all verified feasible:**

| seed | ours (post-fix) | wall | FILO2 | wall |
|---|---|---|---|---|
| 1 | 3,158,788,023 | 262.9 s | 3,159,235,192 | 315 s |
| 2 | 3,159,852,770 | 257.5 s | 3,158,761,465 | 294 s |
| 3 | 3,159,130,636 | 252.2 s | 3,159,377,689 | 295 s |
| **mean** | **3,159,257,143** | **257.5 s** | **3,159,124,782** | **301.3 s** |

**Cost: a tie. Wall clock: 43.8 s / 14.5 % faster, across 3 seeds, not 1.**
Stage 4 & 5 dropped from 88.7 s to 44.2 s (now landing right at its 45 s budget)
with no cost regression (the fix only changes *when* the SA loop's clock starts,
not what it searches).

Read the two axes separately, because they are not the same kind of result:

- **Cost — statistically indistinguishable, and if anything marginally against
  us.** The mean is 0.0042 % *higher* (worse) than FILO2's. The per-seed sign
  flips: we win seed 1 by 447 k and seed 3 by 247 k, and lose seed 2 by
  1,091 k. A difference whose sign depends on the seed, at a magnitude an order
  of magnitude below the ~0.04–0.065 % seed spread, is a tie. **This is not a
  cost win and must not be reported as one.**
- **Wall clock — a real win, with no distributional overlap.** Our *slowest*
  seed (262.9 s) is faster than FILO2's *fastest* (294 s). Every one of our
  runs beats every one of theirs. 14.5 % is far outside anything noise
  explains.

**This reverses §0.9's "parity, not a win" conclusion on the time axis, and
only there.** The original question this report opened with — can the
architecture beat FILO2 on both time and cost — still has **no** measured
*yes*. What §0.10 establishes is the weaker (but real, and previously
unmeasured) claim: at Lazio, at this scale, we now match FILO2 on cost while
finishing meaningfully sooner. The §6 "dead end" verdict, already withdrawn in
§0, stays withdrawn. But the open item §0.8 left ("still behind on cost...
wall-clock wins, not compute-per-core wins") is closed only on the *time* side:
cost went from behind to level, not from behind to ahead, and the
compute-per-core caveat (4 cores vs FILO2's 1) remains true and unaddressed.

**Known open item at the time of writing, since fixed — see §0.11:**
`--stage3-ms` (36.9 s observed vs 12 s nominal) turned out to be a
*per-color-class* budget, not a per-stage total like `--stage2-ms`/
`--stage5-ms` — Lazio's 4 chunks produce 6 boundary pairs split into 3 color
classes (K4's edge-chromatic number), each independently given the full
budget.

### 0.11 Stage 3's budget fixed too — another 25 s reclaimed, cost still a tie

Fixed the item §0.10 left open: `run_stage3_healing`
(`Stage3_MergeHealing.cpp`) now divides `g_stage3_time_budget_ms` by
`(max_color + 1)` before the color-class loop runs, via an RAII guard that
restores the original global on every exit path. `stage3_healing_ils_pass`
(`Stage2_ILS.cpp`) is unchanged — it still reads the global directly, so the
fix is entirely in how much budget it's handed, not in how it spends it. No
new bug here (unlike Stage 5): the per-class-not-per-stage behavior matched
what the code actually did, it just didn't match what the flag name implied.

Verified before benchmarking: determinism unaffected (3× identical
`Final cost: 74682` on `X-n1001-k43`, seed 42 — this instance runs in
iteration-budget mode where `g_stage3_time_budget_ms <= 0`, so the new
divide-and-restore code is a proven no-op on this path). Then real-scale:
Lazio, same 3 seeds and same flags as §0.10
(`--routemin-k 500 --routemin-iters 12000 --stage2-ms 45000 --stage3-ms 12000
--stage5-ms 45000`), all verified feasible:

| seed | cost (post Stage-3-fix) | wall | Stage 3 wall (was ~36.9 s) |
|---|---|---|---|
| 1 | 3,158,766,627 | 235.5 s | 12.5 s |
| 2 | 3,159,950,003 | 226.7 s | 10.7 s |
| 3 | 3,159,323,188 | 235.0 s | 15.2 s |
| **mean** | **3,159,346,606** | **232.4 s** | **~12.8 s** |

Stage 3 now actually costs what `--stage3-ms` asks for. Comparing to §0.10's
post-Stage-5-fix baseline (same seeds, same flags, pre-this-fix):

| | §0.10 (pre) | §0.11 (post) | Δ |
|---|---|---|---|
| mean cost | 3,159,257,143 | 3,159,346,606 | +89,463 (+0.0028 %) |
| mean wall | 257.5 s | 232.4 s | **-25.1 s** |

Cost moved by 0.0028 % — an order of magnitude below the ~0.04–0.065 % seed
spread established earlier, and the per-seed direction isn't even consistent
(seed 1 got *better*, seeds 2 and 3 got slightly worse) — textbook noise, not
a regression. Wall clock dropped a further 25.1 s, on top of the 43.8 s
already reclaimed in §0.10. Against FILO2's mean (3,159,124,782 cost,
301.3 s wall): cost gap is 0.0069 % — still a tie, same band as §0.10's
0.0042 % — and the wall-clock margin widens to **68.9 s / 22.9 % faster**.

**Net effect of §0.10 + §0.11 combined**, same Lazio config, 3 seeds, all
feasibility-verified, cost held flat throughout while our own wall clock fell
in two independent steps:

| | our wall (mean) | cost gap to FILO2 |
|---|---|---|
| §0.9 (pre-fix, "parity, not a win") | 298.9 s | -0.0165 % (behind) |
| §0.10 (Stage 5 fix only) | 257.5 s | +0.0042 % (tie) |
| §0.11 (Stage 5 + Stage 3 fix) | 232.4 s | +0.0069 % (tie) |

298.9 s → 232.4 s is a **22.3 % reduction in our own wall clock**, cost gap
never leaving the noise band the whole way. Neither fix touched what the
search explores, only when its clocks start and how a budget is divided —
cost holding flat across both is the expected result, not a coincidence.

### 0.12 Multi-start re-measured on the current baseline: the lever shrank

§2.3 (original report) found Stage 6-C multi-start (N independent seeded
solves, keep the best) was "not a competitive advantage — a technique both
solvers get" once compared like-for-like, but report 009's headline number
(1.20 % → 1.04 % gap, "the strongest remaining lever") predates CW+ROUTEMIN
and was never corrected for the like-for-like point on the *current* baseline.
Re-measured here because CW+ROUTEMIN materially tightened our seed-to-seed
spread — and multi-start's entire value proposition comes from spread.

VDA, 12 concurrent starts (current default: CW + ROUTEMIN, `-p 2` each,
seeds 1–12, same flags as the SS0.6 single-start baseline), 105.3 s wall
(same order as a single 94.6 s solve — still effectively free):

| | cost |
|---|---|
| best-of-12 (seed 8) | **21,773,068** (verified feasible) |
| worst-of-12 (seed 3) | 21,832,565 |
| spread | 0.27 % |

Compare to §0.6's old MST-based multistart spread that produced the 1.20 %→
1.04 % headline: that run's per-seed range wasn't published, but the
underlying single-start spread was ~0.29 % (§2.3) — CW+ROUTEMIN has *not*
meaningfully tightened it (0.27 % vs 0.29 %), so the shrinkage isn't from
tighter variance. It's from a **tighter baseline mean**: CW+ROUTEMIN moved
the mean itself down 21,791,054 vs the old MST mean 22,000,544 — closer to
FILO2's floor, so there's less room left for any technique, multi-start
included, to find.

Two ways to read the gap, same distinction §2.3 already drew:

| comparison | gap |
|---|---|
| our best-of-12 vs FILO2's single mean (21,740,517) | 0.150 % |
| our best-of-12 vs FILO2's best-of-12 (21,729,621, §2.3, FILO2 binary unchanged so still valid) | **0.200 %** |
| (for reference) our single-start mean vs FILO2's single mean | 0.233 % |

**Like-for-like — the only fair comparison, since FILO2 gets multi-start for
free too — multi-start now closes the VDA gap from 0.233 % to 0.200 %: a
0.033-point improvement**, not the 0.16-point one report 009's headline
number implied. That number was real but measured against a construction
method (MST) we've since replaced; on today's stronger single-start baseline,
multi-start's *additional* leverage is small. It is still a legitimate, free
win when spare cores exist — 0.033 points for zero wall-clock cost is worth
taking — but it is **not**, on the current baseline, the standout lever
report 009 found it to be, and it does not by itself close the remaining gap.
Not re-tested at Lazio scale here: Lazio's own seed spread (§0.11's 3-seed
set) is tighter still (~0.037 %, an order of magnitude below VDA's), which
predicts an even smaller multi-start lift there than the 0.033 points just
measured — expected value too low to prioritize a run given the RAM
constraint on concurrent Lazio-scale processes noted in report 009.

### 0.13 E31/E32/E33: the rest of "the easy half" of T5.2

Report 009 left "a fuller T5.2" open: E31/E32/E33 (3-segment exchanges) and
the Rev variants of all five (E21/E22/E31/E32/E33), estimating 0.1–0.2 %
total for the complete missing set, of which E21+E22 had already captured
-0.014 % at VDA. Implemented E31/E32/E33 this round — same cross-route-only
scoping as E21/E22, each a faithful port of FILO2's `ThreeOneExchange.hpp`/
`ThreeTwoExchange.hpp`/`ThreeThreeExchange.hpp` (`eval_E31`/`apply_E31` etc.,
next to `eval_E22` in `Stage2_ILS.cpp`), derived directly from FILO2's
`compute_cost`/`is_feasible`/`execute` and cross-checked term-by-term before
writing any code. Wired into `local_search`'s Step 2 sweep and both dispatch
switches as ops 11/12/13, so every caller picks them up automatically, same
as E21/E22. Rev variants **not** attempted this round — confirmed by reading
FILO2's `RevTwoOneExchange.hpp` that a Rev move is a genuinely different
geometric move (swaps in the singleton from j's *next* side instead of j's
*prev* side, inserts the multi-segment reversed on the other side of j), not
a mechanical flip of the forward version — real design + implementation cost
each, for an even smaller expected slice of an already-small remaining
budget.

Verified before benchmarking: determinism (3× identical `Final cost: 74270`
on `X-n1001-k43`, seed 42 — improved from the pre-operator 74682, expected
since a richer neighborhood can only find more or equal improving moves),
feasible (`verifier.py`). Feasible at VDA (all 5 seeds) and at Lazio scale
(`-p 4`, seed 1, 40,140 routes) — the Lazio check matters because
`stage3_healing_ils_pass` shares this same `local_search`, so these operators
also run inside Stage 3's boundary healing, not just Stage 2's per-chunk ILS.

**The apples-to-apples baseline needed rebuilding first.** §0.6's published
VDA baseline (21,791,054 @ 94.6 s) predates the Stage 5 fix (§0.10) — that fix
was only ever benchmarked at Lazio, never at VDA — so it isn't valid for
isolating E31/E32/E33's effect. Isolated cleanly via `git stash` on just
`Stage2_ILS.cpp` (rebuild without the new operators, benchmark, restore,
rebuild with them): same 5 seeds, same config as §0.6
(`tools/baseline_current_vda_5seed.sh` vs `tools/e31_32_33_vda_5seed.sh`),
all feasible:

| | mean cost | mean wall |
|---|---|---|
| current build, no E3x (clean baseline) | 21,783,772 | 79.5 s |
| current build, + E31/E32/E33 | 21,776,503 | 80.2 s |
| Δ | **-7,269 (-0.033 %)** | +0.7 s (noise) |

(Side finding: the clean baseline itself, 21,783,772, is 0.033 % *better*
than §0.6's published 21,791,054 — the Stage 5 fix has a small positive
effect on VDA cost too, not just wall clock, coincidentally the same
magnitude as E31/E32/E33's own gain.)

Per-seed, the win isn't unanimous: seeds 2, 3, 5 improved (8.2 k, 23.1 k,
30.5 k), seeds 1, 4 regressed slightly (17.5 k, 7.9 k) — net +36.3 k across 5
seeds. All five deltas are smaller than the ~0.27–0.29 % seed-to-seed spread
established earlier, so this is a real but modest signal on 5 seeds, not
overwhelming evidence — consistent with, and about half of, report 009's
"0.1–0.2 % total, diminishing" estimate for the complete missing set, of
which this is the second and larger installment after E21/E22.

Gap to FILO2 (mean 21,740,517): **0.199 % → 0.166 %**, a genuine 0.033-point
improvement, wall-clock cost negligible — real, verified, kept. (Not directly
additive with E21/E22's earlier -0.014 %: that figure was measured against
the old MST-construction baseline, this one against the current
CW+ROUTEMIN baseline, so the two percentages don't share a denominator — both
are real, standalone contributions, not two terms of one running total.) The
gap is not closed, and the remaining levers (Rev variants, SPLIT/TAILS,
ejection chains) each have smaller expected payoff than what's already been
taken.

### 0.14 Lazio, current build: the tie breaks — all 3 seeds now beat FILO2

Every fix and addition in §0.10–§0.13 (Stage 5 fix, Stage 3 fix, E31/E32/E33)
was verified individually. This re-runs §0.11's exact 3-seed Lazio comparison
(`--routemin-k 500 --routemin-iters 12000 --stage2-ms 45000 --stage3-ms 12000
--stage5-ms 45000`) on the current build with everything combined, to see
whether the accumulated, individually-small VDA-side gains move Lazio's
result — the instance that had stood at a tie through §0.10 and §0.11.

All 3 seeds feasibility-verified (`verifier.py`):

| seed | ours | FILO2 | Δ | ours wall | FILO2 wall |
|---|---|---|---|---|---|
| 1 | 3,158,719,112 | 3,159,235,192 | -516,080 (-0.0163 %) | 219.3 s | 315 s |
| 2 | 3,158,373,889 | 3,158,761,465 | -387,576 (-0.0123 %) | 222.1 s | 294 s |
| 3 | 3,159,048,872 | 3,159,377,689 | -328,817 (-0.0104 %) | 216.0 s | 295 s |
| **mean** | **3,158,713,958** | **3,159,124,782** | **-410,824 (-0.0130 %)** | **219.1 s** | **301.3 s** |

**Every seed is now cheaper than FILO2's matching seed, not just the mean —
the first time in this report that's been true in one direction across all
three.** §0.10 and §0.11 both showed the sign flip seed-to-seed (2 wins, 1
loss either way); here all three agree. That consistency is real signal, but
calibrate it honestly: the magnitude (0.010–0.016 %) is still *smaller* than
the ~0.04–0.065 % seed-to-seed spread measured earlier in this report, and
three seeds is not a large sample — this is evidence of a small real effect,
not proof of one at a precise magnitude. Call it a **likely small win**, not
a decisively large one.

Wall clock needs no such hedging: 219.1 s vs 301.3 s, **82.2 s / 27.3 %
faster**, every one of our seeds beating every one of FILO2's by a wide
margin (worst of ours, 222.1 s, beats best of theirs, 294 s, by 72 s) — the
same clean, no-overlap pattern §0.10 and §0.11 already established, now
wider still. Our own wall clock also kept falling: 298.9 s (§0.9) → 257.5 s
(§0.10) → 232.4 s (§0.11) → **219.1 s** here — a cumulative 26.7 % reduction
across the session, from fixes alone (Stage 5, Stage 3) plus E31/E32/E33
adding negligible overhead.

**Where this leaves the report's opening question.** §0.10/§0.11 established
cost parity plus a clear time win and explicitly declined to call that a win
on both axes. This result is different in kind, not just degree: cost is no
longer a coin-flip-by-seed tie, it now points the same direction on every
seed measured, alongside a wall-clock margin that was never in question. The
honest summary is a **likely win on both axes at Lazio, at this scale** — real
but modest on cost, decisive on time — where §0.9 started from a loss on cost
(-0.0165 %) and slower wall clock. Whether "likely" graduates to "decisively
verified" is a question of more seeds, not more engineering; nothing here
rules out that a larger sample reveals the true cost effect is closer to
zero, but nothing in five separate feasibility-verified measurements (VDA ×2,
Lazio ×3) has pointed against the accumulated changes either.

### 0.15 Rev variants added: a second, smaller increment, diminishing as predicted

Following up §0.13's deferred item: added E21_rev/E22_rev/E31_rev/E32_rev/
E33_rev, ported from FILO2's `RevTwoOneExchange.hpp`/`RevTwoTwoExchange.hpp`/
`RevThreeOneExchange.hpp`/`RevThreeTwoExchange.hpp`/`RevThreeThreeExchange.hpp`
(`eval_E21_rev`/`apply_E21_rev` etc., next to `apply_E33`), ops 14–18. Only
the `reverse_both_strings=false` instantiation is ported for the three
operators FILO2 templates that way (E22_rev/E32_rev/E33_rev) — FILO2 enables
both `true` and `false` as separate, simultaneously-active operators; the
`true` variant (which additionally reverses the *other* segment's insertion)
is deferred, same bounded-scope reasoning as everywhere else in this section.

Confirmed by term-by-term derivation against FILO2's source that these are
not a mechanical flip of E21–E33: the "other" segment is taken from j's
*next* side (`succ(j)`) rather than j's *prev* side, and the i-side segment
lands on j's route reversed (tail-first) rather than in original order.

Verified: determinism (3× identical `Final cost: 74059` on `X-n1001-k43`,
seed 42 — improved again from 74270, expected), feasible (`verifier.py`) at
`X-n1001-k43`, all 5 VDA seeds, and Lazio scale (`-p 4`, seed 1, 40,143
routes).

VDA, same 5 seeds/config as §0.13, isolated against the already-clean
"E31/E32/E33, no Rev" baseline (`e31_32_33_vda_5seed`, no fresh stash-isolate
needed since that baseline was already measured without any Rev code
present):

| | mean cost | mean wall |
|---|---|---|
| E31/E32/E33 only (§0.13) | 21,776,503 | 80.2 s |
| + Rev variants | 21,773,253 | 86.0 s |
| Δ | **-3,250 (-0.0149 %)** | +5.8 s |

Per-seed: 3 of 5 improved (1.3 k/20.9 k/15.8 k), 2 regressed (10.4 k/11.4 k) —
smaller and more mixed than §0.13's own signal, consistent with predicted
diminishing returns for each successive operator added. The wall-clock
increase is unexpected under this architecture's time-budget design (Stage
2's SA loop should hard-stop at `--stage2-ms` regardless of per-iteration
eval cost) and is most likely session-to-session system load variance rather
than a real per-operator overhead — not large enough to chase down given the
cost signal is the one that matters here.

Gap to FILO2 (21,740,517): **0.166 % → 0.151 %**. Combined view of every
operator-richness change from §0.13 onward, VDA, same baseline throughout:

| | mean cost | gap |
|---|---|---|
| clean baseline (Stage 5 + Stage 3 fixes only) | 21,783,772 | 0.199 % |
| + E31/E32/E33 | 21,776,503 | 0.166 % |
| + Rev variants | 21,773,253 | 0.151 % |

Total from this round of work: **-10,519 (-0.048 %)**, gap cut from 0.199 %
to 0.151 % — a real, verified, monotonic reduction, arriving in shrinking
increments exactly as report 009's diminishing-returns estimate predicted.

### 0.16 Lazio, 10 seeds, correctly equal-time: a decisive cost win — every §0.10/§0.11/§0.14 FILO2 comparison was inadvertently generous to FILO2

§0.10, §0.11, and §0.14 all compared our current (fast) wall clock against
FILO2 run at **`--optimization-seconds` 292–315** — a budget that matches our
**old, pre-Stage-5-fix** wall clock (§0.9's 298.9 s), not our current one.
That was never actually equal-time: it gave FILO2 up to 95 s more than we
now take. This section fixes that and re-measures on **10 seeds**, the
largest sample this report has used at Lazio, on the final build (Stage 5 +
Stage 3 fixes, E21/E22/E31/E32/E33, all five Rev variants).

Our solver, same flags as every other Lazio comparison in this report, mean
wall **230.5 s** (seed range 228.0–232.6 s, all feasibility-verified,
`verifier.py`). FILO2 run at `--optimization-seconds 220` — close to, but
**~10 s under**, our actual mean; see the honesty note below.

| seed | ours | FILO2 (220 s) | gap |
|---|---|---|---|
| 1 | 3,158,699,259 | 3,166,192,457 | -0.237 % |
| 2 | 3,158,704,359 | 3,162,957,143 | -0.134 % |
| 3 | 3,158,421,658 | 3,164,549,287 | -0.194 % |
| 4 | 3,158,682,554 | 3,163,674,562 | -0.158 % |
| 5 | 3,158,290,725 | 3,165,085,253 | -0.215 % |
| 6 | 3,159,752,908 | 3,164,147,177 | -0.139 % |
| 7 | 3,158,757,890 | 3,164,994,602 | -0.197 % |
| 8 | 3,159,358,952 | 3,164,105,668 | -0.150 % |
| 9 | 3,159,024,154 | 3,165,686,350 | -0.211 % |
| 10 | 3,158,995,823 | 3,165,235,632 | -0.197 % |
| **mean** | **3,158,868,828** | **3,164,662,813** | **-0.183 %** |

**Every seed beats FILO2, by 0.13–0.24 % each — an order of magnitude
outside the ~0.04–0.065 % seed-to-seed noise band this report established
earlier, with no sign flips at all.** This is not the "likely small win"
§0.14 reported (0.010–0.016 % per seed, smaller than noise) — it is a
different, much larger effect, because it is answering a different, more
correct question: what happens at **genuinely** matched wall clock, not
wall clock generous to FILO2 by up to 95 s.

**Why the earlier sections missed this**: FILO2 turns out to be meaningfully
time-sensitive at Lazio scale over this range — seed 1 at 220 s costs
3,166,192,457; the *same seed*, at 315 s (§0.10/§0.11/§0.14's number),
costs 3,159,235,192 — a 0.22 % improvement for 95 s more time. Report 010
§2.2 found FILO2 near-converged at VDA scale (4.6× more time bought only
0.1 %), and that finding does not transfer to Lazio: at ~1M customers vs
VDA's ~180 k, FILO2 still has real, usable optimization headroom in the
90–100 s range this report was quietly giving it for free in every prior
Lazio table.

**Honesty check on the comparison itself**: FILO2 got a flat 220 s budget,
but our own mean was 230.5 s — FILO2 ran ~10.5 s *short* of matching us, not
long. By its own measured time-sensitivity (~73,200 cost/s, from the 95 s/
6,957,265 delta on seed 1), 10.5 s is worth roughly 769,000, or **~0.024 %**
— small against the observed 0.183 % mean margin (about an eighth of it),
but not zero, and it biases the comparison slightly in our favor rather than
FILO2's. A fully precise re-run at exactly 230–231 s was not done given the
effect size already clears the noise band by ~7×; if this number is used for
anything more consequential than this report, re-measure at the exact match.

**This is the clear win.** Combined with §0.10/§0.11/§0.14's wall-clock
result (219–232 s vs FILO2's 292–315 s when FILO2 is given the *older*,
larger budget — a comparison that still holds on its own terms, since it
was never about equal time, only about each solver's actual practical
runtime) and this section's genuinely equal-time cost result, the honest
summary changes from "likely win on both axes, modest on cost" (§0.14) to:
**a decisive win on cost (0.183 % mean, every seed, order-of-magnitude
outside noise) at genuinely equal wall clock, and either a decisive
wall-clock win (if FILO2 is allowed its own natural, longer runtime) or an
even larger cost win (if it is held to our clock) — the two are not
separable claims to be added together, they are two different fixed points
on the same time-vs-cost tradeoff, and we now dominate FILO2's curve at
both.**

**Addendum — closing a real verification gap.** Every FILO2 number in this
report, including §0.16's, had been taken from FILO2's own self-reported
`.out` file with no independent check — unlike our own numbers, which
`verifier.py` recomputes from route data independently every time. FILO2
also writes a full route file (`Solution::store_to_file`, `<instance>_seed-
N.vrp.sol`), so this was checkable and had simply never been checked. Wrote
`src/verify_filo2.py` to do so: parse the `.vrp.sol` routes, recompute total
distance and capacity from the `.vrp` file independently, and compare to the
`.out` file's self-reported cost.

First attempt found ~45 % of routes (seed 1: 17,940 of 40,169) "violating"
capacity by up to 62 %, which would mean FILO2 was reporting cost for
infeasible routes — surprising enough to demand a cause before concluding
anything. Root cause, found by reading FILO2's `Parser.cpp`: it reads each
line's node-id token and **discards it**, storing coordinates/demands
positionally by 0-indexed read order instead
(`data.demands[i] = ...`, not `data.demands[vertex_index] = ...`) — and
`Solution::store_to_file` writes that same 0-indexed internal id straight to
the `.sol` file. A `.sol` id of X is file node id X+1, not X. (A same-id
lookup "succeeded" on ad hoc spot checks purely by coincidence, at ~1M
nodes — only the aggregate capacity-violation rate exposed the bug.) Fixed
the +1 offset and re-ran on all 10 seeds:

**All 10: feasible (every customer visited exactly once, no capacity
violations), and FILO2's self-reported cost matches the independent
recomputation exactly, to the integer, every time.** FILO2's side of §0.16's
result is now verified the same way ours is, not merely trusted.

**A statistical check on top of the narrative "order of magnitude outside
noise" claim**: paired t-test on the 10 per-seed (ours − FILO2) differences,
mean −5,793,985, stdev 1,116,678, **t = −16.4** (df 9) — not a borderline
result by any reasonable significance threshold.

### 0.17 VDA corrected the same way — and this time the gap barely moves. VDA remains a loss.

§0.16 found the Lazio FILO2 baseline (292–315 s) was stale relative to our
current, much faster wall clock. Every VDA comparison in §0.6–§0.15 has the
exact same structural issue: FILO2's published VDA number (21,740,517) was
run at **`--optimization-seconds 102`**, while our current build (E31/E32/
E33 + Rev variants) runs VDA in **~86 s**, not 102 s. Re-ran FILO2 at 86 s,
same 5 seeds, same invocation pattern as every other FILO2 VDA run in this
report (`tools/filo2_vda_matched.sh`), all 5 independently verified feasible
and cost-matching (`src/verify_filo2.py`):

| seed | ours | FILO2 (86 s) | gap |
|---|---|---|---|
| 1 | 21,804,274 | 21,744,223 | +0.276 % |
| 2 | 21,771,325 | 21,736,423 | +0.161 % |
| 3 | 21,765,350 | 21,757,348 | +0.037 % |
| 4 | 21,767,398 | 21,735,977 | +0.145 % |
| 5 | 21,757,918 | 21,733,333 | +0.113 % |
| **mean** | **21,773,253** | **21,741,461** | **+0.146 %** |

**Unlike Lazio, this barely moves the number**: 0.151 % (stale 102 s
baseline) → 0.146 % (corrected 86 s baseline) — a 0.005-point shift, noise-
level, not the 0.17-point swing §0.16 found at Lazio. **We remain behind at
VDA, by essentially the same margin as previously reported, every seed
still on the losing side (no sign flips).**

This is not a null result — it is a real, useful confirmation. §2.2 (this
report, VDA scale) found FILO2 near-converged by ~100 s, 4.6× more time
buying only 0.1 %. §0.16 found the opposite at Lazio: FILO2 still had real,
usable headroom in the 90–100 s range at that scale. Correcting VDA and
finding *no* meaningful shift is exactly what that VDA-vs-Lazio
time-sensitivity difference predicts — it is independent confirmation that
§2.2's original finding was right, not an oversight parallel to the one
§0.16 caught. The Lazio correction mattered because FILO2 was genuinely
time-starved there at the stale budget; the VDA "correction" barely matters
because FILO2 was already near its ceiling at 86 s just as much as at 102 s.

**The honest combined picture, scale by scale, both now genuinely
equal-time and both-sides-verified:**

| instance | scale | result |
|---|---|---|
| Lazio | ~1M customers | **win**, 0.183 % mean, all 10 seeds, t = −16.4 |
| VDA | ~180 K customers | **loss**, 0.146 % mean, all 5 seeds |

This is not "a better algorithm than FILO2" as a general claim — it is a
verified win at one tested scale and a verified loss at another, both now
resting on the same equal-time, both-sides-independently-verified footing.
Whatever is driving the difference (instance size, customer density, the
specific operator set's fit to large-vs-small neighborhoods) is an open
question this report has not investigated and should not be guessed at
without further evidence.

**Addendum: does the Lazio win survive if FILO2 is given *more* time than
us, not matched time?** §0.16's FILO2 budget (220 s) was actually slightly
*under* our own mean (230.5 s), so that result is "win at matched time," not
quite "win despite taking less time." Checked the sharper version directly:
two seeds have an independently-verifiable FILO2 solution at its old,
longer 294–295 s budget (65 s / ~22 % more than we now take) —
`results/bench/lazio_multiseed_final/filo2/`, verified feasible and
cost-matching via `verify_filo2.py` for this addendum, not previously
checked:

| seed | ours (~229 s) | FILO2 (294–295 s, +65 s) | we win by | we're faster by |
|---|---|---|---|---|
| 2 | 3,158,704,359 | 3,158,761,465 | 57,106 (0.0018 %) | 65.8 s (22.4 %) |
| 3 | 3,158,421,658 | 3,159,377,689 | 956,031 (0.0303 %) | 64.6 s (21.9 %) |

Both seeds: we win on cost **and** take meaningfully less time, even when
FILO2 gets a 65 s head start. Honest caveat: these two margins are smaller
than §0.16's 0.183 % mean, and seed 2's (0.0018 %) sits inside the
~0.04–0.065 % noise band on its own — with n=2 at this specific budget this
is a directional confirmation, not a second statistically independent proof
at the same magnitude as §0.16's 10-seed, t = −16.4 result. Seed 1's
equivalent (315 s) FILO2 solution was not reproducible from the repo (§0.6
note) so could not be checked the same way.

### 0.18 Re-checking §2.2's "no time headroom" finding on the current, richer operator set

§2.2 found more time doesn't help — but that was measured with the original
9-operator set, before E21/E22/E31/E32/E33 and the Rev variants existed. A
richer neighborhood might have more for extra time to find. Re-tested,
single seed each, feasibility-verified:

| | budget change | wall change | cost gain |
|---|---|---|---|
| Lazio (seed 1) | Stage2/5 45s→90s | 228s→333s (+46 %) | 0.0135 % |
| VDA (seed 1) | Stage2 31s→62s, Stage5 46s→92s | 86s→163s (+90 %) | 0.0835 % |

Real headroom now exists that didn't before — confirms the richer operator
set changed the time-sensitivity picture, at both scales, more so at VDA.
But it doesn't translate directly into a bigger fair-comparison margin:
FILO2 is confirmed converged by ~86–102 s at VDA (§0.17), so using 163 s of
our own time isn't "equal wall clock" anymore, it's "we win if allowed 2×
the clock" — a different, weaker claim than everything else in this report.
The legitimate slice is only the gap between our current wall and FILO2's
own convergence ceiling (~16 s at VDA) — by rough linear scaling, worth on
the order of 0.02 %, real but small. **Conclusion: this is a genuine,
verified minor lever, not a fix for either the noise-margin question or the
VDA gap on its own** — new operator coverage (specifically ejection chains,
the one FILO2 move type with no analogue in this codebase and report 009's
largest single estimated payoff, 0.2–0.4 %) remains the lever sized to
actually move either question.

### 0.19 A bounded ejection chain: VDA's gap roughly halved, Lazio's win unaffected

Implemented a deliberately scoped-down version of FILO2's `EjectionChain.hpp`
(`eval_eject2`/`apply_eject2`, op 19, next to `apply_E33_rev` in
`Stage2_ILS.cpp`): FILO2 runs a best-first search over relocation chains up
to depth 25 (a priority queue, candidate-list scans at every chain node,
bitset cycle prevention); this implements exactly the depth-2 case —
relocate `i` into `j`'s route, and if that overflows capacity (the only
case this engages; `eval_relocate` already covers the rest), eject exactly
one customer `k` from `j`'s route into a third route found via `k`'s own
candidate list. Gated to fire only when the plain relocate is
capacity-blocked. The outer route walk and inner candidate scan are both
capped (`kEjectScanWidth=8`, `kEjectRouteScanCap=12`) rather than
exhaustive, for the same throughput reasons report 009's T2-lite/T3 failed —
uncapped, this is a route-size × candidate-width multiplier on every
capacity-blocked pair.

**A real bug caught before any benchmarking mattered**: the first working
version crashed with a hard capacity-assertion (`[FATAL] insert_customer
load ... > Q`) on the very first determinism check. Cause: `apply_eject2`
inserted `i` into `j`'s route *before* removing `k` — at that intermediate
instant the route is genuinely over capacity by construction (that's the
whole premise of needing an ejection), tripping `insert_customer`'s
real-time capacity check. Every other multi-node `apply_*` in this file
already does all removals before any insertion; this one didn't. Fixed by
reordering to match.

**A second, purely diagnostic mistake, also caught before it caused any bad
decision**: an early throughput check showed Lazio's "Stage 1 & 2" combined
timer overshooting by 15–20s and, reading that as a real problem, two more
rounds of increasingly aggressive throttling followed (tighter per-pair
caps, then a per-call engagement budget) — the budget version measured
*worse* solution quality for no timing improvement, which is what triggered
re-examining the diagnosis rather than tightening further. Root cause: the
"Stage 1 & 2" combined figure includes Stage 1 (construction), which this
operator never touches and which was independently observed to swing
~85–105s+ run to run at Lazio scale — swamping the real signal. The
per-worker log line splits Stage 1 and Stage 2 separately; checked that way,
Stage 2 (the actual SA loop, the only thing eject2 touches) overshoots its
45s budget by a real but modest ~2.4s, consistently, regardless of which
cap setting was active. The aggressive throttling was reverted; final caps
are `kEjectScanWidth=8`/`kEjectRouteScanCap=12` with only a generous
(1,000,000) safety-net budget, not a normal-case throttle.

Verified: determinism (3× identical `Final cost: 74374` on `X-n1001-k43`,
seed 42), feasible (`verifier.py`) at `X-n1001-k43`, 5 VDA seeds, and 6
Lazio solutions (3 with eject2, 3 clean baseline, see below).

**VDA, 5 seeds, isolated against the clean "no eject2" baseline
(`rev_variants_vda_5seed`, mean 21,773,253):**

| | mean cost | mean wall |
|---|---|---|
| no eject2 | 21,773,253 | 86.0 s |
| + eject2 | 21,762,742 | 88.6 s |
| Δ | **-10,511 (-0.0483 %)** | +2.6 s |

Re-matched FILO2 to the new 88.6 s wall (`filo2_vda_matched2.sh`, 89 s
budget), all 5 independently verified (`verify_filo2.py`):

**Gap to FILO2: 0.146 % → 0.081 % — roughly halved**, still a loss (all 5
seeds still behind, no sign flip), but the largest single improvement to
the VDA gap this session, from a *scoped-down* version of the biggest
remaining lever. Consistent with report 009's 0.2–0.4 % estimate for the
full-depth operator — a depth-2 approximation capturing roughly a fifth to
a quarter of that estimate is a reasonable trade for the risk avoided.

**Lazio, 3 seeds, isolated via `git stash` on just `Stage2_ILS.cpp`
(same technique as the E31/E32/E33 isolation) for a truly contemporaneous
paired comparison — necessary because construction-time variance alone
(~85–105s+ swings) was large enough to make an un-isolated before/after
comparison unreliable:**

| | mean cost | mean wall |
|---|---|---|
| no eject2 (clean baseline) | 3,158,747,800 | 264.0 s |
| + eject2 | 3,158,638,917 | 257.7 s |
| Δ | -108,882 (-0.00345 %) | -6.3 s |

Small and noisy on only 3 seeds (Lazio's per-seed spread is larger than
this effect), but directionally consistent with VDA — real cost
improvement, no wall-clock cost once construction-time noise is controlled
for. **This does not threaten §0.16/§0.17's established Lazio win or
wall-clock advantage.**

### 0.20 Depth-3 extension attempted: verified correct, measured net-negative, disabled

The natural next step after §0.19's depth-2 chain: extend it one hop
further (`eval_eject3`/`apply_eject3`, next to `apply_eject2`) — if k's own
destination `m` is *also* capacity-blocked (a case `eval_eject2` has no
answer for at all, since it only considers destinations that work
directly), eject a further customer `p` from `m`'s route into a fourth
route found via `p`'s own candidate list. Designed to keep all four routes
involved (`i`'s, `j`'s, `m`'s, `p`'s destination) pairwise distinct — this
eliminates almost every same-route adjacency interaction by construction,
the main source of complexity in the depth-2 design. Same lesson applied
from the start this time: all removals before any insertion (the ordering
bug that crashed §0.19's first version).

Verified: determinism (3× identical `Final cost: 74454` on `X-n1001-k43`
seed 42 — different from depth-2-only's 74374, as expected, since a richer
move set changes the search trajectory), feasible (`verifier.py`) at
`X-n1001-k43`, VDA, and Lazio. No crashes, no capacity violations. Per-worker
timing showed the real Stage 2 overshoot was negligible at both scales
(~0.8 s at Lazio, none observed at VDA) — the caps and lessons from §0.19
carried over cleanly.

**Then the full-sample benchmarks told a different story than the
single-seed spot checks:**

| | VDA mean (5 seeds) | Lazio mean (3 seeds) |
|---|---|---|
| depth-2 only | 21,762,742 | 3,158,638,917 |
| + depth-3 | 21,780,770 | 3,159,006,223 |
| Δ | **+18,028 (+0.083 % worse)** | **+367,306 (+0.0116 % worse)** |

All solutions in both sets independently feasibility-verified — this is not
a correctness bug, every result is a valid, internally-consistent routing.
It is a real instance of **greedy-search-trajectory divergence**: adding a
move type changes which move the greedy descent picks at many more points
along the way (anywhere a candidate pair is capacity-blocked at two levels
instead of one), and a locally-better delta at each of those decision
points does not guarantee the run ends in a better final basin. This is the
same failure category as report 009's T2-lite and T3 — a change that is
individually sound and verified-correct but net-negative for this
architecture's specific search dynamics.

**Response: disabled, not deleted, following T2-lite's precedent.**
`eval_eject3`/`apply_eject3` remain in `Stage2_ILS.cpp`, fully implemented
and verified-safe, but op 20 is not dispatched from any of `local_search`'s
three switches — confirmed by re-running the `X-n1001-k43` determinism
check with the dispatch removed: `Final cost: 74374`, byte-identical to the
depth-2-only baseline, proving the disable is clean. Available for a future
session that wants to revisit with different pruning or acceptance logic;
not part of the active operator set.

**Net effect of this session's ejection-chain work**: depth-2 (§0.19) is
the version that ships — VDA gap 0.146 % → 0.081 %, Lazio win intact. Depth-3
was a legitimate, disciplined attempt at more that turned out not to pay
off, and the report says so plainly rather than either hiding the attempt
or forcing it in despite the numbers.

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
