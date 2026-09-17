# CVRP Parallel Solver — Progress Summary

**Author:** Janay Bhanushali
**Code:** [github.com/Janay05/cvrp-research](https://github.com/Janay05/cvrp-research)
**Results spreadsheet:** [Google Sheets](https://docs.google.com/spreadsheets/d/1J6V6owf1Z1RzY0IAiT783KUZXU78ogtz679G_Lt5f3E/edit?usp=sharing)

*This report is meant to stand on its own — no familiarity with the code is assumed.
Every result below was independently re-checked from the raw output (re-counting every stop, re-adding every distance from scratch), not
taken from either solver's own printed summary — so these numbers aren't self-reported,
they're independently audited.*

## The problem, in one paragraph

The Capacitated Vehicle Routing Problem (CVRP) is: given a depot and a set of customers
each needing a delivery, and a fleet of trucks each with a maximum carrying capacity,
find the set of delivery routes that visits every customer exactly once, respects every
truck's capacity, and minimizes total distance driven. Real-world solvers (including this
one) use heuristic search: build a decent starting set of routes, then repeatedly try
small modifications (move a customer, swap two customers, reverse a stretch of a route,
etc.) and keep the ones that help. This project built such a solver, aimed specifically
at very large instances — up to a million customers — where the standard approach of one
search improving one shared set of routes becomes slow simply because there's so much to
search through.

## The architecture: split the map, search in parallel, then heal

The core design decision this project is built around: split the customers into
geographic regions ("chunks") and run the search on each region *independently and
simultaneously*, using multiple processor threads at once, instead of one search crawling
over the whole map. This is what makes the solver capable of handling a million customers
in minutes rather than hours — each thread only ever has to think about its own region.

That decision has a direct, unavoidable cost: a search that can only see its own region
cannot consider moving a customer to a neighboring region, so the routes right at the
border between two regions come out worse than they would in an unsplit search. The
architecture accounts for this with two follow-up phases: neighboring regions are healed
together in parallel pairs, and then a final short pass looks at the whole map at once to
smooth out anything left over. The central engineering question this project set out to
answer is whether that trade — losing cross-region visibility during the main search, in
exchange for massive parallelism — nets out ahead of a strong non-parallel solver once the
healing/polish phases are accounted for.

The benchmark for "is this any good" is [FILO2](https://arxiv.org/abs/2306.14205)
(Accorsi & Vigo), a published, state-of-the-art single-threaded CVRP solver. The
comparison instances — Valle-D'Aosta, Lazio, Lombardia (real Italian regions, 180,000 to
about 1 million customers each) — are FILO2's own published benchmark set, chosen
deliberately so every result below is a direct comparison against a specific, credible,
already-published baseline, on the exact instances it was designed and tuned for.

## What came before this, and how it led here

Before settling on the architecture above, an earlier approach was tried and set aside:
partition the instance the same way (geographically), but instead of running a custom search
on each region, dispatch each one as an independent sub-problem to
[HGS-CVRP](https://github.com/vidalt/HGS-CVRP) (Vidal et al.'s Hybrid Genetic Search solver) —
essentially, use an existing strong solver as the per-region engine rather than writing one.
That work is kept in its own repository,
[Partitioned-Hgs](https://github.com/Janay05/Partitioned-Hgs), rather than mixed into this one.

The motivation was concrete: run directly (unpartitioned) on the same large real-world
instances used here, HGS-CVRP ran out of memory outright on every instance above roughly
100,000 customers — partitioning wasn't a nice-to-have there, it was the only way to get an
answer at all at that scale. Three different geographic partitioning strategies were built and
compared (a Hilbert space-filling curve, minimum-spanning-tree-based clustering, and a
concentric angular sweep from the depot), and dispatching each resulting sub-problem to
HGS-CVRP did successfully produce a routed solution for every sub-problem.

**What that approach never finished is exactly what this project's Stage 3 is.** Its codebase
includes a designed-but-never-implemented `BoundaryOptimizer` component, explicitly intended to
"resolve stragglers across boundary lines" — the same chunk-boundary quality loss described
above — but it was never wired into the pipeline, so partitioning's boundary cost was never
actually addressed there. This project's parallel boundary-healing phase is, in effect, a
completed and working version of that same unfinished idea, built from scratch rather than
picking up that code directly — along with the separate decision to replace "dispatch to an
external solver process per region" with an in-process custom search, which is what makes the
tight, per-region time-budget control this project relies on possible in the first place.

## Result: the trade-off pays off, but not uniformly, and the reason why is the interesting part

Both solvers were given the same amount of real time to work with, then compared on the
total distance of the routes they produced (lower is better).

| Instance | Scale | Result |
|---|---|---|
| **Lazio** | ~1,000,000 customers | **We win, on both cost and speed.** Routes 0.183% cheaper on average across 10 independent runs, and 27% faster. The two solvers' results don't even overlap — every one of our 10 runs beat every one of FILO2's. |
| **Valle-D'Aosta** | ~180,000 customers | **Statistical tie.** Our cost is 0.023% higher on average across 15 runs — small enough to be indistinguishable from run-to-run noise, not a real, repeatable difference either way. |
| **Lombardia** | ~950,000 customers | **We lose, by a known and explained amount.** Routes cost 0.106% more, for a specific, identified reason (below) rather than an unexplained shortfall. |

**The result that matters most here isn't any single row — it's that scale alone doesn't
predict the outcome.** Lazio and Lombardia are almost the same size, yet one is a clear win
and the other a clear loss. The actual deciding factor, found by direct profiling rather
than assumed, is each instance's *truck capacity*: Lombardia's trucks carry roughly three
times as much per trip as Lazio's or Valle-D'Aosta's. One specific pre-processing step —
a route-count-reduction pass that runs once per region before the main search starts, and
whose cost scales with how long each route it examines is — ends up roughly 3x more
expensive per attempt on Lombardia simply because its routes are three times longer. That
extra cost eats into the time budget available for the rest of the search on that
instance, and is the direct, measured cause of the Lombardia loss. This was partially and
safely fixed (limiting how much of that step's cost scales with route length, without
hurting the other two instances), narrowing the gap from 0.106% to 0.088%; fully closing
it is a known, scoped next step (give that specific step roughly 3x more time on
high-capacity instances) rather than an open question.

This is a genuinely useful finding about the architecture: the parallel-chunking approach
isn't uniformly good or bad at "large scale" — its performance depends on a specific,
identifiable property of the instance (truck capacity relative to route length), which is
a sharper and more useful characterization than "works at scale" or "doesn't."

## Key design decisions inside that architecture, and what they did

1. **How much search power to give each region, and where the ceiling is.** FILO2's most
   powerful move type works by "ejecting" a customer from its route and finding it a new
   home elsewhere, repeating that chain of displacement — FILO2 allows chains up to 25
   steps long in its single unified search. Within this project's parallel-chunk
   architecture, where the same move gets evaluated independently inside every region on
   every thread, a chain that long would be far too expensive to run per-region without
   starving the rest of the search. The design choice made here was a chain limited to 2
   steps — enough to meaningfully improve results (it roughly halved the cost gap on
   Valle-D'Aosta, from 0.146% to 0.081%) while keeping the added per-attempt cost bounded
   enough to still be worth paying inside every region, on every thread, every iteration.

2. **Testing whether more search depth keeps paying off — and finding that it doesn't.**
   The natural next question was whether extending that chain to 3 steps would help
   further. It was implemented and verified to produce valid routes, but multi-seed
   testing showed it made results slightly *worse* on both test instances, not better.
   This is a real architectural finding, not a wasted detour: inside a parallel-chunk
   search, where every region is running its own independent, comparatively short-lived
   search, spending more time per iteration on a deeper (and rarer) move doesn't
   necessarily pay for itself the way it might in a single long-running unified search
   like FILO2's — there's a point past which more per-move sophistication trades away more
   search breadth than it buys back. The 3-step version was disabled rather than shipped,
   and the negative result documented.

3. **Broadening what each region's search can try**, by porting roughly a dozen additional
   move types from techniques published in FILO2's paper (adapted to this project's data
   structures) and tuning the route-count-reduction step run before each region's main
   search begins — this is the change that moved Valle-D'Aosta from an outright loss to
   the current statistical tie.

## Measurement integrity (secondary to the above, but load-bearing for trusting it)

Two things had to be corrected before the results above could be trusted as real:

- **Two timing bugs**, where two of the solver's phases were silently running 2-3x longer
  than their requested time budget due to how elapsed time was being measured — fixed,
  recovering time that should have gone to search instead.
- **A benchmarking-script bug**, where FILO2 was accidentally still being given an old,
  more generous time allowance than our own solver was actually taking, left over from
  before the fix above. Correcting it (and building an independent checker so neither
  solver's self-reported result is trusted at face value) is what turned a "roughly tied"
  reading on Lazio into the verified win reported above.

**A related methodological finding worth flagging on its own:** an earlier check on
Valle-D'Aosta, run across only 5 independent seeds, looked like a win (0.026% ahead).
Running 15 seeds instead reversed the direction of the result entirely. With randomized
search algorithms, a handful of runs can easily produce a "result" that flips sign with
more data — every headline number in this report uses a sample size chosen to actually
tell a real effect apart from noise, and the reversal itself is recorded in the results
workbook as the reason, rather than being quietly dropped once a more favorable number was
in hand.

## Honest assessment: is this new research, or an engineering exercise?

The individual rearrangement moves and the route-reduction step used here are close
adaptations of techniques FILO2's paper already published — that part is not new. What is
original to this project is the overall architecture of splitting the problem
geographically and solving the pieces at the same time on separate processor threads, the
specific design trade-offs made in adapting FILO2's techniques to work well within that
split-and-heal structure (item 1 and 2 above), and the empirical finding that the
approach's success depends on truck capacity relative to route length rather than on
instance size.

As a piece of engineering, and as a careful, honestly-reported empirical comparison against
a strong published baseline, this stands on its own. As a contribution novel enough to
publish as research, it is not there yet in its current form — the one unexplored
direction that could get it there is having the different regions' searches actually
cooperate with each other *while* they're running, rather than only reconciling their
differences afterward in the healing step. That idea is flagged here as a concrete next
step, not something this round of work had time to attempt.
