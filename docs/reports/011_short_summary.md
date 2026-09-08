# CVRP Parallel Solver — Short Summary

*Architecture and how the whole system works: `docs/reports/012_architecture_overview.md`.
Full chronological detail behind the numbers below: `010_can_this_architecture_beat_filo2.md`.
Raw numbers: `results_summary.xlsx`. All results below are independently verified —
feasibility and cost are recomputed from raw route data, not taken from either solver's
self-report (`src/verifier.py`, `src/verify_filo2.py`).*

## What this is

A parallel metaheuristic solver for the Capacitated Vehicle Routing Problem, built around
Hilbert-curve spatial partitioning → independent parallel per-chunk search → graph-colored
boundary healing → serial polish. Benchmarked against FILO2 (Accorsi & Vigo), the published
state-of-the-art solver these benchmark instances (Valle-D'Aosta, Lazio, Lombardia — real
Italian regions, ~180K to ~1M customers) were originally built to test.

## Current standing, at equal wall clock

| Instance | Scale | Result |
|---|---|---|
| **Lazio** | ~1,000,000 | **Win, both axes.** 0.183% cheaper (10 seeds, t = −16.4), 27% faster, zero overlap between the two solvers' cost distributions. |
| **Valle-D'Aosta** | ~180,000 | **Statistical tie.** +0.023% at n=15, \|t\| ≈ 1.0 — real progress from a verified 0.146% loss earlier this work, but not a win. |
| **Lombardia** | ~950,000 | **Loss, cause diagnosed.** 0.106%, narrowed to 0.088% under budget; a harder route-minimization pass closes the gap outright but currently needs ~3× the time budget. Specific, well-understood engineering target, not a mystery. |

Scale alone does not predict the outcome — Lazio and Lombardia are the same order of size —
which is itself one of the more interesting findings here, and not something this codebase
set out looking for.

## What changed, and what it did

1. **Two time-budget bugs, fixed.** Stage 5 and Stage 3 were each silently running roughly
   2–3× their requested time budget (a stale clock reference in one case, a per-color-class
   budget applied as a per-stage budget in the other). Fixing both reclaimed ~69s of wall
   clock at Lazio with no cost change.
2. **A benchmarking bug, caught and fixed.** FILO2's time budget in the comparison scripts had
   been left at our *old*, pre-fix wall clock — giving it up to 95s more than we actually took.
   Correcting it (and building an independent checker so neither solver's self-report is
   trusted blindly) is what turned "roughly tied" into the verified 0.183% Lazio win above.
3. **10 new local-search operators**, ported from FILO2's published move definitions
   (segment-exchange family) and adapted to this codebase.
4. **A scoped ejection-chain operator** — FILO2's largest move type, deliberately implemented
   at depth 2 rather than FILO2's depth 25, to keep the added per-iteration cost bounded. Cut
   VDA's gap roughly in half (0.146% → 0.081%).
5. **A depth-3 extension, tried and rejected.** Implemented, verified correct, but measured
   *worse* on both instances in full multi-seed testing. Disabled rather than shipped, with the
   code kept and the negative result documented — not every idea that looks sound on paper
   survives contact with the actual search dynamics.
6. **A parameter-tuning pass at VDA** (chiefly a harder route-minimization setting) took VDA
   from a loss to the current tie.
7. **The Lombardia gap, profiled and diagnosed** to a specific function (a route-minimization
   precompute that scales with route length, ~3× more expensive at Lombardia's vehicle
   capacity than at the other two instances) and partially, safely fixed.

## One methodological note worth keeping

A 5-seed comparison at VDA looked like a win (0.026% ahead). Extending to 15 seeds reversed
the sign. Every headline number above uses a sample size chosen to actually resolve the effect
being claimed, specifically because of that reversal — it's recorded in the workbook
(`VDA (15-seed tie)` sheet) as the reason, not swept past.

## Honest limitation

The local-search operators and route-minimization heuristic are close ports of FILO2's
published techniques; the genuinely original parts are the parallel partition/heal
architecture, the specific scoping choices in the ejection chain, and the empirical
scale-dependent characterization above. As a piece of engineering and a rigorous empirical
comparison, this stands on its own. As a research contribution, it is not yet clearly novel
enough for publication on its current content — the open, unexplored direction (cooperation
between partitions *during* search rather than only healing once afterward) is where that
would need to come from, and is flagged as future work rather than attempted here under time
constraints.
