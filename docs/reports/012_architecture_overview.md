# Architecture Overview — A Parallel Solver for the Capacitated Vehicle Routing Problem

*This is a standalone description of the whole system: what problem it solves, how it's
structured end to end, and why each major piece exists. For the chronological log of
investigations that produced this system, see `010_can_this_architecture_beat_filo2.md` and
the numbered reports before it. For a short changes-and-effect summary, see
`011_short_summary.md`.*

## Contents

1. [Problem and goal](#1-problem-and-goal)
2. [Pipeline at a glance](#2-pipeline-at-a-glance)
3. [Core data structures](#3-core-data-structures)
4. [Stage 0 — partitioning and candidate lists](#4-stage-0--partitioning-and-candidate-lists)
5. [Stage 1 — construction](#5-stage-1--construction)
6. [Route minimization (T3)](#6-route-minimization-t3)
7. [Stage 2 — parallel iterated local search](#7-stage-2--parallel-iterated-local-search)
8. [The local-search operator catalog](#8-the-local-search-operator-catalog)
9. [Stage 3 — boundary healing](#9-stage-3--boundary-healing)
10. [Stage 4 — route cleanup](#10-stage-4--route-cleanup)
11. [Stage 5 — serial polish](#11-stage-5--serial-polish)
12. [Time-budget scheduling](#12-time-budget-scheduling)
13. [Verification and benchmarking methodology](#13-verification-and-benchmarking-methodology)
14. [Current standing](#14-current-standing)
15. [What's original here, and what isn't](#15-whats-original-here-and-what-isnt)
16. [Known open items](#16-known-open-items)

---

## 1. Problem and goal

The Capacitated Vehicle Routing Problem (CVRP): given a depot, a set of customers each with a
demand, a distance metric, and a fleet of identical vehicles with capacity `Q`, find a set of
routes — each starting and ending at the depot, each not exceeding `Q` — that visits every
customer exactly once at minimum total distance. It's NP-hard; real instances range from a few
hundred customers to, in this project's benchmark set, about one million.

The goal of this project was to build a solver competitive with **FILO2** (Accorsi & Vigo), a
published, single-threaded, state-of-the-art heuristic specifically designed for very large
CVRP instances, and to do so by exploiting multi-core parallelism — a resource FILO2's design
does not use. The benchmark instances (Valle-D'Aosta ~180K customers, Lazio ~1M, Lombardia
~950K) were deliberately chosen to be FILO2's own published benchmark suite ("Routing One
Million Customers in a Handful of Minutes," Accorsi & Vigo 2023) specifically so that every
comparison in this project is against numbers FILO2's own paper already reports, on the exact
instances it was designed and tuned for — the standard, correct way to run a fair, direct
comparison against a specific baseline, rather than an incidental fact about the project.

## 2. Pipeline at a glance

```
Stage 0   Partition customers into P chunks (Hilbert curve) + build candidate lists
              |
Stage 1   Per chunk, in parallel: construct an initial solution (MST or Clarke-Wright)
              |
  T3       Per chunk: ROUTEMIN — a destroy/repair pass aimed at reducing route count
              |
Stage 2   Per chunk, in parallel: Iterated Local Search (ruin + recreate + local search)
              |
          [ all P chunks join here — this is the only synchronization point before Stage 3 ]
              |
Stage 3   Heal chunk-boundary damage: pairs of adjacent chunks run a shared local search
          over just their shared boundary customers, scheduled by graph coloring so pairs
          that don't share a chunk run concurrently
              |
Stage 4   Cheap cleanup: dissolve near-empty routes into other routes where it helps
              |
Stage 5   Serial polish: one more full local-search + simulated-annealing pass over the
          whole (now merged) solution
```

Every stage after Stage 0 shares the same local-search machinery (Section 8) — Stage 1's
ROUTEMIN, Stage 2's main loop, Stage 3's boundary healing, and Stage 5's polish are all built
on the same `local_search` function, called with different inputs (candidate-list width,
which customers are eligible, time budget). This is a deliberate design choice: one
well-verified local-search core, reused everywhere, rather than five separate ones.

## 3. Core data structures

**`Solution`** (`Solution.hpp`) represents a set of routes as a doubly-linked list per route,
flattened into shared arrays indexed by customer ID — not an array of route objects:

- `pred[v]`, `succ[v]` — the customer immediately before/after `v` in its route (depot = `0`)
- `routeOf[v]` — which route index `v` belongs to
- `routeHead[r]`, `routeTail[r]`, `routeLoad[r]` — per-route bookkeeping
- `routePosition[v]`, `cumLoad[v]` — `v`'s position and cumulative load within its route (used
  for O(1) capacity checks in inter-route moves)
- `costToPred[v]` — the cached distance `dist(pred[v], v)`, maintained incrementally so most
  operators never need to call the distance function to look up an edge they already know
- `totalCost` — maintained incrementally, not recomputed from scratch after each move

This layout makes route mutation (remove/insert a customer) a handful of pointer updates, not
an array shuffle, and makes most local-search delta calculations reference-lookups rather than
route walks.

**`ThreadArena`** (`ThreadArena.hpp`) is per-thread scratch space, allocated once and reused —
not per-move — covering:

- **A do/undo log**: every `remove_customer`/`insert_customer` call appends an entry recording
  enough to reverse it. A rejected simulated-annealing move is undone by replaying this log
  backward (O(moves), not a full solution copy), and an *accepted* move's log is simply
  discarded rather than rolled back.
- **SWAP\* precompute tables** (`Top3Insertions`) — the top-3 cheapest insertion points for a
  customer into a given route, computed once per route per local-search pass and reused across
  every candidate pair that touches that route in the same pass.
- **A generation-stamped pair-delta cache** (`pairCache`) — an experimental per-candidate-pair
  delta cache, implemented and verified correctness-neutral, but measured to target the wrong
  bottleneck (see Section 16) and left disabled by default.

**`SVCCache`** — the queue of customers still eligible for local-search evaluation in the
current pass. Any operator that moves a customer inserts every customer whose neighborhood
just changed back into this queue, so the local search only re-examines what actually needs
re-examining, not the whole instance, each iteration.

## 4. Stage 0 — partitioning and candidate lists

Customers are sorted along a **Hilbert space-filling curve** — a `O(n log n)` sort, not a
clustering algorithm — which has the property that customers close together on the curve are
close together geographically. The sorted sequence is sliced into `P` contiguous chunks
(`P` = number of worker threads), giving each thread a spatially compact, roughly equal-sized
sub-problem to solve independently.

This stage also builds the candidate neighbor lists (k-nearest-neighbor lists per customer,
used by every operator to restrict which move partners are even considered — no operator ever
searches over the full customer set) and, when route minimization or Clarke-Wright is enabled,
a second, wider candidate list for those specific uses (Section 5, 6).

## 5. Stage 1 — construction

Two construction methods exist, selectable with `--construction`:

- **MST + randomized DFS** (the original method): build a minimum spanning tree over each
  chunk, then walk it via randomized depth-first traversal, splitting into vehicle-capacity-
  sized routes as demand accumulates.
- **Clarke & Wright savings** (`--construction cw`, the current default): the classical savings
  algorithm — start with one route per customer, then greedily merge the pair of routes with
  the largest "savings" (cost saved by joining them end-to-end) that doesn't violate capacity.
  Clarke & Wright alone produces *worse* solutions than MST + DFS on its own, but combined with
  route minimization immediately after (Section 6) it produces both fewer routes and lower
  cost — the two techniques were found to be complementary, not substitutes.

Clarke & Wright uses the *wide* candidate list, not the narrow one used for local search:
measured directly, its solution quality degrades sharply as the candidate list narrows (VDA:
k=30 → 846 routes / 23.5M cost, k=1000 → 800 routes / 21.9M cost), because a route whose
endpoints can't see a merge partner in a narrow list simply never merges.

## 6. Route minimization (T3)

ROUTEMIN (`stage1_5_routemin`, `Stage2_ILS.cpp`) runs once per chunk, immediately after
construction, mirroring FILO2's own pipeline placement. Its job is specifically to reduce
route *count* (not cost directly) — every removed route is one fewer depot round-trip, and at
the cost scales this project operates at, route count is a large fraction of total cost.

Mechanically, it's a destroy-and-repair loop: pick a customer and its route, plus one
neighboring route; destroy both routes entirely; try to reinsert every displaced customer into
some other route with residual capacity (using the same wide candidate list Clarke & Wright
uses, for the same reason); anything that can't be reinserted either opens a new route or, if
the live route count is still above a computed lower bound (`kmin`, a greedy first-fit-
decreasing estimate), is left for the next iteration to try again. A local search pass (using
the *narrow* candidate list — see Section 8's cost note) tidies each iteration's result before
accepting or rejecting it via simulated annealing.

This stage was directly profiled at large scale and found to be the specific mechanism behind
this project's largest remaining performance gap (Section 14, Lombardia) — see
`010_can_this_architecture_beat_filo2.md` §0.22–0.23 for the full investigation.

## 7. Stage 2 — parallel iterated local search

Per chunk, independently and in parallel: a **ruin-and-recreate** loop wrapped in **simulated
annealing** acceptance.

- **Ruin**: pick a random customer, remove it, then remove a short random walk of its
  geographic neighbors (walk length `ceil(ln(chunk size))` by default, fixed rather than
  adaptive, tunable via `--ruin-mult`) — a small, localized destruction, not a large-scale
  restructuring. Measured directly: this fixed walk destroys ~9.2 customers per iteration on
  average against FILO2's own adaptive walk, observed at ~23 — a real, measured difference,
  and one lever partially explored (`--ruin-mult`) but not conclusively resolved, see
  Section 16.
- **Recreate**: reinsert every removed customer at its best-found position among nearby routes.
- **Local search to convergence**: after recreate, run the full local-search operator sweep
  (Section 8) on every customer whose neighborhood changed, repeatedly, until no further
  improving move is found.
- **Accept/reject**: standard simulated annealing — always accept an improving result; accept a
  worsening one with probability `exp(-Δ/T)`, `T` cooling geometrically over the run.

Each chunk's Stage 2 is bounded by a wall-clock time budget (`--stage2-ms`), not an iteration
count (Section 12).

## 8. The local-search operator catalog

Twenty move operators are implemented, dispatched from one shared evaluation loop (`local_
search`, `Stage2_ILS.cpp`) that, for a popped customer `i` and each of its `k` nearest
candidates `j`, evaluates every applicable operator's cost delta and applies whichever is
best-improving (subject to SA acceptance):

| family | operators | notes |
|---|---|---|
| Single relocation | `relocate` | move one customer to a better position |
| Multi-customer relocation | `relocate2`, `relocate3` (+ reversed variants) | move a 2- or 3-customer segment |
| Exchange | `swap` | swap two customers' positions |
| 2-opt family | `2opt`, `2opt*` | reverse a route segment; reconnect two routes across a cut |
| `swap*` | — | exchange two customers between routes at their best (not necessarily current) insertion points, using the Stage 1-precomputed top-3 tables |
| Segment exchange | `E21`/`E22`/`E31`/`E32`/`E33` + reversed variants | swap a 2- or 3-customer segment from one route for a 1-, 2-, or 3-customer segment from another; cross-route only |
| Ejection chain (depth 2) | `eject2` | when a plain relocation would overflow the destination route's capacity, eject one customer from that route into a third route to make room, rather than giving up |

The segment-exchange family and the ejection chain are ports of FILO2's own published move
definitions, re-derived term-by-term against FILO2's source and adapted to this project's
`pred`/`succ` + `costToPred` data layout (FILO2 uses a different internal representation). The
ejection chain specifically is a deliberately *bounded* approximation: FILO2's own version
searches ejection chains to depth 25 via a priority queue; this implementation stops at depth
2, a scoping choice made to bound per-move cost (searching deeper multiplies cost by route
size × candidate width at every additional hop) rather than a claim that depth 2 is sufficient
in general. A depth-3 extension was implemented and tested; it was measured to make solution
quality *worse* on average despite being individually correct (a search-trajectory effect, not
a bug — see `010...md` §0.20) and is present in the code but not active.

Every operator's cost delta is evaluated in O(1) — a function of the specific edges a move
would change, using the cached `costToPred` values, never a route walk — with one documented
exception: `swap*`'s Stage 1 precompute (`get_top3_insertions`) does walk a whole candidate
route once per route per local-search pass. This is the project's most expensive remaining
per-iteration cost and the specific mechanism found behind ROUTEMIN's scaling problems at
large vehicle capacities (Section 6, Section 14).

## 9. Stage 3 — boundary healing

Partitioning (Stage 0) is what makes Stage 2 parallel with zero lock contention, but it comes
at a real cost: routes near a chunk boundary were optimized as if the neighboring chunk didn't
exist, so some clearly-beneficial cross-chunk moves are missed. Stage 3 repairs this.

After all chunks' Stage 2 finishes and their solutions are merged into one global solution,
every pair of chunks that share boundary customers gets a dedicated local-search pass over
just those boundary customers (the same `local_search` core as everywhere else, restricted to
the two chunks' customers). Running every chunk pair naively, one at a time, would make this
stage a serial bottleneck exactly where parallelism mattered most — so instead, the chunk-pair
"conflict graph" (two pairs conflict if they share a chunk) is **edge-colored**: pairs in the
same color class share no chunk, so they can run fully in parallel with no locking; different
color classes run one after another. At `P=4` chunks this needs exactly 3 color classes (the
edge-chromatic number of the complete graph on 4 nodes), so boundary healing gets most of the
benefit of full parallelism without the coordination cost of solving it for every pair at once.

## 10. Stage 4 — route cleanup

A cheap, single-threaded pass: any route below a load threshold (default 20% of `Q`, or with
2 or fewer customers) has its customers evaluated for relocation into a different route. Every
relocation is checked against the same "never increase cost" acceptance rule as everywhere
else — this stage cannot make the solution worse, only occasionally consolidate a
near-empty route into others.

## 11. Stage 5 — serial polish

One final pass over the fully merged, healed solution: a full local-search sweep to a local
optimum, followed by the same ruin-recreate-anneal loop as Stage 2, but single-threaded and
over the whole instance rather than one chunk. This is where any remaining cross-chunk
improvement opportunities that Stage 3's narrower healing didn't reach get a chance to surface,
at the cost of being the one part of the pipeline that doesn't parallelize.

## 12. Time-budget scheduling

Every stage that runs a search loop (Stage 2, Stage 3, Stage 5, and ROUTEMIN) is bounded by a
**wall-clock time budget**, not an iteration count (`--stage2-ms`, `--stage3-ms`,
`--stage5-ms`, `--routemin-iters`) — this is what makes equal-time comparisons against FILO2
meaningful, and what makes the pipeline's own stage-to-stage time split independently tunable.
Two real bugs were found and fixed in this mechanism during this project (a stale clock
reference that let Stage 5 silently run ~2× its budget, and a per-color-class budget in Stage 3
being applied as if it were a per-stage total) — see `010...md` §0.10–§0.11 for the full
writeup, since getting this scheduling right turned out to matter as much for the final
results as any algorithmic change.

## 13. Verification and benchmarking methodology

Three principles run through every measurement in this project:

1. **Feasibility and cost are independently recomputed, never trusted from either solver's
   self-report.** `src/verifier.py` recomputes this solver's route costs and capacity
   constraints directly from its route output; `src/verify_filo2.py` does the same for FILO2's
   `.vrp.sol` output (built specifically after a stale-ID-encoding bug in FILO2's own output
   format produced a false capacity-violation signal that turned out to be in the *checker*,
   not FILO2 — see `010...md` §0.16's addendum).
2. **Comparisons are run at genuinely equal wall clock**, with both solvers' actual time
   budgets checked, not assumed — a stale FILO2 time budget in early comparison scripts (left
   at this project's *older*, slower wall clock after two timing bugs were fixed) silently
   understated this project's real standing for a period, until caught and corrected.
3. **Sample sizes are chosen to actually resolve the effect being measured, not the first
   sample that gives a clean answer.** A 5-seed comparison at Valle-D'Aosta looked like a win;
   extending to 15 seeds reversed the sign. Every headline result in this project's later
   reports states its sample size and, where relevant, a significance statistic (a two-sample
   t-test), specifically because of that reversal.

## 14. Current standing

At genuinely equal wall clock, both sides independently verified:

| Instance | Scale | Result |
|---|---|---|
| Lazio | ~1,000,000 | **Win, both axes** — 0.183% cheaper (10 seeds, t = −16.4), 27% faster |
| Valle-D'Aosta | ~180,000 | **Statistical tie** — +0.023% at n=15, \|t\| ≈ 1.0 |
| Lombardia | ~950,000 | **Loss, cause diagnosed** — 0.106%, narrowed to 0.088%; the full gap closes with more route-minimization time than currently fits the budget |

Full detail, including the complete chronological derivation of these numbers and every
verification step: `010_can_this_architecture_beat_filo2.md`. Raw per-seed data:
`results_summary.xlsx`.

## 15. What's original here, and what isn't

Worth stating plainly rather than leaving implicit: the local-search operator definitions
(segment exchange, ejection chain) and the route-minimization heuristic are ports of FILO2's
published techniques, re-derived and re-implemented against this project's own data
structures, not independent inventions. The parts that are this project's own design are the
partition/parallel-search/heal architecture as a whole, the specific scoping decisions within
it (the depth-2 ejection chain bound, the edge-coloring healing scheduler, the capacity-
adaptive tuning found while closing the Lombardia gap), and the empirical, scale-dependent
characterization of when this architecture wins, ties, or loses against a strong single-
threaded baseline. As engineering and as a rigorous empirical comparison this stands on its
own; as a research contribution it is not yet clearly novel — see `010...md`'s closing
sections for a fuller discussion of that specific question and the open direction (cooperation
between chunks *during* search, not only after it) that would need exploring to change that.

## 16. Known open items

- **ROUTEMIN's per-iteration cost scales with vehicle capacity** (via `swap*`'s route-walking
  precompute), and at large `Q` this currently prevents route minimization from running long
  enough, in the allotted time budget, to close the full Lombardia gap — the mechanism is
  understood and partially mitigated (a capacity-adaptive cap on the precompute), but a
  genuinely faster route-minimization routine would likely close the remaining gap outright.
- **This project's ruin size is smaller than FILO2's** (~9–13 customers destroyed per iteration
  vs FILO2's ~23) at default settings; whether closing that gap helps was only partially
  explored (`--ruin-mult`) and not conclusively resolved.
- **The pair-delta cache (`pairCache`/T2-lite) targets the wrong bottleneck.** It was built to
  speed up the per-candidate-pair evaluation loop, but profiling showed the actual dominant
  cost is the `swap*` route-walking precompute (Step 1), which the cache doesn't touch — it
  measurably reduced throughput without reducing real work, and is disabled by default.
- **Boundary healing (Stage 3) only reconciles chunks once, after both finish independently.**
  Cooperation between chunks *during* search rather than only after it is the open,
  unexplored direction most likely to be a genuinely novel contribution rather than a
  refinement of existing technique — see `010...md`'s discussion of this for the reasoning and
  the relevant literature.
