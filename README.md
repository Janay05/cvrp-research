# Parallel Chunked CVRP Solver

**Author:** Janay Bhanushali · [github.com/Janay05](https://github.com/Janay05) · [MIT licensed](LICENSE)

A multi-threaded C++ solver for the Capacitated Vehicle Routing Problem (CVRP). Instead of
running one search over the whole instance, it partitions the graph geographically into
independent chunks, runs iterated local search on each chunk in parallel, then heals the
boundaries the partitioning created. It's benchmarked directly against
[FILO2](https://arxiv.org/abs/2306.14205) (Accorsi & Vigo), the published state-of-the-art
CVRP solver, on FILO2's own Italian-region benchmark instances (Valle-D'Aosta, Lazio,
Lombardia — 180K to ~1M customers), so every comparison below is against numbers that
solver's own paper reports, on the instances it was designed and tuned for.

## Current results, at equal wall clock

| Instance | Scale | Result |
|---|---|---|
| **Lazio** | ~1,000,000 customers | **Win, both axes.** 0.183% cheaper (10 seeds, t = -16.4), 27% faster, zero overlap between the two solvers' cost distributions. |
| **Valle-D'Aosta** | ~180,000 customers | **Statistical tie.** +0.023% at n=15, \|t\| ~ 1.0 (a real ILS win requires \|t\| well above that). |
| **Lombardia** | ~950,000 customers | **Loss, cause diagnosed.** 0.106%, narrowed to 0.088% under budget; closing it outright needs ~3x the current time budget in one specific pass — a known, scoped engineering target, not an open question. |

Every number above is independently recomputed from raw route data (`src/verifier.py` for
our own output, `src/verify_filo2.py` for FILO2's), never taken from either solver's own
self-reported cost. Full methodology, the seed counts behind each result, and a
5-seed-vs-15-seed reversal that's the reason those seed counts are what they are:
[`docs/reports/010_can_this_architecture_beat_filo2.md`](docs/reports/010_can_this_architecture_beat_filo2.md).

## How the pipeline works (summary — full depth in [`docs/reports/012_architecture_overview.md`](docs/reports/012_architecture_overview.md))

```
Stage 0: Hilbert-curve partition + k-NN candidate lists
   |
Stage 1: per-chunk construction (MST+DFS or Clarke-Wright, --construction)
   |
T3/ROUTEMIN: per-chunk route-count minimization (ported from FILO2)
   |
Stage 2: per-chunk parallel ILS (ruin / recreate / local_search), P threads, no cross-chunk visibility
   |
Stage 3: graph-colored parallel boundary healing (disjoint chunk-pairs run concurrently, lock-free)
   |
Stage 4: route cleanup (dissolve near-empty routes where it doesn't cost anything)
   |
Stage 5: single-threaded polish over the full, un-partitioned graph
```

Threads never touch another thread's chunk during Stage 1/2 — that's what makes them
embarrassingly parallel. The cost of that isolation is chunk-boundary artifacts, which Stage 3
(parallel, still no cross-thread locking on the hot path — chunk-pairs are edge-colored so a
color class's pairs are pairwise disjoint) and Stage 5 (serial, whole graph, cheap because it's
short) exist specifically to fix.

**Local search** (`local_search` in `Stage2_ILS.cpp`) dispatches 20 operators from one shared
sweep: relocate/relocate2/relocate3 (+reversed variants), swap, 2-opt, 2-opt\*, SWAP\*,
E21/E22/E31/E32/E33 (+reversed variants, FILO2's segment-exchange family), and a depth-2
ejection chain (`eject2`). A depth-3 extension (`eject3`) is implemented and kept in the code
but not dispatched — measured net-negative in multi-seed testing (see report 012, "Known open
items").

**Core data structures**: `Solution` is a flattened linked list (`pred`/`succ` arrays) with
`routeOf`, `routeHead`/`routeTail`/`routeLoad`, `routePosition`/`cumLoad` (both kept current in
O(route length), never O(N)), and an incrementally-tracked `totalCost`. Each thread owns a
`ThreadArena` (do/undo log for rollback, SWAP\*'s top-3-insertion cache, an `SVCCache` gating
queue so idle nodes aren't re-evaluated every iteration) — no heap allocation happens in the
search hot path.

## Building

Requires CMake and a C++17 compiler. **Benchmark/timing claims must be built this way** — see
"A note on compilers" below.

```bash
# WSL / Linux (the verified benchmarking baseline)
cd src
mkdir build_wsl && cd build_wsl
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . -j
```

```powershell
# Windows / MSVC (fine for day-to-day development, not for speed comparisons)
cd src
mkdir build; cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . --config Release
```

**A note on compilers**: MSVC on Windows silently ignores `-O3 -march=native`. Every FILO2
comparison in this repo is run with both solvers rebuilt from scratch under WSL with the same
g++ invocation, verified directly from the compile command line — not just trusted from
CMake's summary — because an earlier round of benchmarking accidentally compared two different
compilers' codegen instead of the two solvers (`docs/reports/008_verified_linux_benchmarking.md`).
Building on Windows is fine for iterating on the code; don't use it to produce a timing claim.

## Running

```bash
./cvrp_parallel -f <path/to/instance.vrp> -p <num_threads> [flags...]
```

| Flag | Meaning | Default |
|---|---|---|
| `-f <path>` | CVRPLIB-format `.vrp` instance to load | (generates a synthetic 2000-node instance) |
| `-p <n>` | number of parallel chunks/threads | 4 |
| `--seed <n>` | RNG seed | — |
| `--out <path>` | output solution path | `results/final_solution.txt` |
| `--log <path>` | per-worker log path | — |
| `--stage2-ms <n>` | Stage 2 (per-chunk ILS) time budget, ms | — (legacy iteration mode if unset) |
| `--stage3-ms <n>` | Stage 3 (boundary healing) time budget, ms | — |
| `--stage5-ms <n>` | Stage 5 (serial polish) time budget, ms | — |
| `--iters-per-node <k>` | legacy mode only: per-thread iterations = `inst.n * k` | 50 |
| `--max-iterations <n>` | legacy mode only: absolute per-thread iteration override | — |
| `--construction <cw\|mst>` | Stage 1 construction heuristic | mst |
| `--cw-neighbors <n>` | candidate-list width for Clarke-Wright construction | — |
| `--routemin-iters <n>` | ROUTEMIN (route-count minimization) iteration budget; 0 disables it | 0 |
| `--routemin-k <n>` | candidate-list width used by ROUTEMIN | — |
| `--ruin-mult <x>` | scales ruin-walk length (`ceil(ln(chunkSize) * x)`) | 1.0 |
| `--stage4-dissolve-frac <x>` | Stage 4's route-dissolution load threshold, as a fraction of `Q` | 0.2 |

Time-budget mode (`--stageN-ms`) is what every reported benchmark number in this repo uses —
it gives every thread the same wall-clock allowance regardless of instance size, which the
older iteration-count flags don't (see report 012, "Time-budget scheduling"). The legacy flags
still work and are regression-tested, but are mainly useful for exact-iteration-count
reproducibility, not for a real benchmark.

**Example** (Lazio, the actual settled configuration behind this repo's headline result — see `docs/reports/010_can_this_architecture_beat_filo2.md`):
```bash
./cvrp_parallel -f data/instances/I/Lazio.vrp -p 4 --seed 1 \
  --routemin-k 500 --routemin-iters 12000 \
  --stage2-ms 45000 --stage3-ms 12000 --stage5-ms 45000
```
Note the low `-p`: this project's own benchmarking settled on `-p 4` even at ~1M customers — a
higher `-p` was tried in earlier work (see reports 003/008) but isn't the current validated
configuration for any of the three headline instances. Don't assume a higher `-p` is safe or
beneficial without re-measuring; see "Memory" below.

## Memory

Each worker thread's scratch memory (`ThreadArena`) is sized against the *full* instance, not
its chunk share — so peak memory scales with `P` (thread count) at a given instance size, not
just with instance size alone. Measured directly (`/usr/bin/time -v`, Lazio, ~1M customers,
`-p 4`, the real settled config above): **9.29 GB peak resident memory**, about 23% more than
FILO2's 7.17 GB on the identical instance and machine. At `-p 16` on the same instance this
reliably exceeds a 10GB memory ceiling and crashes rather than degrading gracefully — confirmed
directly, not theoretical.

**Practical implication:** running the three headline instances (all ~180K-1M customers) at
their validated configs wants **10GB+ of available RAM**; on a tighter budget, use a lower `-p`
first rather than a shorter time budget. This is a known, understood cost of the per-thread
arena design (see `docs/reports/012_architecture_overview.md`, "Known open items") — not
something a config flag currently works around.

## Verifying a result

Never trust a solver's self-reported cost — independently recompute it:
```bash
python src/verifier.py <instance.vrp> <solution_output>       # our solver's output
python src/verify_filo2.py <instance.vrp> <filo2.sol>         # FILO2's native .sol format
```
Both recompute every edge cost from raw coordinates and check every customer is visited
exactly once with no route over capacity, rather than trusting either solver's own printed
header.

## Codebase map

| File | Purpose |
|---|---|
| `main.cpp` | Entry point: CLI parsing, orchestrates all 6 stages, writes `results/final_solution.txt`. |
| `Types.hpp` | `NodeId`, `Cost`, and the `Solution` struct (linked-list + dense-state arrays). |
| `ThreadArena.hpp` | Per-thread scratch: do/undo log, SWAP\* top-3 cache, `SVCCache`. |
| `Stage0_Partitioning.{hpp,cpp}` | Hilbert-curve chunking, k-NN candidate list construction. |
| `Stage1_Construction.{hpp,cpp}` | Per-chunk MST+DFS / Clarke-Wright construction. |
| `Stage2_ILS.{hpp,cpp}` | The bulk of the solver: all 20 `eval_*`/`apply_*` local-search operators, `ruin`/`recreate`, `local_search`, `stage2_ils`, ROUTEMIN (`stage1_5_routemin`), Stage 5's `stage5_serial_polish`. |
| `Stage3_MergeHealing.{hpp,cpp}` | Boundary-pair graph coloring and the parallel healing pass. |
| `Stage4_5_CleanupPolish.{hpp,cpp}` | Stage 4 route cleanup. |
| `Worker.{hpp,cpp}` | Per-thread orchestration (Stage 1 -> ROUTEMIN -> Stage 2), called from `main.cpp`. |
| `VrpParser.{hpp,cpp}` | CVRPLIB `.vrp` file reader. |
| `verifier.py` / `verify_filo2.py` | Independent cost/feasibility checkers, see above. |
| `tools/bench.py`, `tools/compare_bench.py`, `tools/score_sol.py` | Multi-seed benchmarking harness, BKS/FILO2 comparison scoring. |

## Documentation

- **[`docs/reports/012_architecture_overview.md`](docs/reports/012_architecture_overview.md)** — the full standalone architecture deep-dive: every stage in detail, the complete operator catalog, verification/benchmarking methodology, and an explicit "what's original here vs. what's a port of FILO2" section.
- **[`docs/reports/011_short_summary.md`](docs/reports/011_short_summary.md)** — a 2-page summary of what changed in the most recent work pass and its measured effect.
- **[`docs/reports/010_can_this_architecture_beat_filo2.md`](docs/reports/010_can_this_architecture_beat_filo2.md)** — the detailed, still-growing research log behind the current results table above.
- **[`docs/reports/`](docs/reports/)** — the full numbered history (001-010) this project's results were built up through. Kept for provenance: several source-code comments cite specific reports/phases by name as the record of *why* a non-obvious piece of code is the way it is (e.g. `Stage2_ILS.cpp`'s SA temperature, `ThreadArena.hpp`'s cap sizing) — grep for `docs/reports/` in `src/` before removing any of them. Most readers taking over this project should start with 012 or 011 above, not this log.

## Notes for future maintainers (architectural invariants)

1. **Never use full `Solution` copies inside concurrently-running threads.** Reverting via
   `bestSol = globalSolution` inside Stage 3 (which shares one `globalSolution` across threads)
   causes real data races on `std::vector`'s internal pointers. Per-chunk `Solution`s in Stage 2
   are thread-local and fine to copy.
2. **`apply_undo_list` (`Stage2_ILS.cpp`) rolls back by flipping logged `INSERT`/`REMOVE`
   entries on the `pred`/`succ` pointers**, then calls `update_route_info()` to rebuild
   `routePosition`/`cumLoad` in O(route length). Don't try to reverse those dense arrays via
   manual delta offsets — the full O(L) rebuild is both simpler and fast enough (routes are
   short) that there's no real performance case for avoiding it.
3. **Route-slot capacity must be a provable bound, not a cushion.** `local_search`'s evaluation
   phase reads `routeHead`/`routeTail`/`routeLoad` without a lock by design; if `recreate()`
   ever needs to actually reallocate one of those vectors while another thread holds an
   unlocked pointer into it, that's a real crash (this happened twice historically — a stale
   `N+1` sizing and later an under-sized "right-sized" arena). The current bound is
   `2 * inst.n + 10000`, used consistently by Stage 3, Stage 5's arena, and anything else that
   creates routes. Any new per-route array must use the same bound.
4. **`numRoutes` snapshot/restore-on-rejection is only safe for a thread-local `Solution`.**
   `stage2_ils` and `stage5_serial_polish` both do it safely. `stage3_healing_ils_pass` shares
   one `globalSolution` across concurrently-running threads and deliberately does **not** —
   restoring a shared counter from a thread-local snapshot would race with another thread's
   concurrent route creation and silently erase its work. `apply_undo_list` already unwinds a
   rejected iteration correctly on its own (any route it created is just left empty, which is
   harmless); don't "fix" this by adding the snapshot/restore back to Stage 3.
5. **MSVC doesn't support ThreadSanitizer.** Concurrency correctness here is checked
   empirically: `run_loop.ps1` runs the solver repeatedly at a fixed seed. **In legacy
   iteration-count mode, this must be bit-identical every time** — any divergence there means a
   real race, not float non-determinism (the solver has none — costs are integer/deterministic
   given a seed). **In time-budget mode (`--stageN-ms`, what every real benchmark uses), small
   run-to-run cost variance is expected and not a bug**: Stage 5's polish loop stops on elapsed
   wall-clock time, so ordinary scheduling jitter changes how many iterations complete.
   Confirmed directly (2026-09): 3 repeated runs at a fixed seed in time-budget mode gave
   slightly different final costs, while the same seed in legacy mode gave the exact same cost
   3/3 times — both outputs were fully feasible in every case. If you need bit-identical output
   for debugging, use legacy mode (`--max-iterations`), not `--stageN-ms`.

## Honest scope note

The local-search operators and ROUTEMIN are close ports of FILO2's published techniques,
adapted to this codebase's data structures. What's original here is the parallel
partition/heal architecture itself, the specific scoping decisions in the operators ported
into it (e.g. the ejection chain's depth-2 cutoff), and the empirical scale-dependent
characterization in the results table above (the finding that Lazio and Lombardia, similar
scale, similar architecture, land on opposite sides of a win/loss is not something this
project set out looking for). See report 012's "What's original here, and what isn't" section
for the full accounting, and report 011's closing note for where this stands relative to a
publishable research contribution.
