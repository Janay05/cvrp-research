# Parallel Chunked CVRP Solver

This repository contains a high-performance, multi-threaded C++ solver for the Capacitated Vehicle Routing Problem (CVRP). It accelerates traditional Iterated Local Search (ILS) and Hybrid Genetic Search (HGS) paradigms by geographically partitioning the routing graph and solving the sub-problems concurrently, followed by a parallel boundary-healing phase.

## Current State & Results (August 2026)
The pipeline is **stable and feasible under its default configuration** (no data races, no dropped/duplicated nodes, independently verified). The P=4 speedup is genuine, but it is **not an apples-to-apples comparison with P=1** — the two configurations do different amounts of optimization work. See [`docs/performance_report.md`](docs/performance_report.md) (or the equivalent published report) for the full stage-by-stage breakdown and FILO2 comparison; the summary below is kept current with the latest verified run.

**Benchmark on `test_2000.vrp` (N=2000 nodes), rebuilt Release binary, re-run and independently re-verified 2026-08-06:**
- **FILO2 (Baseline, Single-Threaded, 100,000 iterations):** 114s | Cost: 51,878 | *(real measurement — actual `filo2.exe` binary run against our exact instance, feasibility confirmed independently)*
- **Our Solver (P=1, 100,000 ILS iterations):** ~90-110s (run-to-run variance) | Cost: 54,589
- **Our Solver (P=4, default config — 25,000 ILS iterations *per thread*):** ~7-8s | Cost: 56,131

Both of our solver's default-config outputs (P=1 and P=4) pass full independent verification: every node visited exactly once, no route over capacity, and cost/route-count headers now cross-checked against a from-scratch recomputation.

**Why P=4 is fast — explained, not assumed:**
`Stage2_ILS.cpp`'s `max_iterations = chunkSize * 50` means P=4's four threads each run only `(2000/4)*50 = 25,000` iterations on their own 500-node sub-problem, not 100,000 each — the *aggregate* iteration count across all threads happens to equal P=1's 100,000, but each thread individually does 4x less search. On top of that, each iteration is intrinsically cheaper on a smaller chunk (candidate neighbors are pruned to same-chunk-only, `Stage2_ILS.cpp:672-678`), which is why distance evaluations drop ~8.4x, not just ~4x. A controlled experiment (forcing each P=4 thread to run the same absolute 100,000 iterations as P=1) confirmed wall time rises to ~19-27s — a genuine ~4-5x speedup from parallelism + cheaper per-iteration cost on smaller chunks, not the ~13x the default config shows. That experiment also surfaced the capacity-violation bug noted below, so its cost figure is not reported here as a valid quality comparison.

**Verification:**
An independent Python script (`verifier.py`) strictly verifies the C++ outputs (`results/final_solution.txt`): every node visited exactly once, no route exceeds capacity, Euclidean distances recomputed from scratch (not trusted from the solver), and the recomputed cost and route count are asserted to match the `Final Cost:`/`Num Routes:` header the solver reports, rather than just being printed alongside it.

**Bugs found and fixed this pass:**
- ~~`Stage4_5_CleanupPolish.cpp` / `main.cpp`: stale `Num Routes` header~~ — **fixed**. The header used to report the allocated route-slot count rather than the number of routes actually written (routes emptied by Stage 5's ILS moves weren't excluded), e.g. reporting 21 when only 20 routes had customers. `main.cpp` now counts and reports only the routes it actually writes.
- ~~`Stage3_MergeHealing.cpp`: incremental cost bookkeeping drift~~ — **fixed**. Accepted moves in `stage3_healing_ils_pass` never updated `globalSolution.totalCost`, so the incrementally-tracked cost silently drifted from the true cost on every P>1 run (e.g. `Incremental=56744` vs `Scratch=56372`), masked only because `main.cpp` already recomputed and overwrote the cost from scratch afterward. Each healing thread now accumulates its own delta and returns it; deltas are summed into `totalCost` once per graph-coloring class after that class's threads join, avoiding a shared-scalar race across concurrently healing chunk pairs.

**Known issue found, not yet fixed (out of scope for this pass):**
- **Capacity constraint violations under sustained optimization pressure.** Forcing each P=4 thread to run 100,000 ILS iterations on its 500-node sub-problem (4x the default `25,000 = chunkSize*50` budget) produced a solution where 4 of 18 routes exceeded the 100-unit capacity (up to 170 units in one route) — caught by `verifier.py`'s independent capacity check, not by the solver's own `routeLoad` bookkeeping (which matched the actual, invalid, customer count exactly, so this isn't a load-tracking bug — it's a real gap in a capacity feasibility check somewhere in the relocate/recreate/local-search move evaluation in `Stage2_ILS.cpp`, or in the chunk-restricted variant used by Stage 3 healing, that only manifests after enough iterations). **This did not appear in either default-config run (P=1 or P=4) reported above — both were independently verified feasible** — but it means pushing the iteration budget higher (e.g. to close the P=4 cost gap, as this README previously suggested) is not currently safe without fixing this first. Recommended follow-up: bisect which move type produces the violation and add/repair the capacity check before increasing any iteration budget in production use.

---

## Project Structure & Pipeline Architecture

The pipeline executes in 5 discrete stages:

### Stage 0: Partitioning & Setup (`Stage0_Partitioning.hpp`)
- Reads the VRP instance and builds spatial $k$-Nearest Neighbor ($k$-NN) lists.
- Partitions the $N$-node graph into $P$ geographic chunks using a $k$-means clustering algorithm.
- Identifies **boundary nodes** (nodes in one chunk that share $k$-NN edges with nodes in another chunk) and builds a `boundaryChunkPair` graph.

### Stage 1: Parallel Construction (`Stage1_Construction.cpp`)
- Threads are spawned for each chunk ($P$ threads).
- Each thread runs a Greedy Insertion heuristic restricted purely to the nodes assigned to its chunk, building an initial set of valid routes.

### Stage 2: Parallel Iterated Local Search (`Stage2_ILS.cpp`)
- The threads continue operating strictly within their isolated chunks.
- Each thread runs Simulated Annealing with Iterated Local Search (ILS) consisting of `ruin`, `recreate`, and exhaustive `local_search` (relocate, swap, 2-opt, SWAP*).
- **$O(1)$ State Management:** To keep the inner loop incredibly fast, we avoid $O(N)$ route scans:
  - `routePosition[]` and `cumLoad[]`: Dense integer arrays tracking node positions and cumulative route loads. Re-calculated in robust $O(L)$ time during `update_route_info` (where $L \le 50$, the route length).
  - **Top-3 SWAP* Precomputation**: Precomputes the top-3 best insertion points for nodes before the inner $O(N^2)$ customer pair loop.
  - **`SVCCache` Gating**: A ring buffer cache that tracks recently modified nodes. The local search only evaluates neighborhoods around nodes in the cache, preventing redundant scans of stagnant routes.

### Stage 3: Parallel Merge Healing (`Stage3_MergeHealing.cpp`)
- To fix the sub-optimal routes artificially created along the chunk boundaries, we perform Merge Healing.
- **Concurrency Safety (Graph Coloring):** The `boundaryChunkPair` graph is edge-colored. All chunk-pairs within a single color class are guaranteed to be mutually disjoint. Threads are mapped to these disjoint pairs, allowing them to concurrently perform ILS on the boundaries without any data races or locking overhead.
- Note: Stage 3 `stage3_healing_ils_pass` relies on greedy descent in-place rather than making full `Solution` object copies, successfully avoiding C++ vector memory data races.

### Stage 4 & 5: Cleanup and Serial Polish (`Stage2_ILS.cpp` & `main.cpp`)
- Stage 4 performs internal memory cleanups.
- Stage 5 runs a brief, single-threaded serial polish across the entire un-partitioned $N$-node graph to smooth out any remaining global inefficiencies that the chunked boundaries missed.

---

## Codebase Map

| File | Purpose |
|------|---------|
| `main.cpp` | Entry point. Orchestrates the 5 stages, manages the timing profilers, and outputs to `results/final_solution.txt`. |
| `Types.hpp` | Defines base types (`NodeId`, `Cost`, `Solution` struct containing linked-list arrays `pred`, `succ`, and dense-state arrays). |
| `ThreadArena.hpp` | Defines the lock-free memory arenas (`DoUndoEntry`, `SVCCache`, `Top3Insertions`) used by individual threads to avoid heap allocations during search. |
| `Stage0_Partitioning.hpp` | $k$-means chunking and $k$-NN list generation. |
| `Stage1_Construction.cpp` | Greedy route construction. |
| `Stage2_ILS.cpp` | Core $O(1)$ state evaluation logic (`eval_relocate`, `eval_swap_star`), `apply_undo_list` for rollback, `ruin`/`recreate`, and the main SA loop. |
| `Stage3_MergeHealing.cpp` | Graph coloring and disjoint thread scheduling for boundary healing. |
| `verifier.py` | Independent Python script to validate CVRP constraints and recompute total cost. |
| `run_loop.ps1` | PowerShell stress-test script that runs the solver sequentially to hunt for non-deterministic thread behavior. |
| `run_p1.ps1` | PowerShell script to run the single-threaded baseline (`-p 1`). |

---

## Notes for Claude Code (AI Assistant Context)
If you are taking over this project, please note the following architectural invariants:
1. **Never use full `Solution` copies in parallel threads**: Reverting to `bestSol = globalSolution` inside concurrent execution spaces (like Stage 3) will immediately trigger undefined behavior data races due to `std::vector` internal pointer mutations.
2. **`apply_undo_list` design**: The `apply_undo_list` logic in `Stage2_ILS.cpp` handles rollback by flipping `INSERT` logs to `REMOVE` actions (and vice-versa) on the `pred`/`succ` pointers. Crucially, it then aggregates the modified routes and calls `update_route_info()` to regenerate the `routePosition` and `cumLoad` arrays in $O(L)$ time. Do not attempt to reverse these dense integer arrays manually using delta offsets; the current $O(L)$ full-rebuild is mathematically safer and microsecond-fast.
3. **Stage 5 capacity crash**: Stage 5 previously crashed because the `ThreadArena` arrays (like `route_visited_iter`) were sized to `N+1`. Stage 3 pre-allocates an expanded route limit (`N+100`), which caused Stage 5 to access out-of-bounds indices. The fix involved proper `prevNumRoutes` tracking to prevent unbounded route expansion. Ensure any new `ThreadArena` structures accommodate the expanded route capacity.
4. **Compiler Support**: MSVC on Windows does not support `-fsanitize=thread` (TSan). Concurrency integrity is verified empirically via `run_loop.ps1` (expecting bit-identical costs across 20+ runs).
