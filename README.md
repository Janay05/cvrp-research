# Parallel Chunked CVRP Solver

This repository contains a high-performance, multi-threaded C++ solver for the Capacitated Vehicle Routing Problem (CVRP). It accelerates traditional Iterated Local Search (ILS) and Hybrid Genetic Search (HGS) paradigms by geographically partitioning the routing graph and solving the sub-problems concurrently, followed by a parallel boundary-healing phase.

## Current State & Results (August 2026)
The pipeline is **stable and feasible under its default configuration** (no data races, no dropped/duplicated nodes, independently verified) and now **loads real CVRPLIB `.vrp` files** via `-f <path>` (it previously only ran on its own synthetic 2000-node instance). See [`docs/reports/`](docs/reports/) for the full sequential history (stage-by-stage breakdowns, FILO2 comparisons, bugs found/fixed, and what changed between each report).

**Scale testing ([report 003](docs/reports/003_scale_testing.md)):** verified correct and feasible from 50 to ~1,000,000 customers (CMT1 → Valle-D-Aosta → Lazio, the same instance family FILO2 ships). At 50 nodes the solver is essentially optimal against the published best-known solution (525 vs. 524.611).

**Time-budget scheduling & cost tuning ([report 004](docs/reports/004_time_budget_scheduling.md)):** replaced the not-scale-aware iteration-budget formula with wall-clock time budgets per stage (`--stage2-ms`/`--stage3-ms`/`--stage5-ms`), modeled on FILO2's own time-based cooling schedule — the same requested budget now behaves sensibly at any instance size, and this also fixed a per-thread timing imbalance found in report 003 (workers used to finish up to 175s apart on equal-sized chunks; now within milliseconds of each other). Using the new mechanism to tune cost: **at 20,000 nodes, now 2.3x faster than FILO2 at only +1.45% cost** (beating the previous default on both speed and quality); **at 1,000,000 nodes, 3.87x faster at +1.10% cost**. Legacy behavior (no time-budget flags) is unchanged and regression-tested against reports 001/002's exact numbers.

The N=2000 numbers below are kept for continuity with reports 001/002; see report 003 for the more representative large-scale results.

**Benchmark on `test_2000.vrp` (N=2000 nodes), rebuilt Release binary, re-run and independently re-verified 2026-08-06:**
- **FILO2 (Baseline, Single-Threaded, 100,000 iterations):** 114s | Cost: 51,878 | *(real measurement — actual `filo2.exe` binary run against our exact instance, feasibility confirmed independently)*
- **Our Solver (P=1, 100,000 ILS iterations):** ~77-110s (run-to-run variance) | Cost: 54,589
- **Our Solver (P=4, default config — 100,000 ILS iterations *per thread*, same as P=1):** ~11.7s | Cost: 55,715

Both of our solver's default-config outputs (P=1 and P=4) pass full independent verification: every node visited exactly once, no route over capacity, and cost/route-count headers cross-checked against a from-scratch recomputation.

**Verification:**
An independent Python script (`verifier.py`) strictly verifies the C++ outputs (`results/final_solution.txt`): every node visited exactly once, no route exceeds capacity, Euclidean distances recomputed from scratch (not trusted from the solver), and the recomputed cost and route count are asserted to match the `Final Cost:`/`Num Routes:` header the solver reports, rather than just being printed alongside it.

**Bugs found and fixed:**
- ~~Stale `Num Routes` header~~ ([report 001](docs/reports/001_p1_p4_filo2_baseline.md)) — **fixed**. `main.cpp` now counts and reports only the routes it actually writes, rather than trusting a stale allocated-slot count.
- ~~Stage 3 incremental cost bookkeeping drift~~ ([report 001](docs/reports/001_p1_p4_filo2_baseline.md)) — **fixed**. Accepted healing moves never updated `totalCost`; each healing thread now accumulates its own delta and deltas are summed once per graph-coloring class after that class's threads join (no shared-scalar race).
- ~~Capacity constraint violations under sustained optimization pressure~~ ([report 002](docs/reports/002_capacity_fix_and_rebalance.md)) — **fixed**. Root cause: `stage5_serial_polish` (`Stage2_ILS.cpp`) was missing a full route-info rescan between `recreate()` and `local_search()` that `stage2_ils` and `stage3_healing_ils_pass` both have, so `eval_2opt_star`'s capacity check could read stale `cumLoad[]` and pass an over-capacity move. Fixed by adding the missing rescan; verified via bisection instrumentation and independently reconfirmed feasible by `verifier.py`.

**Why P=4 is fast, and what the remaining gap looks like:** `Stage2_ILS.cpp`'s iteration budget is now `inst.n * 50` per thread (was `chunkSize * 50`), so P=4 threads do the *same absolute* search as P=1, just on smaller, cheaper 500-node sub-problems in parallel — that's why 4x more aggregate search (100k vs P=1's 100k... per thread, so 400k total) still finishes in ~11.7s. P=4 is now within **+7.4% of FILO2** and **+2.1% of P=1** on cost (down from +8.2%/+2.8% under the old, iteration-starved default), at 6-10x the speed of either. Doubling the iteration budget further only closed another 0.5% of the gap — a diminishing-returns signal that the bottleneck is now structural (chunk-boundary blindness during Stage 2, only partially recovered by Stage 3/5), not raw iteration count. See [report 002 §5](docs/reports/002_capacity_fix_and_rebalance.md#5-whats-actually-causing-the-remaining-7-gap-and-what-to-do-about-it) for the recommended next steps (extending Stage 5's and Stage 3's currently-small, hardcoded iteration budgets is the top candidate).

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
