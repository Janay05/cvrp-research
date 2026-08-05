# Parallel Chunked CVRP Solver

This repository contains a high-performance, multi-threaded C++ solver for the Capacitated Vehicle Routing Problem (CVRP). It accelerates traditional Iterated Local Search (ILS) and Hybrid Genetic Search (HGS) paradigms by geographically partitioning the routing graph and solving the sub-problems concurrently, followed by a parallel boundary-healing phase.

## Current State & Results (August 2026)
The pipeline is **stable and feasible on every run verified so far** (no data races, no dropped/duplicated nodes). The P=4 speedup is genuine, but it is **not an apples-to-apples comparison with P=1** — the two configurations do different amounts of optimization work, and the numbers below reflect that honestly (previous numbers in this section overstated the speedup; see the investigation notes below for how these were re-measured).

**Benchmark on `test_2000.vrp` (N=2000 nodes), rebuilt Release binary, re-run and independently verified 2026-08-06:**
- **FILO2 (Baseline, Single-Threaded):** 92 seconds | Cost: 51,878 *(unverified, external figure — not re-measured in this pass)*
- **Our Solver (P=1, 100,000 ILS iterations):** 90.7 seconds | Cost: 54,589 | 9.39B distance evaluations
- **Our Solver (P=4, default config — 25,000 ILS iterations *per thread*):** 6.8 seconds | Cost: 56,131 | 1.12B distance evaluations
- **Our Solver (P=4, equal-iteration control — 100,000 ILS iterations per thread, same as P=1):** 18.4 seconds | Cost: 53,616 | 3.80B distance evaluations

**Why P=4 is fast — explained, not assumed:**
`Stage2_ILS.cpp`'s `max_iterations = chunkSize * 50` means P=4's four threads each run only `(2000/4)*50 = 25,000` iterations on their own 500-node sub-problem, not 100,000 each — the *aggregate* iteration count across all threads happens to equal P=1's 100,000, but each thread individually does 4x less search. On top of that, each iteration is intrinsically cheaper on a smaller chunk (candidate neighbors are pruned to same-chunk-only, `Stage2_ILS.cpp:672-678`), which is why distance evaluations drop ~8.4x, not just ~4x. Running the controlled experiment above — giving each P=4 thread the *same absolute* iteration count as P=1 — confirms this: wall time rises to 18.4s (a genuine ~4.9x speedup from parallelism + cheaper per-iteration cost on smaller chunks) and the cost actually **improves** on P=1 (53,616 vs 54,589), since each subproblem gets far more search intensity per node. The default config's 6.8s is real, but it's fast specifically because it does less absolute work per thread, not because chunking made the same work 13x cheaper.

**Verification:**
An independent Python script (`verifier.py`) strictly verifies the C++ outputs (`results/final_solution.txt`): every node visited exactly once, no route exceeds capacity, Euclidean distances recomputed from scratch (not trusted from the solver), and — as of this pass — the recomputed cost and route count are now asserted to match the `Final Cost:`/`Num Routes:` header the solver reports, rather than just being printed alongside it.

**Known issues found while re-verifying (not yet fixed, tracked here for follow-up):**
- `Stage4_5_CleanupPolish.cpp:101` sets `numRoutes = routeHead.size()` (the allocated route-slot count) rather than the number of routes that still contain customers. When a route is emptied during cleanup without shrinking `routeHead`, the `Num Routes:` header in `final_solution.txt` overcounts by the number of emptied slots (e.g. reported 21 vs. 20 actual routes on a P=4 run). Feasibility and cost are unaffected; the hardened `verifier.py` now catches this via the header cross-check.
- Stage 3 healing's incremental cost bookkeeping occasionally drifts from the from-scratch recomputed cost under P>1 (e.g. `Incremental=56744 vs Scratch=56372` on one P=4 run) — `main.cpp` already detects and self-corrects this by overwriting `totalCost` with the scratch value, so final results are unaffected, but the incremental delta tracking in `Stage3_MergeHealing.cpp` has a real drift bug worth investigating.

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
