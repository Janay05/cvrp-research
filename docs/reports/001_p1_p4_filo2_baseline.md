# Performance Report — Parallel Chunked CVRP Solver

**Instance:** `test_2000.vrp` (N=2000 customers, uniform demand=1, capacity=100, EUC_2D) — generated deterministically (unseeded `rand()` behaves as `srand(1)`), so all runs below solve the identical instance.
**Build:** `src/build/Release/cvrp_parallel.exe`, MSVC Release (`-O3`/`-march=native` requested; MSVC ignores both flags — no `-O3` equivalent is currently applied on this platform, see Caveats).
**Date:** 2026-08-06. All numbers below are from binaries built after the bug fixes described in [Bugs Fixed](#bugs-fixed-this-pass), independently re-verified by a hardened `verifier.py` that recomputes cost and route count from scratch rather than trusting the solver's own report.

---

## 1. Headline comparison

| Solver | Iterations | Wall time | Cost | vs. FILO2 cost | Feasible? |
|---|---|---|---|---|---|
| **FILO2** (external baseline) | 100,000 (default) | 114.0 s | 51,878 | — | ✅ verified |
| **Our solver, P=1** | 100,000 | ~90–110 s (109.8 s shown) | 54,589 | +5.2% worse | ✅ verified |
| **Our solver, P=4 (default)** | 25,000 / thread | 7.9 s | 56,131 | +8.2% worse | ✅ verified |

P=1 is within striking distance of FILO2 on both cost and time using the same 100,000-iteration budget — a reasonable sequential ILS implementation, not yet as refined as FILO2's more elaborate move set (SWAP\*, granular RVND composer, adaptive shaking) but in the same ballpark. P=4 trades cost for a large wall-clock reduction, but — as detailed below — that reduction comes from doing genuinely less search per sub-problem, not from a free lunch.

---

## 2. Stage-wise breakdown

Time attribution differs structurally between the two codebases (FILO2 is a single monolithic ILS loop; ours is a 5-stage pipeline), so they're broken out separately.

### 2.1 Our solver — P=1 (single chunk, N=2000)

| Stage | Time | % of total |
|---|---|---|
| Stage 0 — Partitioning & k-NN setup | 8.8 ms | 0.01% |
| Stage 1 — Greedy construction | 25.3 ms | 0.02% |
| Stage 2 — ILS (100,000 iterations, chunk=2000) | 109,433 ms | 99.66% |
| Stage 3 — Boundary healing | 0.2 ms | ~0% *(no chunk pairs to heal at P=1)* |
| Stage 4 & 5 — Cleanup + serial polish | 332.6 ms | 0.30% |
| **Total** | **109,806 ms** | 100% |

Distance evaluations: **9,389,872,286**. Stage 2 (the ILS core) is essentially the entire runtime at P=1 — everything else is noise.

### 2.2 Our solver — P=4 (four 500-node chunks)

| Stage | Time | % of total |
|---|---|---|
| Stage 0 — Partitioning & k-NN setup | 12.7 ms | 0.16% |
| Stage 1 & 2 — Parallel construction + ILS (25,000 iter/thread, chunk=500) | 5,977.9 ms | 75.6% |
| Stage 3 — Parallel boundary healing (5 chunk-pairs, graph-colored) | 1,307.6 ms | 16.5% |
| Stage 4 & 5 — Cleanup + serial polish (full N=2000 graph) | 605.4 ms | 7.7% |
| **Total** | **7,903.6 ms** | 100% |

Distance evaluations: **1,122,380,414** (8.4x fewer than P=1, despite the *aggregate* iteration count across all 4 threads summing to the same 100,000).

Per-thread detail (Stage 1 / Stage 2 split, from per-worker logging added this pass):

| Worker | Chunk size | Stage 1 (construct) | Stage 2 (ILS) | Distance evals |
|---|---|---|---|---|
| 0 | 500 | 7.9 ms | 5,960.1 ms | 368,430,302 |
| 1 | 500 | 8.8 ms | 4,634.9 ms | 299,341,051 |
| 2 | 500 | 8.8 ms | 3,671.3 ms | 228,956,282 |
| 3 | 500 | 8.7 ms | 3,585.9 ms | 225,652,779 |

Stage 1&2's reported wall time (5,977.9 ms) is bounded by the *slowest* thread (Worker 0 at ~5,968 ms combined), as expected for a barrier-synchronized parallel stage — the other three threads finish early and idle. Note Stage 3 (healing) and Stage 4/5 (full-graph cleanup/polish) are proportionally much more expensive at P=4 (24% of total) than at P=1 (0.3% of total) — those stages don't shrink with chunking and become a growing share of the budget as P increases.

### 2.3 Why the ~8.4x drop in distance evaluations, not just ~4x

Aggregate ILS iteration count is identical between P=1 and P=4 (both sum to 100,000 across all threads — `Stage2_ILS.cpp:685`: `max_iterations = chunkSize * 50`, and chunks partition N evenly). The extra work reduction comes from two additional effects:
1. Each chunk's candidate neighbor lists are pruned to same-chunk-only (`Stage2_ILS.cpp:672-678`), so each local-search iteration considers a smaller candidate set on a 500-node sub-problem than on the full 2000-node graph.
2. Stage 1's greedy construction cost — separate from the iteration count — scales worse than linearly with problem size, so building 4 small routes concurrently is more than 4x cheaper in aggregate than building one large one.

A controlled experiment (forcing each P=4 thread to run the *same absolute* 100,000 iterations as P=1, rather than the default 25,000) produced a wall time of ~19-27s — consistent with ~4-5x speedup from parallelism plus the smaller-candidate-set effect, not the ~13.9x (109.8s → 7.9s) the default configuration shows. **That experiment also surfaced the capacity-violation bug described below, so its cost/quality result is not usable as a valid data point** — only its timing behavior is reported here as corroborating evidence for the above explanation.

### 2.4 FILO2 (baseline)

FILO2 doesn't expose a stage-by-stage timing breakdown in its default (non-`ENABLE_VERBOSE`) build; the pre-built `baselines/filo2/build/filo2.exe` reports only final cost and total wall time (`51878   114` in `<instance>_seed-0.out`). Structurally its pipeline is: Clarke & Wright construction → greedy route-count minimization (routemin) → 100,000-iteration ruin-and-recreate + randomized VND local search, all single-threaded, operating on the full instance at every iteration (no chunking).

---

## 3. Is the P=4 speedup real?

**Yes, but it is not doing the same amount of work as P=1** — this was the original question that triggered this investigation, and it's now answered with evidence rather than assumption:

- The default P=4 config's 7.9s is real and reproducible, and its output passes full independent verification (feasible, cost matches recomputation).
- It is fast specifically because each thread searches a 4x-smaller sub-problem for 4x fewer iterations with a 4x-smaller candidate-neighbor pool — not because parallel chunking made the same search 13.9x cheaper.
- When forced to do the same absolute amount of search as P=1 (100,000 iterations/thread), wall time rises to ~19-27s, consistent with a genuine ~4-5x speedup attributable to parallelism + smaller per-iteration cost — the actual "architecture benefit," once work is held constant.
- The cost gap (56,131 vs 54,589, +2.4%) reflects each 500-node sub-problem getting proportionally less absolute search budget than the full 2000-node problem gets under P=1, plus residual chunk-boundary artifacts only partially fixed by Stage 3 healing.

---

## 4. Bugs found and fixed this pass

Both were caught by hardening `verifier.py` to cross-check the solver's self-reported header against an independent from-scratch recomputation, rather than just printing both side by side.

1. **Stale `Num Routes` header** (`main.cpp`, previously trusting `globalSolution.numRoutes` directly). Stage 5's ILS moves can empty a route without compacting the route-slot array; the header then overcounts (e.g. reported 21 when only 20 routes had customers). **Fixed** — `main.cpp` now counts and reports only the routes it actually writes to the output file.
2. **Stage 3 incremental cost drift** (`Stage3_MergeHealing.cpp` / `Stage2_ILS.cpp`'s `stage3_healing_ils_pass`). Accepted healing moves never updated `globalSolution.totalCost` at all, so the incrementally-tracked cost silently diverged from the true cost on every P>1 run (e.g. `Incremental=56744` vs `Scratch=56372`), previously masked only by `main.cpp`'s existing from-scratch recompute-and-overwrite safety net. **Fixed** — each healing thread now accumulates its own delta locally (no shared mutable state while threads run concurrently) and the deltas are summed into `totalCost` once per graph-coloring class, after that class's threads have joined.

Both fixes were verified by rerunning P=1 and P=4 and confirming (a) `Incremental Cost Bookkeeping` now exactly matches `Scratch Computed Final cost` with no warning, and (b) `verifier.py`'s header cross-check passes for both.

## 5. Known issue found, not fixed (flagged for follow-up)

**Capacity constraint violations under sustained optimization pressure.** The controlled experiment in §2.3 (each P=4 thread running 100,000 iterations instead of the default 25,000) produced a solution where **4 of 18 routes exceeded the 100-unit vehicle capacity** (up to 170 units in one route). This was caught by `verifier.py`'s independent capacity check (which recomputes route load from raw demand data), not by the solver's own `routeLoad` bookkeeping — that bookkeeping was internally consistent with the (invalid) route contents, so this is not a load-tracking arithmetic bug. It indicates a real gap in a capacity feasibility check somewhere in the relocate / recreate / local-search move evaluation logic in `Stage2_ILS.cpp` (shared by Stage 2, Stage 3 healing, and Stage 5 polish) that only manifests after enough ILS iterations/moves accumulate.

**This did not appear in either default-configuration run reported in §1 (P=1 or P=4) — both were independently verified feasible.** But it means increasing the iteration budget beyond current defaults (e.g. to close the P=4 cost gap, as previously suggested) is not safe until this is root-caused. Recommended follow-up: bisect which move type (relocate / swap / SWAP\* / 2-opt) admits the violation, most likely by re-running the equal-iteration experiment with capacity assertions enabled after every accepted move.

---

## 6. Caveats on this comparison

- **Wall-clock variance:** P=1 timing varied between 90.7s and 109.8s across otherwise-identical runs on this machine (no code change), presumably from background system load — timing numbers here should be read as "same order of magnitude," not precise to the second. Cost values were bit-identical across repeated runs (seeded RNG, deterministic partitioning), which is the more meaningful reproducibility signal.
- **Compiler flags:** MSVC on this Windows build silently ignores the `-O3 -march=native` flags requested in `CMakeLists.txt` (`cl : command line warning D9002`) — both our solver and this comparison are effectively running at MSVC's default Release optimization level, not the GCC/Clang-style flags the CMake file assumes. This affects both P=1 and P=4 equally (same binary), so it doesn't bias the internal P=1-vs-P=4 comparison, but it does mean the FILO2 comparison isn't necessarily an apples-to-apples compiler-optimization comparison unless FILO2's prebuilt binary was built the same way (unverified — `baselines/filo2/build` was pre-built, not rebuilt as part of this pass).
- **Single instance, single seed:** all numbers are from one 2000-node synthetic instance with uniform demand=1 (capacity constraints are structurally close to vacuous — every feasible route is close to exactly 100 customers). Results may not generalize to instances with heterogeneous demand or different size regimes without re-running this comparison there.
