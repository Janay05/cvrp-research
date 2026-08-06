# Report 003 — Scale testing: 50 to 1,000,000 nodes

**Follows:** [002_capacity_fix_and_rebalance.md](002_capacity_fix_and_rebalance.md). All prior reports tested exclusively on a synthetic, solver-generated 2000-node instance. This report tests the pipeline on real CVRPLIB-format instances at increasing scale, since that's the actual target regime motivating the parallel-chunked architecture.

## 0. Prerequisite: the pipeline could not read external files at all

Before this pass, `main.cpp` only ever generated its own synthetic N=2000 instance in memory — there was no code path to load an arbitrary `.vrp` file. This had to be built first:

- **`VrpParser.hpp`/`.cpp`** (new): tolerant CVRPLIB-format reader (`NAME`/`DIMENSION`/`CAPACITY`/`NODE_COORD_SECTION`/`DEMAND_SECTION`/`DEPOT_SECTION`), handling the header spacing variance seen across sources (our own generated files vs. the Accorsi & Vigo I-dataset's `KEY : \tvalue\t` style vs. classic CMT/Golden's `KEY : value`). Remaps whatever file node id is listed as the depot to internal index 0.
- **`main.cpp`**: new `-f <path>` flag to load a file instead of generating the synthetic instance (default behavior unchanged when `-f` is omitted, so existing scripts/benchmarks aren't affected). Also added `--iters-per-node <k>` (scales the per-thread iteration budget introduced in report 002, default 50) and `--max-iterations <n>` (an absolute per-thread override, bypassing the `inst.n * k` formula entirely — needed once instance sizes made that formula impractical, see §3).
- **`verifier.py`**: now takes the `.vrp` and solution paths as CLI args instead of being hardcoded to `test_2000.vrp`.

**A real memory-scaling bug surfaced immediately once large instances became possible**, before any test even ran: `ThreadArena::reserve_fixed_capacity()` (`ThreadArena.hpp`) sized its `doList`/`undoList` buffers as `max_chunk_size * 50`. At N=2000 that's a harmless 100,000 entries. Passed the instance's full N (needed for the arena's *other*, node-indexed arrays — see the code comment added at the fix site for why that part is correct), at N=1,000,000 this would have allocated `50,000,000` entries (~1.8GB) *per array, per thread* — tens of GB across a realistic thread count, an near-certain crash or severe thrashing. Fixed by capping this specific pair of buffers at 500,000 entries regardless of `max_chunk_size` — they only need to bound a single ruin+recreate+local_search cascade's operation count, which isn't proportional to instance size at all; the original formula conflated "how big is the instance" with "how long can one iteration's edit sequence get."

## 1. Test plan and correctness ladder

Rather than attempt the 1M-node instance directly, testing ramped up in three steps so a failure at any step would be diagnosable rather than a single opaque crash/hang on the full target:

| Step | Instance | Customers | Purpose |
|---|---|---|---|
| 1 | CMT1 (classic CVRPLIB) | 50 | Correctness: does the new parser + full pipeline produce a feasible, near-optimal solution against a **known** best solution? |
| 2 | Valle-D-Aosta (Accorsi & Vigo I-dataset, same family FILO2 ships) | 20,000 | Mid-scale: does timing/memory hold up an order of magnitude below target, and how does the current default config compare to FILO2 here? |
| 3 | Lazio (same dataset, largest instance) | 999,999 | Target scale. |

## 2. Step 1 — CMT1 (50 customers), P=1

| Metric | Value |
|---|---|
| Cost | 525 |
| Known best solution (BKS) | 524.611 |
| Gap | 0.07% (rounding-level; our `dist()` and the reference both round each edge to the nearest integer) |
| Routes | 5 (matches BKS route count) |
| Time | 141 ms |
| Feasibility | ✅ verified independently |

Essentially optimal. This validates the new file parser and the full pipeline end-to-end against ground truth before trusting it on anything larger.

## 3. Step 2 — Valle-D-Aosta (20,000 customers)

First attempt used the report-002 default config unmodified (P=4, `inst.n * 50` = 1,000,000 iterations *per thread*, regardless of P). This is where the first scale-sensitivity issue showed up:

| Config | Time | Cost | Notes |
|---|---|---|---|
| FILO2 (100,000 iterations, its default) | 237 s | 21,732,499 | ✅ verified feasible |
| Ours, P=4, default (1,000,000 iter/thread) | 460 s | 22,069,827 | ✅ verified feasible, but **1.9x slower than FILO2** |
| Ours, P=4, `--iters-per-node 5` (100,000 iter/thread) | 40.5 s | 22,135,708 | ✅ verified feasible, **5.9x faster than FILO2**, cost only 1.86% worse |

The report-002 rebalance (give every thread the same *absolute* iteration count P=1 would use on the whole graph) was tuned and validated at N=2000. It does not automatically generalize: at N=20,000, "the same absolute count P=1 would use" is 1,000,000 iterations/thread — a budget that made sense when the whole point was recovering iterations lost to chunking at a small scale, but becomes needlessly large once the per-chunk subproblem itself is already sizeable (5,000 nodes/chunk at P=4). The 100,000-iteration config (still 4x the default's *old* pre-002 formula, just far short of the current default) gets within 2% of FILO2's quality in a fraction of the time. **The iteration-budget formula needs to be scale-aware, not a single constant tuned at one instance size** — flagged here, not fixed in this pass (see §5).

One more observation from this step: per-thread completion times were noticeably uneven under the default config (665s config: workers finished at 284s, 299s, 427s, 459s — a ~1.6x spread for nominally equal-sized Hilbert-partitioned chunks doing identical iteration counts). Worth investigating whether this is chunk-content-dependent (harder local optima needing longer local-search cascades) or something else, since it affects how well wall-clock time is predictable at scale.

## 4. Step 3 — Lazio (999,999 customers)

Given §3's finding, the default `inst.n * 50` formula was not attempted at this scale (would be 50,000,000 iterations/thread — clearly impractical). Used the new `--max-iterations` absolute override instead, at P=16 (chunks of ~62,500 nodes), with a deliberately conservative 10,000 iterations/thread to get a first real data point without an open-ended runtime risk.

| Config | Time | Cost | Iterations/thread |
|---|---|---|---|
| FILO2 (100,000 iterations, its default) | 641 s (~10.7 min) | 3,158,419,623 | 100,000 |
| Ours, P=16, `--max-iterations 10000` | 264 s (~4.4 min) | 3,194,025,414 | 10,000 |

**Both independently verified feasible** (999,999/999,999 customers routed exactly once, no capacity violations, for both solvers).

Ours: **2.4x faster than FILO2, using 10x fewer iterations, and within 1.1% of FILO2's cost.** This is the strongest result in this report — at the scale the architecture was actually motivated by, chunking's per-iteration cost advantage (smaller, cheaper subproblems, per report 001) compounds enough to win on wall-clock decisively even against a well-optimized single-threaded baseline, while giving up very little quality.

Stage-wise breakdown for this run:

| Stage | Time | % of total |
|---|---|---|
| Stage 0 — Partitioning & k-NN setup (1M points) | 14.9 s | 5.6% |
| Stage 1 & 2 — Parallel construction + ILS (10,000 iter/thread) | 128.9 s | 48.8% |
| Stage 3 — Parallel boundary healing | 85.1 s | 32.2% |
| Stage 4 & 5 — Cleanup + full-graph serial polish | 35.5 s | 13.4% |

Notably, Stage 3 (healing) is a much larger share of total time here (32%) than at N=2000 (5% in report 002) or even N=20,000 — with P=16 there are far more chunk-pair boundaries to heal, and this hasn't been tuned for this regime at all.

## 5. What this means and what's next

**Correctness holds across four orders of magnitude** (50 to ~1M customers) — every configuration tested produced a verified-feasible solution, and quality is competitive with FILO2 throughout (worst case +1.9% at 20k with a since-identified inefficient config; best case +1.1% at 1M while being 2.4x faster).

**The one thing that doesn't yet hold across scales is the iteration-budget formula.** Report 002's fix (`inst.n * 50` per thread) was correctly motivated and validated at N=2000, but §3 shows it stops being the right choice well before 20,000 nodes, and would be unusable unmodified at 1,000,000. This isn't a correctness bug — every config tested was feasible — it's a tuning gap. Recommended next steps, roughly in order:

1. **Make the iteration budget scale-aware** — e.g. cap absolute iterations per thread at some ceiling independent of N once N passes a threshold, or switch to a time-budget-based stopping criterion (run each stage for a wall-clock budget rather than a fixed iteration count) so the same `-p`/config choices behave sensibly whether N is 2,000 or 2,000,000 without hand-picking `--max-iterations` per instance.
2. **Investigate the per-thread timing imbalance** found in §3 (up to 1.6x spread on equal-sized chunks) — could change how chunk sizing or work distribution should happen.
3. **Tune Stage 3 healing for high-P regimes** — it's already a known lever from report 002, but §4 shows it becomes proportionally much more expensive as chunk count grows, which report 002 didn't anticipate since it only tested P=4.
4. **Re-run Lazio at a larger iteration budget** once (1) exists, to see how much of the remaining 1.1% gap to FILO2 closes with a properly-scaled search budget rather than the conservative 10,000 used here as a first probe.

## 6. Caveats

- Single run per configuration (no repeated-run variance measurement at these larger scales, unlike the repeated small-scale runs in reports 001/002) — timing numbers, especially the Valle-D-Aosta worker-imbalance observation, should be treated as a first data point, not a fully characterized distribution.
- FILO2 was run via its prebuilt binary (`baselines/filo2/build/filo2.exe`) without rebuilding — same caveat as prior reports about unverified compiler-flag parity.
- Lazio was tested with a manually-chosen, conservative iteration budget rather than any principled scale-aware formula (since one doesn't exist yet, per §5.1) — the reported numbers are a real, valid data point, but not necessarily this architecture's best achievable result at this scale.
