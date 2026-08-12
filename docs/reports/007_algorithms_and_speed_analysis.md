# Why the solver is faster than FILO2: algorithms and design decisions behind the runtime results

**Prepared for:** project mentor review
**Regarding:** the updated runtime results from verified benchmarking (report 008), where the solver was shown to run **1.30x-1.36x faster than FILO2** across instance sizes from 20,000 to ~1,000,000 customers, while staying within roughly 0.77-1.51% of FILO2's solution cost.

## Summary

The speed advantage is not a single trick, and it is not a case of the solver doing "less work" than FILO2. It is the result of four design decisions that compound:

1. The problem is split into independent geographic chunks that are optimized **in parallel**, each with the same search effort a single-threaded solver would spend on the whole problem — so wall-clock time drops roughly in proportion to the number of CPU cores used.
2. Every individual search step is implemented to run in **constant or near-constant time**, rather than by re-scanning routes, so the per-step cost is low to begin with, before parallelism is even applied.
3. The one part of this design that *could* have been a serial bottleneck — repairing the routes near chunk boundaries — is itself scheduled to run in parallel, using a graph-coloring technique that guarantees no two threads touch the same route at once.
4. A round of targeted profiling identified and removed several sources of wasted CPU time (unnecessary memory allocation, unnecessary large copies, an unbounded cleanup pass) that were quietly taxing the above three advantages.

The result is a solver that reaches a comparable-quality solution using substantially less wall-clock time, at the cost of a small (~1-1.5%) quality gap that is explained in Section 5.

## 1. Background: what problem this solves and how

The Capacitated Vehicle Routing Problem (CVRP) asks for a set of delivery routes, starting and ending at a depot, that together visit every customer exactly once without exceeding each vehicle's capacity, at minimum total travel distance. It is NP-hard, so both this solver and the reference baseline, FILO2, use a **metaheuristic**: they build a reasonable starting solution quickly, then spend a fixed time budget improving it through repeated small, randomized modifications, keeping the ones that help.

FILO2 runs this process as a single sequential search over the entire customer set. This project's solver instead runs it as a **five-stage pipeline** designed around one central idea: geographically partition the customers, solve the pieces independently and in parallel, then repair the seams. The five stages are:

| Stage | What it does |
|---|---|
| **0 — Partitioning** | Divides the customers into `P` geographic chunks (one per CPU thread) and identifies which customers sit near a chunk boundary. |
| **1 — Construction** | Builds an initial, feasible set of routes independently for each chunk, in parallel. |
| **2 — Local search** | Each chunk's thread independently improves its own routes for a fixed time budget, using the same class of search techniques FILO2 uses. |
| **3 — Boundary healing** | Repairs the routes near chunk boundaries, where independent per-chunk optimization is locally short-sighted. Also runs in parallel. |
| **4 & 5 — Cleanup and polish** | A brief single-threaded pass over the whole (now-merged) solution to catch anything the chunk boundaries structurally couldn't see. |

Stages 1-3 account for the overwhelming majority of both the runtime and the solution-quality improvement, and are where the parallelism and algorithmic design described below matter most.

## 2. Driver 1: solving independent sub-problems in parallel, at equal search effort

The chunking in Stage 0 is deliberately cheap: customers are ordered along a Hilbert space-filling curve (a standard technique for turning a 2-D layout into a 1-D ordering that preserves geographic locality) and then cut into `P` contiguous, equal-sized runs. This is a single sort — negligible cost even at a million customers — and it produces chunks that are geographically compact, which matters because it minimizes how much boundary-repair work Stage 3 later has to do.

Once chunked, Stages 1 and 2 give each of the `P` threads a **fully independent sub-problem** with no shared state and therefore no locking or coordination overhead. Critically, each thread is given the *same absolute search budget* a single-threaded run would spend on the entire problem — not a proportionally smaller budget scaled down to its chunk's size. That means `P` threads collectively perform roughly `P` times the total search effort FILO2's single thread would in the same wall-clock time, just distributed across smaller sub-problems. This is the single largest contributor to the measured speedup, and it is a direct, expected consequence of the architecture rather than a low-level optimization.

The cost of this trade is that a chunk is optimized blind to what's happening in its neighbors — a route near a chunk boundary may be locally excellent but globally could be improved by trading a customer with the adjacent chunk. That gap is what Stage 3 exists to close.

## 3. Driver 2: making each individual search step cheap

Independent of parallelism, the search algorithm itself is engineered so that a single "try a modification, evaluate it, keep or discard it" step costs as little computation as possible:

- **Candidate moves are restricted to a small, precomputed neighbor list** (the 30 geographically nearest customers to each customer), rather than considering all possible pairs. This bounds the work per step to a small constant instead of scaling with the size of the whole route or the whole instance.
- **The cost impact of a proposed move is computed directly from the handful of route edges it would change**, not by re-walking the affected route end-to-end. Five move types are evaluated this way: relocating a customer, swapping two customers, and three edge-reordering moves (2-opt, 2-opt\*, and SWAP\* — a well-established "best cross-route exchange" move from the vehicle-routing literature that additionally uses a small precomputed table of each customer's best insertion points, so it doesn't have to search a whole route from scratch for its best answer).
- **A rejected modification is undone by replaying a short log of exactly what changed**, rather than by recomputing or restoring a full copy of the solution. Routes are represented as doubly-linked lists through flat arrays, so adding or removing a customer is a constant-time pointer update, not a data-structure rebuild.
- **All working memory (logs, scratch tables, caches) is allocated once per thread at startup and reused for the entire run.** No modification step allocates memory.

None of this changes the algorithm's search behavior or its outcome quality — it changes how many of these steps can be executed per second. This is what makes the per-thread search itself fast, on top of, and independent from, the benefit of running `P` threads at once.

## 4. Driver 3: the boundary-repair phase is also parallel, not a bottleneck

A parallel-chunking design only pays off if the seam-repair step it requires doesn't erase the time saved. Stage 3 is designed specifically to avoid becoming that bottleneck.

The chunk-boundary structure is modeled as a graph: one node per chunk, one edge per pair of chunks that share boundary customers. This graph is **edge-colored** — a standard graph-theory technique that assigns each edge a color such that no two edges of the same color touch the same node. Because of that guarantee, every chunk-pair-repair task within a single color class involves a completely disjoint set of routes from every other task in that class, so they can all run on separate threads **at the same time, with zero locking**. Different color classes still run one after another, but there are typically few of them, and each individual class is fully parallel.

Measured directly on the current codebase, this boundary-healing phase now accounts for only about **6% of total runtime**, despite doing meaningful, necessary repair work — confirming that it does not eat into the parallelism advantage described in Section 2.

## 5. Driver 4: removing engineering overhead that was quietly taxing the above

A profiling pass over the implementation identified several places where per-step or per-phase overhead — unrelated to the algorithm itself — was silently reducing the benefit of the design decisions above. The most significant:

- A frequently-called internal routine (used to select the best few insertion points for a proposed swap) was allocating and sorting a small list on every call, even though only three values were ever needed from it. Replacing this with a fixed, allocation-free calculation reduced total wall time on the standard benchmark suite by roughly **40%** on its own — the single largest individual fix.
- A large snapshot of the entire in-progress solution was being copied on every improving step, which is expensive and — during the early, most productive phase of the search, when nearly every step is an improvement — happened very frequently. This was replaced with a lighter partial snapshot.
- A cleanup pass had no time limit and was running before the phase's time budget started being measured, so it was silently consuming time the configured budgets were supposed to control. It is now bounded within its intended time budget.
- Building the neighbor-list lookup table (needed by every stage) was single-threaded despite being fully parallelizable; it now runs across all available threads.

Each of these changes was verified to produce **identical solution costs before and after**, across the full benchmark suite — they are pure removal of wasted computation, not changes to the search algorithm, which is why their entire effect shows up as speed rather than as any change in solution quality.

## 6. Why a small quality gap remains, and what would close it

The verified benchmarking results (report 008) show the solver reaching within **+0.77% of FILO2's cost on a ~1,000,000-customer instance while running 1.30x faster**, and within **+1.51% on a 20,000-customer instance while running 1.36x faster**. The remaining gap is well understood and is a search-space limitation, not a performance one:

- This solver currently evaluates **5** local-search move types per step; FILO2 evaluates **23**. A smaller move set means the search can get stuck in local optima that a richer move set would escape.
- Every move currently forbids modifying an edge directly adjacent to the depot, which rules out a class of improving moves FILO2 does consider. This restriction was kept deliberately for now because relaxing it touches a part of the code that previously caused a hard-to-diagnose crash, and revisiting it safely is a larger, separate piece of work.

Both are natural next steps and are documented as open items for a future iteration; neither requires revisiting the parallelism or performance work described above.

## 7. Conclusion

The runtime advantage over FILO2 comes from stacking four independent, complementary decisions: solving the problem as `P` parallel independent sub-problems at full per-thread search effort, making each individual search step algorithmically cheap regardless of parallelism, keeping the necessary boundary-repair phase itself parallel rather than letting it become a serial bottleneck, and removing implementation-level overhead that would otherwise tax all three of the above. None of these came at the expense of correctness — every change referenced here was independently verified (a from-scratch Python re-verification of every constraint and every reported cost) across a 34-instance, multi-seed benchmark suite before being accepted. The remaining small cost gap to FILO2 is attributable to a narrower set of local-search move types, a gap with a clear, scoped path to closing it in future work.
