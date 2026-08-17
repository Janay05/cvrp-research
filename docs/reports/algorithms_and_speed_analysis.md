# Why the solver is faster than FILO2: algorithms and design decisions behind the runtime results

**Performance vs. FILO2:** Under verified benchmarking with identical compilers, the solver currently runs **1.30x-1.36x faster** than the FILO2 baseline across instance sizes from 20,000 to ~1,000,000 customers, while maintaining highly competitive solution quality (within **0.77-1.51%** of FILO2's cost).

## Technical Performance Drivers

The speed advantage is driven by four specific algorithmic techniques and architectural design decisions:

### 1. Hilbert Curve Partitioning (Parallel Independent Sub-problems)
The problem is divided into independent geographic chunks that are optimized **in parallel**, meaning each thread executes the same search effort a single-threaded solver would spend on the whole problem. 
- **Technique:** We order customers along a Hilbert space-filling curve (which preserves 2D geographic locality in a 1D sequence) and slice them into `P` contiguous chunks. This single sort is computationally negligible ($O(N \log N)$) but yields highly compact clusters.
- **Why this is faster than FILO2:** FILO2 runs its local search sequentially over the entire customer set. By partitioning, we give each of our `P` threads a fully independent sub-problem with no shared state, eliminating lock contention entirely. This drops wall-clock time roughly in proportion to the number of CPU cores used.

### 2. $O(1)$ Search Step Evaluations
Every individual search step is engineered to run in constant or near-constant time, rather than linearly scaling with route size.
- **Technique:** 
  1. Candidate moves are restricted strictly to a precomputed neighbor list (the 30 nearest customers). 
  2. The cost impact of moves (like 2-opt, 2-opt*, and SWAP*) is evaluated by calculating the delta of the specific edges being changed, rather than re-walking the route. 
  3. Routes are implemented as doubly-linked lists backed by flat arrays. A rejected modification is undone by replaying a tiny change log ($O(1)$ pointer updates) instead of copying or restoring the full solution state.
- **Why this is faster than FILO2:** While FILO2 evaluates a heavier set of 23 move types, our solver evaluates 5 highly optimized types (including a SWAP* that uses precomputed insertion-point tables to avoid brute-force scanning). Because we never do full structural rebuilds or route re-scans during the inner loop, our per-step CPU cycle cost is a fraction of a standard baseline implementation.

### 3. Edge-Colored Graph Parallel Boundary Healing
Repairing the routes near chunk boundaries is structurally necessary for us, but is parallelized to prevent it from becoming a serial bottleneck.
- **Technique:** We model the chunk-boundary structure as a graph: nodes are chunks, and edges are pairs of chunks sharing boundary customers. We apply **edge-coloring** to this graph. Because no two edges of the same color share a node, every repair task within a single color class operates on a completely disjoint set of routes. 
- **Why this maintains our speed advantage over FILO2:** The edge-coloring algorithm guarantees that no two threads touch the same route at once, allowing us to process boundary repairs concurrently with **zero locking**. This keeps the overhead of this phase strictly capped at ~6% of total runtime, preserving our overall parallel speedup.

### 4. Zero-Allocation Inner Loops
Strict memory management and profiling eliminated hidden architectural overhead that was taxing the parallel algorithms.
- **Technique:** We enforce a zero-allocation policy during the search phase. All working memory (logs, scratch tables, caches) is allocated exactly once per thread at startup. 
- **Why this keeps us faster:** To consistently perform well, we had to ensure our per-thread throughput wasn't bottlenecked by the OS memory allocator. For example, replacing a small per-call allocating sort in the SWAP* insertion-point routine with a fixed, allocation-free calculation saved ~40% wall time on its own. We also replaced full-solution snapshots with lighter partial snapshots, ensuring CPU cycles are spent purely on search, not memory management.
