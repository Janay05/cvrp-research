#pragma once
#include "Types.hpp"
#include <vector>

struct DoUndoEntry {
    enum Type { REMOVE, INSERT } type;
    NodeId customer;
    NodeId prevPred, prevSucc; // state before the change (for undo)
    NodeId newPred, newSucc; // state after the change (for do)
    int prevRoute, newRoute;
    Cost costDelta; // signed change to totalCost from this single edit
    // costToPred snapshot for undo (T1, see Solution.hpp): the caller (remove_customer/
    // insert_customer) already computes these dist() values while building costDelta, so
    // stashing them here lets apply_undo_list restore costToPred on rollback without any
    // extra dist() calls -- the whole point of caching it. Meaning depends on type:
    //  - type==INSERT (an undo entry for an original REMOVE): undoCostC/undoCostS are the
    //    customer's and its old successor's costToPred as they were *before* the removal --
    //    exactly what re-inserting on undo must restore.
    //  - type==REMOVE (an undo entry for an original INSERT): undoCostS is the successor's
    //    costToPred as it was *before* the insertion (i.e. dist(p, s)) -- what removing on
    //    undo must restore. undoCostC is unused (the customer is leaving the route).
    Cost undoCostC = 0, undoCostS = 0;
};

struct SVCCache {
    static const int CAPACITY = 50;
    NodeId buffer[CAPACITY];
    std::vector<bool> inCache;
    int head = 0, count = 0;

    void init(int max_chunk_size) {
        inCache.assign(max_chunk_size + 1, false);
    }

    void clear() {
        for (int i = 0; i < count; i++) {
            inCache[buffer[(head + i) % CAPACITY]] = false;
        }
        head = 0; count = 0;
    }

    void insert(NodeId v) {
        if (v >= inCache.size()) return; // Bounds check
        if (inCache[v]) return;
        if (count == CAPACITY) {
            NodeId evicted = buffer[head];
            inCache[evicted] = false;
            head = (head + 1) % CAPACITY;
            count--;
        }
        int pos = (head + count) % CAPACITY;
        buffer[pos] = v;
        inCache[v] = true;
        count++;
    }

    bool contains(NodeId v) const { 
        if (v >= inCache.size()) return false;
        return inCache[v]; 
    }
    
    NodeId pop() {
        if (count == 0) return 0;
        NodeId v = buffer[head];
        inCache[v] = false;
        head = (head + 1) % CAPACITY;
        count--;
        return v;
    }
};

struct Top3Insertions {
    Cost delta[3];
    NodeId pos_pred[3];
    NodeId pos_succ[3];
    int count = 0;
};

// T2-lite (docs/reports/009_plan_beating_filo2.md): per-(i,j)-candidate-pair
// best-of-9-operators delta cache for local_search, scoped to a single
// local_search-to-convergence call (see ThreadArena::pairCacheGen below). gen==0
// means "never computed"; a real computed entry's gen is set to whatever
// pairCacheGen was at compute time, so it reads as valid only while that
// generation is still current, and as invalid (without touching the entry
// itself) the instant pairCacheGen is bumped for the next call. Explicit
// invalidation (a touched vertex) sets gen to -1, a sentinel distinct from any
// real generation value (which starts at 1 and only increments).
//
// loadRi/loadRj: routeLoad[routeOf[i]] and routeLoad[routeOf[j]] at compute time.
// Several operators' capacity feasibility depends on OTHER routes' total load (e.g.
// eval_relocate rejects if routeLoad[r_j]+demand[i]>Q), which can change because a
// DIFFERENT customer was inserted into/removed from r_i or r_j elsewhere in the
// route -- a move that never touches i or j directly, so the vertex-based
// invalidation in invalidate_pair_cache_one() wouldn't catch it. Re-checking these
// two integers on every cache hit (no dist() calls, routeLoad is always kept exactly
// current) closes that gap cheaply instead of requiring a much wider, route-scoped
// invalidation footprint.
struct PairCacheEntry {
    Cost delta = 0;
    Cost loadRi = 0, loadRj = 0;
    int8_t op = -1;   // same encoding as bestOp in local_search (0..8); meaningless when invalid
    int32_t gen = 0;
};

struct alignas(64) ThreadArena {
    std::vector<DoUndoEntry> doList; 
    std::vector<DoUndoEntry> undoList; 
    int doCount = 0, undoCount = 0; 
    Cost pendingDelta = 0;

    std::vector<NodeId> scratchTop3Pos; 

    // Zero-allocation buffers requested
    std::vector<NodeId> removed_customers;
    int removed_count = 0;

    std::vector<Top3Insertions> top3_i_to_V;
    std::vector<int> route_visited_iter;
    std::vector<Top3Insertions> top3_j_to_U;
    std::vector<int> node_visited_iter;

    // apply_undo_list's dedup scratch space: a "last-touched generation" marker per route
    // plus a compact list of routes touched this call, so the set of routes needing a
    // update_route_info() rescan on rollback isn't capped at a fixed small count (see
    // docs/reports/005_cost_optimization.md Phase 1.2 -- the previous int[10] cap silently
    // dropped routes past the 10th, leaving their routePosition/cumLoad stale).
    std::vector<int> route_modified_gen;
    std::vector<int> modified_routes_list;
    int modified_routes_gen = 0;

    // T2-lite pair cache (see PairCacheEntry above). Flat array indexed
    // nodeId*pairCacheKMax + j_idx; only stage2_ils's main SA loop populates/reads this
    // (passed as nullptr everywhere else, which local_search/invalidate_svc/apply_*
    // treat as "caching disabled", their exact pre-T2 behavior). pairCacheGen is bumped
    // once per SA iteration by stage2_ils, right before calling local_search -- an O(1)
    // operation that invalidates every entry from the previous iteration for free.
    std::vector<PairCacheEntry> pairCache;
    int32_t pairCacheGen = 0;
    int pairCacheKMax = 0;

    // max_routes: upper bound on route ids this arena will ever be indexed by (for
    // top3_i_to_V/route_visited_iter/route_modified_gen/modified_routes_list, all
    // route-indexed). Defaults to max_chunk_size + 100, correct for Stage 2 (Worker.cpp)
    // where a chunk's own route count is naturally far below its customer count. Stage 3
    // must pass the SAME bound Stage3_MergeHealing.cpp uses for globalSolution's
    // routeHead/Tail/Load (currently 2*inst.n + 10000, see the comment there) -- sizing
    // these by max_chunk_size (= inst.n) alone was a latent out-of-bounds read/write once
    // report 005's fix let numRoutes grow past inst.n+100 during a long Stage 3 run (see
    // docs/reports/006_throughput_and_parallelism.md Phase 4.1). Stage 5 shares
    // globalSolution post-Stage-3 and needs the same bound for the same reason.
    // k_max=0 (default) skips allocating pairCache entirely -- correct as long as T2-lite
    // stays disabled (kEnablePairCache=false in stage2_ils, Stage2_ILS.cpp), which it
    // currently is everywhere. This buffer is unlike doList/undoList (which are capped at
    // min(2,000,000, ...) regardless of instance size): it scales directly with
    // max_chunk_size, and EVERY call site (Worker.cpp, main.cpp's globalArena, Stage3's
    // arena_pool) passes the FULL instance n here, not a per-chunk share -- so at Lazio scale
    // (n~1,000,000, k=30, 32-byte PairCacheEntry) an unconditional allocation costs ~960MB
    // PER ARENA. With P worker arenas plus Stage 3's arena_pool (one arena per color-class
    // slot), that was measured contributing multiple GB of pure waste for a feature that's
    // switched off. If T2-lite is ever re-enabled, the caller that flips
    // kEnablePairCache=true must also pass a real k_max here (e.g. 30, matching
    // neighborLists.build's k in main.cpp) -- the two are coupled and there's no compile-time
    // check tying them together.
    void reserve_fixed_capacity(int max_chunk_size, int max_routes = -1, int k_max = 0) {
        if (max_routes < 0) max_routes = max_chunk_size + 100;

        pairCacheKMax = k_max;
        if (k_max > 0) {
            pairCache.assign((size_t)(max_chunk_size + 1) * k_max, PairCacheEntry{});
        } else {
            pairCache.clear();
            pairCache.shrink_to_fit();
        }

        // doList/undoList just log the sequence of edit operations within a single SA
        // iteration (ruin + recreate + local_search cascade) -- that cascade length isn't
        // proportional to instance size, so this must be capped independent of
        // max_chunk_size. A later change raised the cap to min(10_000_000, max_chunk_size*100)
        // to give the new multi-string ruin (up to 40 strings/iteration) more headroom, but
        // 10,000,000 entries is 400MB per list (800MB/arena) -- at Lazio scale (P=16), Stage
        // 3's arena_pool pre-allocates one such arena per thread in its largest color class,
        // easily exceeding this machine's ~7.6GB WSL memory budget and crashing the whole VM
        // (not a graceful OOM) rather than the process. Even 40 strings x ~15 removals/string
        // (ln(n) walk length) plus recreate's reinsertions is only ~1,200 entries; local_search's
        // own cascade adds at most a few thousand more per iteration (route lengths are capacity-
        // bounded, not instance-size-bounded). 2,000,000 is 4x the original proven-safe 500,000
        // ceiling -- ample headroom for the new multi-string ruin -- while cutting worst-case
        // memory by 5x from the 10,000,000 figure.
        doList.resize(std::min(2000000, std::max(max_chunk_size * 50, 100000)));
        undoList.resize(std::min(2000000, std::max(max_chunk_size * 50, 100000)));
        removed_customers.resize(max_chunk_size + 100);
        scratchTop3Pos.reserve(3);

        top3_i_to_V.resize(max_routes);          // route-indexed
        route_visited_iter.resize(max_routes, 0); // route-indexed
        top3_j_to_U.resize(max_chunk_size + 100); // node-indexed
        node_visited_iter.resize(max_chunk_size + 100, 0); // node-indexed

        route_modified_gen.resize(max_routes, 0);
        modified_routes_list.resize(max_routes);
    }
};
