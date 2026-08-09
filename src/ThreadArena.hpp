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
    void reserve_fixed_capacity(int max_chunk_size, int max_routes = -1) {
        if (max_routes < 0) max_routes = max_chunk_size + 100;

        // doList/undoList just log the sequence of edit operations within a single SA
        // iteration (ruin + recreate + local_search cascade) -- that cascade length isn't
        // proportional to instance size, so this must be capped independent of
        // max_chunk_size. Uncapped (max_chunk_size*50) was fine at N=2000 (100,000 entries)
        // but allocates tens of millions of entries -- gigabytes per thread -- once N grows
        // into the hundreds of thousands to millions.
        int64_t doUndoCapacity = std::min((int64_t)std::max(max_chunk_size * 50, 100000), (int64_t)500000);
        doList.resize(doUndoCapacity);
        undoList.resize(doUndoCapacity);
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
