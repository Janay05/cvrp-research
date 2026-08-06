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
    
    std::vector<Top3Insertions> top3_cache;
    
    std::vector<Top3Insertions> top3_i_to_V;
    std::vector<int> route_visited_iter;
    std::vector<Top3Insertions> top3_j_to_U;
    std::vector<int> node_visited_iter;

    void reserve_fixed_capacity(int max_chunk_size) {
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
        top3_cache.resize(max_chunk_size + 1);
        
        top3_i_to_V.resize(max_chunk_size + 100);
        route_visited_iter.resize(max_chunk_size + 100, 0);
        top3_j_to_U.resize(max_chunk_size + 100);
        node_visited_iter.resize(max_chunk_size + 100, 0);
    }
};
