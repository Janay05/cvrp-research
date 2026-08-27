#pragma once
#include "Types.hpp"
#include <vector>
#include <utility>

struct KDNode {
    NodeId id;
    int axis; // 0 = split on x, 1 = split on y
    int left, right; // indices into a flat KDNode array, -1 if none
};

struct KDTree {
    std::vector<KDNode> nodes;
    int root;

    void build(const Instance& inst);
};

struct NeighborLists {
    int k;
    std::vector<std::vector<NodeId>> nbr; // nbr[i] = k nearest to i, sorted by distance

    // num_threads: each node's kNN query is independent (reads the already-built, immutable
    // KDTree; writes only to its own nbr[i] slot), so this parallelizes over node ranges
    // with no synchronization needed and identical, deterministic output regardless of
    // thread count -- see docs/reports/006_throughput_and_parallelism.md Phase 2.5.
    void build(const Instance& inst, int k_neighbors, int num_threads = 1);

    // Post-pass: makes the candidate relation closer to symmetric (see KDTree.cpp for why
    // the raw kNN query is asymmetric and what this does about it).
    // num_threads: output is identical for any thread count (see KDTree.cpp) -- this was
    // measured as the dominant cost of list construction at scale (137 s of a 180 s Stage 0
    // at Lazio/k=300, versus 3.4 s for the actual kNN queries), so it must not stay serial.
    void symmetrize(const Instance& inst, int k_neighbors, int num_threads = 1);

    // T2-lite (docs/reports/009_plan_beating_filo2.md): reverseIdx[v] lists every (i, j_idx)
    // such that nbr[i][j_idx] == v -- "who has v as a candidate". Purely structural (depends
    // only on nbr, not on solution state), so it's safe to build once per chunk and reuse for
    // that chunk's entire Stage 2 run; used by local_search/invalidate_svc's pair-cache
    // invalidation to find which OTHER nodes' cached entries a touched vertex v affects.
    std::vector<std::vector<std::pair<NodeId,int>>> reverseIdx;
    void build_reverse_index();
};
