#pragma once
#include "Types.hpp"
#include <vector>

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
    void symmetrize(const Instance& inst, int k_neighbors);
};
