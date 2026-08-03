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

    void build(const Instance& inst, int k_neighbors);
};
