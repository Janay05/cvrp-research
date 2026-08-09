#pragma once
#include "Types.hpp"
#include "KDTree.hpp"
#include <vector>
#include <set>

struct Stage0Result {
    std::vector<int> chunkOf; // globalId -> chunk index
    std::vector<int> localId; // globalId -> localId
    std::vector<std::vector<int>> globalId; // globalId[chunk][localId] -> globalId
    
    std::vector<bool> isBoundary; // globalId -> bool
    std::vector<int> boundaryList; // list of globalIds where isBoundary is true
    std::vector<std::vector<int>> boundaryChunkPair; // globalId -> list of adjacent chunk indices
    
    int numChunks;
    std::vector<int> chunkSize; // chunk index -> size

    // Median of each node's mean-kNN-edge-length, i.e. a robust instance-scaled estimate of
    // a "typical" arc cost. Used to set the simulated annealing T0/Tf in Stage2_ILS.cpp
    // instead of a hardcoded constant -- see docs/reports/005_cost_optimization.md Phase 1.3.
    double medianKnnEdgeLen = 100.0;
};

Stage0Result run_stage0(const Instance& inst, const NeighborLists& neighborLists, int num_chunks);
