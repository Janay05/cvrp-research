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
};

Stage0Result run_stage0(const Instance& inst, const NeighborLists& neighborLists, int num_chunks);
