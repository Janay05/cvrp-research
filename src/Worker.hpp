#pragma once
#include "Types.hpp"
#include "Barrier.hpp"
#include "Stage0_Partitioning.hpp"
#include "Solution.hpp"
#include "KDTree.hpp"
#include <vector>
#include <random>

struct alignas(64) WorkerContext {
    int workerId;
    std::vector<int> assignedChunks;
    Barrier* stage_barrier;
    
    // shared read-only pointers
    const Instance* instance;
    const NeighborLists* neighborLists;
    const Stage0Result* partitionInfo;
    
    // filled by this worker
    std::vector<Solution> chunkSolutions;
    
    // Isolated per-thread RNG
    std::mt19937 rng;
};

void worker_main(WorkerContext& ctx);
