#pragma once
#include "Types.hpp"
#include "Solution.hpp"
#include "ThreadArena.hpp"
#include "Stage0_Partitioning.hpp"
#include "KDTree.hpp"
#include <random>

Solution stage2_ils(Solution sol, ThreadArena& arena, SVCCache& cache,
                    const Instance& inst, const Stage0Result& partitionInfo,
                    const NeighborLists& neighborLists, int chunkId, std::mt19937& rng,
                    int* out_iterations_completed = nullptr);
