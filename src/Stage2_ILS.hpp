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

void stage2_ils(Solution& sol, ThreadArena& arena, const Instance& inst, 
                const NeighborLists& granular_lists, int max_iter, std::mt19937& rng);
                
void stage5_serial_polish(Solution& globalSolution, ThreadArena& arena, const Instance& inst, const NeighborLists& neighborLists, double avgArcCostEstimate);

void update_route_info(Solution& sol, int route, const Instance& inst);
