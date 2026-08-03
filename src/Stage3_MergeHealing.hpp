#pragma once
#include "Types.hpp"
#include "Solution.hpp"
#include "Stage0_Partitioning.hpp"
#include "ThreadArena.hpp"
#include "KDTree.hpp"
#include <vector>

void merge_all_chunks_into_global(Solution& globalSolution, 
                                  const std::vector<Solution>& chunkSolutions, 
                                  const Stage0Result& partitionInfo,
                                  const Instance& inst);

void run_stage3_healing(Solution& globalSolution, 
                        const Instance& inst, 
                        const Stage0Result& partitionInfo,
                        const NeighborLists& neighborLists);
