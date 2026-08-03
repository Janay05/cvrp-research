#pragma once
#include "Types.hpp"
#include "Solution.hpp"
#include "Stage0_Partitioning.hpp"
#include <vector>
#include <random>

Solution stage1_construct(int chunkId, const Instance& inst, const Stage0Result& partitionInfo, const NeighborLists& neighborLists, std::mt19937& rng);
