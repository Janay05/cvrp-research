#pragma once
#include "Types.hpp"
#include "Solution.hpp"
#include "Stage0_Partitioning.hpp"
#include <vector>
#include <random>

// cw_neighborLists: wider candidate list used ONLY by the Clarke & Wright path
// (--construction cw). CW's merge quality depends directly on how many (i,j) savings pairs
// it can consider -- with the k=30 granular list it leaves far too many routes unmerged
// (846 routes / 23,534,274 at VDA, versus 810 / 22,231,600 for FILO2, which draws 100
// candidates from a 1500-wide list). The MST path continues to use neighborLists.
Solution stage1_construct(int chunkId, const Instance& inst, const Stage0Result& partitionInfo,
                          const NeighborLists& neighborLists, std::mt19937& rng,
                          const NeighborLists* cw_neighborLists = nullptr);
