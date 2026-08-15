#pragma once
#include "Types.hpp"
#include "Solution.hpp"
#include "ThreadArena.hpp"
#include "KDTree.hpp"

void stage4_route_cleanup(Solution& globalSolution, const Instance& inst, const NeighborLists& neighborLists);

void stage5_serial_polish(Solution& globalSolution, ThreadArena& arena, const Instance& inst, const NeighborLists& neighborLists, double avgArcCostEstimate);
void stage5_fleet_minimization(Solution& sol, ThreadArena& arena, const Instance& inst, const NeighborLists& granular_lists, double avgArcCostEstimate);
