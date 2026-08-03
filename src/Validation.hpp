#pragma once
#include "Types.hpp"
#include "Solution.hpp"

void check_feasibility(const Solution& sol, const Instance& inst);
void check_cost_consistency(const Solution& sol, const Instance& inst);
