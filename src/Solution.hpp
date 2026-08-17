#pragma once
#include "Types.hpp"
#include <vector>

struct Solution {
    std::vector<NodeId> pred, succ; // predecessor/successor customer id in its route
    std::vector<int> routeOf; // which route index this customer currently belongs to
    std::vector<Cost> loadUpTo; // optional cached cumulative load
    std::vector<std::vector<NodeId>> routeStart; // not used directly; routes are traversed via pred/succ from depot sentinels
    std::vector<NodeId> routeHead; // routeHead[r] = first customer in route r
    std::vector<NodeId> routeTail; // routeTail[r] = last customer in route r
    std::vector<Cost> routeLoad; // current total demand of route r
    std::vector<int> routePosition; // dense-integer position of customer in its route
    std::vector<Cost> cumLoad; // cumulative load strictly before the customer
    int numRoutes = 0;
    Cost totalCost = 0; // maintained incrementally
};
