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
    // costToPred[v] = dist(pred[v], v) -- maintained incrementally by remove_customer/
    // insert_customer/apply_undo_list (Stage2_ILS.cpp) so operators evaluating "the edge
    // currently entering v" don't need to call dist() to look it up. Only valid while v is
    // in a route (routeOf[v] != -1); undefined/stale otherwise, same convention as pred/succ
    // themselves being 0 for a removed customer. See docs/reports/009_plan_beating_filo2.md T1.
    std::vector<Cost> costToPred;
    int numRoutes = 0;
    Cost totalCost = 0; // maintained incrementally
};
