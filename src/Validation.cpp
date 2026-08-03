#include "Validation.hpp"
#include <iostream>
#include <stdexcept>
#include <vector>

void check_feasibility(const Solution& sol, const Instance& inst) {
    int n = inst.n;
    std::vector<int> visits(n + 1, 0);
    
    for (int r = 0; r < sol.numRoutes; ++r) {
        NodeId curr = sol.routeHead[r];
        Cost load = 0;
        
        while (curr != 0) {
            if (curr > n) throw std::runtime_error("Invalid node id");
            visits[curr]++;
            load += inst.demand[curr];
            curr = sol.succ[curr];
        }
        
        if (load > inst.Q) {
            throw std::runtime_error("Route capacity exceeded");
        }
    }
    
    for (int i = 1; i <= n; ++i) {
        if (visits[i] != 1) {
            std::cout << "Node " << i << " visited " << visits[i] << " times.\n";
            throw std::runtime_error("Customer not visited exactly once");
        }
    }
}

void check_cost_consistency(const Solution& sol, const Instance& inst) {
    Cost computed_cost = 0;
    
    for (int r = 0; r < sol.numRoutes; ++r) {
        NodeId curr = sol.routeHead[r];
        if (curr == 0) continue;
        
        computed_cost += dist(inst, 0, curr);
        while (sol.succ[curr] != 0) {
            computed_cost += dist(inst, curr, sol.succ[curr]);
            curr = sol.succ[curr];
        }
        computed_cost += dist(inst, curr, 0);
    }
    
    if (computed_cost != sol.totalCost) {
        std::cout << "Computed: " << computed_cost << " Incremental: " << sol.totalCost << "\n";
        throw std::runtime_error("Cost consistency failed");
    }
}
