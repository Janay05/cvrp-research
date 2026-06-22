#pragma once

#include "DataStructures.hpp"
#include <vector>

namespace cvrp::partitioning {

    class BoundaryOptimizer {
    public:
        // Threshold for route capacity utilization to identify "stragglers" (e.g., < 80%)
        explicit BoundaryOptimizer(double min_utilization_threshold = 0.80);

        // Stage 3: Resolving Stragglers across boundary lines.
        // Scans the global set of generated routes from independent HGS runs.
        // Identifies under-utilized routes near macro/micro boundaries, 
        // extracts them, merges their nodes into a localized sub-problem, 
        // and executes a fast local search (SWAP/RELOCATE) to pack them efficiently.
        void optimize_boundaries(
            const Node& depot,
            const std::vector<Node>& global_nodes,
            std::vector<Route>& global_routes,
            double vehicle_capacity
        );

    private:
        double min_utilization_threshold_;

        // Identifies indices of routes in global_routes that are under-utilized 
        // and geographically near partitioning boundaries.
        std::vector<int> identify_straggler_routes(
            const std::vector<Node>& global_nodes,
            const std::vector<Route>& global_routes
        );

        // Extracts the nodes belonging to the identified straggler routes,
        // runs SWAP/RELOCATE on this merged sub-problem, and updates global_routes.
        void local_search_repack(
            const Node& depot,
            const std::vector<Node>& global_nodes,
            std::vector<Route>& global_routes,
            const std::vector<int>& straggler_route_indices,
            double vehicle_capacity
        );
    };

} // namespace cvrp::partitioning
