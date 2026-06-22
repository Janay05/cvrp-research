#pragma once

#include "DataStructures.hpp"
#include <vector>

namespace cvrp::partitioning {

    class MacroPartitioner {
    public:
        // Configurable angular sweep size in degrees (e.g., 40 to 45 degrees)
        // For 10^6 nodes, 45 degrees generates exactly 8 macro-wedges.
        explicit MacroPartitioner(double sweep_angle_degrees = 45.0);

        // Takes the global nodes vector, computes polar angles relative to the depot,
        // sorts the array in-place by angle O(N log N), and partitions it into 
        // disjoint, contiguous spans.
        // Depot is assumed to be provided separately or at a known index (usually not included in spans).
        std::vector<NodeSpan> partition(const Node& depot, std::vector<Node>& global_nodes);

    private:
        double sweep_angle_degrees_;
        
        void compute_polar_angles(const Node& depot, std::vector<Node>& nodes);
        void sort_by_angle(std::vector<Node>& nodes);
    };

} // namespace cvrp::partitioning
