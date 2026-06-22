#pragma once

#include "DataStructures.hpp"
#include <vector>

namespace cvrp::partitioning {

    // Abstract interface for Stage 2: Micro-Clustering
    // Breaks down a ~125,000 node macro-wedge into HGS-feasible sub-problems (~1000-2000 nodes).
    class MicroClusterer {
    public:
        virtual ~MicroClusterer() = default;

        // Partition the given macro-wedge (NodeSpan) into smaller, capacity-feasible SubProblems.
        // Depot is passed explicitly to compute true distances.
        virtual std::vector<SubProblem> cluster(
            const Node& depot,
            NodeSpan wedge_nodes, 
            double vehicle_capacity,
            int target_subproblem_size
        ) = 0;
    };

    // 1. Concentric Sweep Clusterer
    // Buckets nodes into concentric distance bands from the depot, 
    // then runs a capacity-aware polar sweep within each band.
    class ConcentricSweepClusterer : public MicroClusterer {
    public:
        // Configurable number of distance bands, or auto-computed if required.
        explicit ConcentricSweepClusterer(int num_bands = 10);

        std::vector<SubProblem> cluster(
            const Node& depot,
            NodeSpan wedge_nodes, 
            double vehicle_capacity,
            int target_subproblem_size
        ) override;
        
    private:
        int num_bands_;
    };

    // 2. MST Clusterer
    // Computes a Minimum Spanning Tree within the macro-wedge and snips edges via DFS 
    // when accumulated demand exceeds vehicle capacity.
    class MSTClusterer : public MicroClusterer {
    public:
        std::vector<SubProblem> cluster(
            const Node& depot,
            NodeSpan wedge_nodes, 
            double vehicle_capacity,
            int target_subproblem_size
        ) override;
    };

    // 3. Clarke Wright Clusterer
    // Runs the classic Clarke & Wright Savings algorithm (S_ij = d_i0 + d_0j - d_ij) 
    // to group nodes into routes, stopping when capacity/size is reached.
    class ClarkeWrightClusterer : public MicroClusterer {
    public:
        std::vector<SubProblem> cluster(
            const Node& depot,
            NodeSpan wedge_nodes, 
            double vehicle_capacity,
            int target_subproblem_size
        ) override;
    };

} // namespace cvrp::partitioning
