#include "MicroClusterer.hpp"
#include <algorithm>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace cvrp::partitioning {

    ConcentricSweepClusterer::ConcentricSweepClusterer(int num_bands) 
        : num_bands_(num_bands) {}

    std::vector<SubProblem> ConcentricSweepClusterer::cluster(
        const Node& depot,
        NodeSpan wedge_nodes, 
        double vehicle_capacity,
        int target_subproblem_size
    ) {
        std::vector<SubProblem> micro_partitions;
        if (wedge_nodes.empty()) return micro_partitions;

        // 1. Radial Distance Computation O(N)
        // Computes exact Euclidean distance from Depot (Node 0) to each node.
        for (auto& node : wedge_nodes) {
            double dx = node.x - depot.x;
            double dy = node.y - depot.y;
            node.distance_to_depot = std::sqrt(dx * dx + dy * dy);
            node.polar_angle = std::atan2(dy, dx);
            if (node.polar_angle < 0) {
                node.polar_angle += 2.0 * M_PI;
            }
        }

        // 2. Concentric Banding (Bucketing)
        // Sort the subset strictly by distance from the depot.
        // Because wedge_nodes is a NodeSpan, this sorts the global nodes in-place for this thread.
        std::sort(wedge_nodes.begin(), wedge_nodes.end(), [](const Node& a, const Node& b) {
            return a.distance_to_depot < b.distance_to_depot;
        });

        // Use quantiles to determine dynamic band sizes
        int nodes_per_band = wedge_nodes.size() / num_bands_;
        if (nodes_per_band < target_subproblem_size) {
            nodes_per_band = target_subproblem_size; // Ensure at least one subproblem per band
        }

        int current_band_start = 0;
        int subproblem_id_counter = 0;

        while (current_band_start < wedge_nodes.size()) {
            int current_band_end = std::min(static_cast<int>(wedge_nodes.size()), current_band_start + nodes_per_band);
            
            // If the remaining nodes after this band are too few, absorb them into the current band
            if (wedge_nodes.size() - current_band_end < target_subproblem_size / 2) {
                current_band_end = wedge_nodes.size();
            }

            auto band_span = wedge_nodes.subspan(current_band_start, current_band_end - current_band_start);

            // 3. Angular Sorting within Bands O(B log B)
            // For every independent distance Band, sort strictly by polar angle.
            std::sort(band_span.begin(), band_span.end(), [](const Node& a, const Node& b) {
                return a.polar_angle < b.polar_angle;
            });

            // 4. Capacity-Aware Micro-Sweeping O(B)
            // Calculate total demand in the band to estimate a reasonable target demand per cluster
            double total_band_demand = 0.0;
            for (const auto& node : band_span) {
                total_band_demand += node.demand;
            }
            
            double avg_demand_per_node = total_band_demand / band_span.size();
            double target_cluster_demand = avg_demand_per_node * target_subproblem_size;

            int micro_start = 0;
            double accumulated_demand = 0.0;
            int accumulated_nodes = 0;

            std::vector<std::pair<int, int>> band_cuts; // stores {start, end} indices relative to band_span

            for (int i = 0; i < band_span.size(); ++i) {
                accumulated_demand += band_span[i].demand;
                accumulated_nodes++;

                // Cut Condition: Slices the array when accumulated demand or nodes reach the target
                if (accumulated_nodes >= target_subproblem_size || accumulated_demand >= target_cluster_demand) {
                    band_cuts.push_back({micro_start, i + 1});
                    micro_start = i + 1;
                    accumulated_demand = 0.0;
                    accumulated_nodes = 0;
                }
            }

            // 5. Edge Case Handling (Stragglers within the Band)
            if (micro_start < band_span.size()) {
                int leftover_nodes = band_span.size() - micro_start;
                // Rule: If remainder is < 20% of target, merge with the preceding micro-partition
                if (leftover_nodes < 0.2 * target_subproblem_size && !band_cuts.empty()) {
                    band_cuts.back().second = band_span.size();
                } else {
                    band_cuts.push_back({micro_start, static_cast<int>(band_span.size())});
                }
            }

            // Translate cuts into SubProblems using zero-copy NodeSpans
            for (const auto& cut : band_cuts) {
                SubProblem sp;
                sp.id = ++subproblem_id_counter;
                sp.nodes = band_span.subspan(cut.first, cut.second - cut.first);
                micro_partitions.push_back(sp);
            }

            current_band_start = current_band_end;
        }

        return micro_partitions;
    }

} // namespace cvrp::partitioning
