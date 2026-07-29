#include "MacroPartitioner.hpp"
#include <algorithm>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace cvrp::partitioning {

    MacroPartitioner::MacroPartitioner(double sweep_angle_degrees) 
        : sweep_angle_degrees_(sweep_angle_degrees) {}

    void MacroPartitioner::compute_polar_angles(const Node& depot, std::vector<Node>& nodes) {
        for (auto& node : nodes) {
            double dx = node.x - depot.x;
            double dy = node.y - depot.y;
            node.polar_angle = std::atan2(dy, dx);
            // Convert from [-pi, pi] to [0, 2*pi]
            if (node.polar_angle < 0) {
                node.polar_angle += 2.0 * M_PI;
            }
        }
    }

    void MacroPartitioner::sort_by_angle(std::vector<Node>& nodes) {
        std::sort(nodes.begin(), nodes.end(), [](const Node& a, const Node& b) {
            return a.polar_angle < b.polar_angle;
        });
    }

    std::vector<NodeSpan> MacroPartitioner::partition(const Node& depot, std::vector<Node>& global_nodes) {
        std::vector<NodeSpan> wedges;
        if (global_nodes.empty()) return wedges;

        // 1. Compute angles
        compute_polar_angles(depot, global_nodes);

        // 2. Sort nodes by angle
        sort_by_angle(global_nodes);

        // 3. Partition into wedges
        double sweep_rads = sweep_angle_degrees_ * (M_PI / 180.0);
        
        int start_idx = 0;
        double current_wedge_end_angle = global_nodes[0].polar_angle + sweep_rads;

        for (int i = 0; i < global_nodes.size(); ++i) {
            if (global_nodes[i].polar_angle > current_wedge_end_angle) {
                // Cut the wedge
                wedges.push_back(NodeSpan(&global_nodes[start_idx], i - start_idx));
                start_idx = i;
                current_wedge_end_angle += sweep_rads;
                
                // Fast-forward empty wedges if data is sparse
                while (global_nodes[i].polar_angle > current_wedge_end_angle) {
                    current_wedge_end_angle += sweep_rads;
                }
            }
        }

        // Add the final wedge
        if (start_idx < global_nodes.size()) {
            wedges.push_back(NodeSpan(&global_nodes[start_idx], global_nodes.size() - start_idx));
        }

        return wedges;
    }

} // namespace cvrp::partitioning
