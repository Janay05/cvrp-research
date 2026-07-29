#include "MicroClusterer.hpp"
#include <algorithm>
#include <iostream>
#include <cmath>

namespace cvrp::partitioning {

namespace {
    // Rotate/flip a quadrant appropriately for Hilbert curve
    void rot(uint32_t n, uint32_t *x, uint32_t *y, uint32_t rx, uint32_t ry) {
        if (ry == 0) {
            if (rx == 1) {
                *x = n - 1 - *x;
                *y = n - 1 - *y;
            }
            // Swap x and y
            uint32_t t = *x;
            *x = *y;
            *y = t;
        }
    }

    // Convert (x,y) to d (Hilbert index)
    uint64_t xy2d(uint32_t n, uint32_t x, uint32_t y) {
        uint32_t rx, ry, s;
        uint64_t d = 0;
        for (s = n / 2; s > 0; s /= 2) {
            rx = (x & s) > 0;
            ry = (y & s) > 0;
            d += (uint64_t)s * s * ((3 * rx) ^ ry);
            rot(s, &x, &y, rx, ry);
        }
        return d;
    }
}

std::vector<SubProblem> HilbertCurveClusterer::cluster(
    const Node& depot,
    NodeSpan wedge_nodes, 
    double vehicle_capacity,
    int target_subproblem_size) 
{
    if (wedge_nodes.size() == 0) return {};

    // 1. Find the bounding box of the wedge
    double min_x = wedge_nodes[0].x;
    double max_x = wedge_nodes[0].x;
    double min_y = wedge_nodes[0].y;
    double max_y = wedge_nodes[0].y;

    for (size_t i = 1; i < wedge_nodes.size(); ++i) {
        if (wedge_nodes[i].x < min_x) min_x = wedge_nodes[i].x;
        if (wedge_nodes[i].x > max_x) max_x = wedge_nodes[i].x;
        if (wedge_nodes[i].y < min_y) min_y = wedge_nodes[i].y;
        if (wedge_nodes[i].y > max_y) max_y = wedge_nodes[i].y;
    }

    // Prevent division by zero if all points are identical
    double range_x = std::max(max_x - min_x, 1e-9);
    double range_y = std::max(max_y - min_y, 1e-9);
    double range = std::max(range_x, range_y);

    // 2. We use a 2^31 x 2^31 grid for near-perfect precision
    uint32_t n = 1U << 31;

    // 3. Sort the NodeSpan in-place based on their Hilbert index
    std::sort(wedge_nodes.begin(), wedge_nodes.end(), 
        [=](const Node& a, const Node& b) {
            // Normalize coordinates to [0, n-1]
            uint32_t ax = static_cast<uint32_t>(((a.x - min_x) / range) * (n - 1));
            uint32_t ay = static_cast<uint32_t>(((a.y - min_y) / range) * (n - 1));
            uint64_t ha = xy2d(n, ax, ay);

            uint32_t bx = static_cast<uint32_t>(((b.x - min_x) / range) * (n - 1));
            uint32_t by = static_cast<uint32_t>(((b.y - min_y) / range) * (n - 1));
            uint64_t hb = xy2d(n, bx, by);

            return ha < hb;
        }
    );

    // 4. Slice the sorted 1D geographical list into chunks
    std::vector<SubProblem> subproblems;
    size_t start_idx = 0;
    
    while (start_idx < wedge_nodes.size()) {
        size_t end_idx = start_idx;
        double current_demand = 0.0;
        int current_nodes = 0;

        // Add nodes until we hit target size (or capacity if we wanted strict capacity constraints)
        while (end_idx < wedge_nodes.size() && current_nodes < target_subproblem_size) {
            // Note: If strict vehicle capacity per subproblem is desired, uncomment this:
            // if (current_demand + wedge_nodes[end_idx].demand > vehicle_capacity) break;
            
            current_demand += wedge_nodes[end_idx].demand;
            current_nodes++;
            end_idx++;
        }

        // Extract the subspan
        SubProblem sp;
        sp.nodes = wedge_nodes.subspan(start_idx, end_idx - start_idx);
        subproblems.push_back(sp);

        start_idx = end_idx;
    }

    return subproblems;
}

} // namespace cvrp::partitioning
