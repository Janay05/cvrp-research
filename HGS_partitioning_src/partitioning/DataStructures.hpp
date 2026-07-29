#pragma once

#include <vector>
// No <span> in C++14
#include <cstdint>
#include <cmath>
#include <cstddef>

namespace cvrp::partitioning {

    struct Node {
        int original_id;
        double x;
        double y;
        double demand;
        
        // Computed metrics for Stage 1 (Macro-Partitioning)
        double polar_angle;       // Angle relative to depot (node 0)
        double distance_to_depot; // Radial distance from depot
    };

    template <typename T>
    struct Span {
        T* ptr;
        std::size_t len;

        Span() : ptr(nullptr), len(0) {}
        Span(T* p, std::size_t l) : ptr(p), len(l) {}
        Span(std::vector<T>& v) : ptr(v.data()), len(v.size()) {}

        T* begin() const { return ptr; }
        T* end() const { return ptr + len; }
        std::size_t size() const { return len; }
        bool empty() const { return len == 0; }
        T& operator[](std::size_t i) const { return ptr[i]; }
        
        Span<T> subspan(std::size_t offset, std::size_t count) const {
            return Span<T>(ptr + offset, count);
        }
    };

    // A contiguous view over the global node array. 
    // Stage 1 sorts the global array by polar angle, enabling contiguous macro wedges.
    // This allows zero-copy partitioning for multithreading.
    using NodeSpan = Span<Node>;
    using ConstNodeSpan = Span<const Node>;

    // Represents a constructed route. To avoid duplicating Node data,
    // we store indices that reference the global Node array.
    struct Route {
        int route_id;
        std::vector<int> node_indices; // Indices referencing the global Node array
        double total_demand;
        double total_distance; // Includes stem distance to/from depot
        
        // Helper to check utilization
        double capacity_utilization(double max_capacity) const {
            return total_demand / max_capacity;
        }
    };

    // Represents a sub-problem (e.g., a micro cluster passed to HGS)
    // containing a subset of nodes and the routes formed from them.
    struct SubProblem {
        int id;
        NodeSpan nodes; // Contiguous block of nodes for this sub-problem
        
        // In some clustering algorithms, nodes might not be perfectly contiguous 
        // after further processing, so an index-based view could also be provided if needed.
        // std::vector<int> node_indices; 
    };

} // namespace cvrp::partitioning
