#include "MicroClusterer.hpp"
#include <limits>
#include <cmath>
#include <iostream>

namespace cvrp::partitioning {

std::vector<SubProblem> MSTClusterer::cluster(
    const Node& depot,
    NodeSpan wedge_nodes, 
    double vehicle_capacity,
    int target_subproblem_size
) {
    int N = wedge_nodes.size();
    std::vector<SubProblem> result;
    if (N == 0) return result;

    // Prim's algorithm on the fly
    std::vector<double> min_wt(N, std::numeric_limits<double>::max());
    std::vector<int> parent(N, -1);

    std::vector<int> unvisited;
    unvisited.reserve(N);
    for (int i = 1; i < N; ++i) {
        unvisited.push_back(i);
    }
    
    int u = 0; // Start MST from node 0

    for (int count = 1; count < N; count++) {
        double ux = wedge_nodes[u].x;
        double uy = wedge_nodes[u].y;

        int next_u = -1;
        int next_u_idx = -1;
        double min_val = std::numeric_limits<double>::max();

        // Update neighbors and find the next node simultaneously
        for (size_t i = 0; i < unvisited.size(); i++) {
            int v = unvisited[i];
            double dx = wedge_nodes[v].x - ux;
            double dy = wedge_nodes[v].y - uy;
            double dist_sq = dx*dx + dy*dy;
            
            if (dist_sq < min_wt[v]) {
                min_wt[v] = dist_sq;
                parent[v] = u;
            }
            
            if (min_wt[v] < min_val) {
                min_val = min_wt[v];
                next_u = v;
                next_u_idx = i;
            }
        }

        if (next_u == -1) break; // Disconnected graph or precision failure
        
        u = next_u;
        // Fast removal of next_u from unvisited
        unvisited[next_u_idx] = unvisited.back();
        unvisited.pop_back();
    }

    // Build adjacency list for the MST
    std::vector<std::vector<int>> adj(N);
    for (int i = 1; i < N; i++) {
        if (parent[i] != -1) {
            adj[parent[i]].push_back(i);
            adj[i].push_back(parent[i]);
        }
    }

    // DFS to get pre-order traversal sequence
    std::vector<int> pre_order;
    pre_order.reserve(N);
    
    // Iterative DFS to prevent stack overflow on deep trees
    std::vector<int> stack;
    std::vector<bool> visited(N, false);
    stack.push_back(0);
    visited[0] = true;

    while (!stack.empty()) {
        int curr = stack.back();
        stack.pop_back();
        pre_order.push_back(curr);

        for (int v : adj[curr]) {
            if (!visited[v]) {
                visited[v] = true;
                stack.push_back(v);
            }
        }
    }

    // Permute wedge_nodes to match pre_order
    std::vector<Node> permuted(N);
    for (int i = 0; i < N; i++) {
        permuted[i] = wedge_nodes[pre_order[i]];
    }
    for (int i = 0; i < N; i++) {
        wedge_nodes[i] = permuted[i];
    }

    // Slice the rearranged wedge_nodes into chunks of target_subproblem_size
    for (int i = 0; i < N; i += target_subproblem_size) {
        SubProblem sp;
        int len = std::min(N - i, target_subproblem_size);
        sp.nodes = wedge_nodes.subspan(i, len);
        result.push_back(sp);
    }

    return result;
}

} // namespace cvrp::partitioning
