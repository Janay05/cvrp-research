#include "Stage1_Construction.hpp"
#include "UnionFind.hpp"
#include <algorithm>

namespace {
    struct Edge {
        NodeId u, v;
        Cost cost;
        bool operator<(const Edge& o) const { return cost < o.cost; }
    };

    void dfs_recursive(NodeId node, const std::vector<std::vector<NodeId>>& mst_adj, std::vector<bool>& visited, std::vector<NodeId>& order, NodeId depot_local) {
        visited[node] = true;
        if (node != depot_local) {
            order.push_back(node);
        }
        for (NodeId neighbor : mst_adj[node]) {
            if (!visited[neighbor]) {
                dfs_recursive(neighbor, mst_adj, visited, order, depot_local);
            }
        }
    }

    std::vector<std::vector<NodeId>> convert_to_routes(const std::vector<NodeId>& order, const std::vector<int64_t>& demand_local, Cost Q) {
        std::vector<std::vector<NodeId>> routes;
        std::vector<NodeId> current_route;
        Cost residual = Q;

        for (NodeId node : order) {
            if (residual - demand_local[node] >= 0) {
                current_route.push_back(node);
                residual -= demand_local[node];
            } else {
                routes.push_back(current_route);
                current_route.clear();
                current_route.push_back(node);
                residual = Q - demand_local[node];
            }
        }
        if (!current_route.empty()) {
            routes.push_back(current_route);
        }
        return routes;
    }

    Cost compute_total_cost(const std::vector<std::vector<NodeId>>& routes, const std::vector<NodeId>& localToGlobal, const Instance& inst) {
        Cost total = 0;
        for (const auto& r : routes) {
            if (r.empty()) continue;
            total += dist(inst, 0, localToGlobal[r.front()]);
            for (size_t i = 0; i < r.size() - 1; ++i) {
                total += dist(inst, localToGlobal[r[i]], localToGlobal[r[i+1]]);
            }
            total += dist(inst, localToGlobal[r.back()], 0);
        }
        return total;
    }
}

Solution stage1_construct(int chunkId, const Instance& inst, const Stage0Result& partitionInfo, const NeighborLists& neighborLists, std::mt19937& rng) {
    const auto& globalIds = partitionInfo.globalId[chunkId];
    int chunkSize = globalIds.size() - 1; // excluding depot
    NodeId depot_local = 0;

    std::vector<int64_t> demand_local(chunkSize + 1);
    demand_local[0] = 0;
    for (int i = 1; i <= chunkSize; ++i) {
        demand_local[i] = inst.demand[globalIds[i]];
    }

    std::vector<Edge> chunk_edges;
    for (int i = 1; i <= chunkSize; ++i) {
        NodeId global_i = globalIds[i];
        for (NodeId global_j : neighborLists.nbr[global_i]) {
            if (partitionInfo.chunkOf[global_j] == chunkId) {
                NodeId local_j = partitionInfo.localId[global_j];
                chunk_edges.push_back({(NodeId)i, local_j, dist(inst, global_i, global_j)});
            }
        }
        chunk_edges.push_back({depot_local, (NodeId)i, dist(inst, 0, global_i)});
    }

    std::sort(chunk_edges.begin(), chunk_edges.end());
    UnionFind uf(chunkSize + 1);
    std::vector<Edge> mst_edges;
    
    for (const auto& edge : chunk_edges) {
        if (uf.unite(edge.u, edge.v)) {
            mst_edges.push_back(edge);
            if (mst_edges.size() == (size_t)chunkSize) break;
        }
    }

    if (mst_edges.size() < (size_t)chunkSize) {
        std::vector<Edge> fallback_edges;
        for (int i = 0; i <= chunkSize; ++i) {
            for (int j = i + 1; j <= chunkSize; ++j) {
                if (uf.find(i) != uf.find(j)) {
                    Cost d = dist(inst, globalIds[i], globalIds[j]);
                    fallback_edges.push_back({(NodeId)i, (NodeId)j, d});
                }
            }
        }
        std::sort(fallback_edges.begin(), fallback_edges.end());
        for (const auto& edge : fallback_edges) {
            if (uf.unite(edge.u, edge.v)) {
                mst_edges.push_back(edge);
                if (mst_edges.size() == (size_t)chunkSize) break;
            }
        }
    }

    std::vector<std::vector<NodeId>> mst_adj(chunkSize + 1);
    for (const auto& edge : mst_edges) {
        mst_adj[edge.u].push_back(edge.v);
        mst_adj[edge.v].push_back(edge.u);
    }

    int rho = 100;
    
    Cost best_cost = -1;
    std::vector<std::vector<NodeId>> best_routes;

    for (int attempt = 0; attempt < rho; ++attempt) {
        auto adj_copy = mst_adj;
        for (auto& neighbors : adj_copy) {
            std::shuffle(neighbors.begin(), neighbors.end(), rng);
        }
        
        std::vector<bool> visited(chunkSize + 1, false);
        std::vector<NodeId> order;
        dfs_recursive(depot_local, adj_copy, visited, order, depot_local);
        
        auto routes = convert_to_routes(order, demand_local, inst.Q);
        Cost cost = compute_total_cost(routes, globalIds, inst);
        
        if (best_cost == -1 || cost < best_cost) {
            best_cost = cost;
            best_routes = routes;
        }
    }

    Solution sol;
    sol.pred.assign(inst.n + 1, 0);
    sol.succ.assign(inst.n + 1, 0);
    sol.routeOf.assign(inst.n + 1, -1);
    sol.routeHead.resize(best_routes.size());
    sol.routeTail.resize(best_routes.size());
    sol.routeLoad.resize(best_routes.size());
    sol.routePosition.assign(inst.n + 1, 0);
    sol.cumLoad.assign(inst.n + 1, 0);
    sol.numRoutes = best_routes.size();
    sol.totalCost = best_cost;

    for (size_t r = 0; r < best_routes.size(); ++r) {
        const auto& route = best_routes[r];
        sol.routeHead[r] = globalIds[route.front()];
        sol.routeTail[r] = globalIds[route.back()];
        Cost load = 0;
        
        for (size_t i = 0; i < route.size(); ++i) {
            NodeId local_c = route[i];
            NodeId c = globalIds[local_c];
            sol.routeOf[c] = r;
            load += demand_local[local_c];
            
            if (i > 0) sol.pred[c] = globalIds[route[i-1]];
            else sol.pred[c] = 0;
            
            if (i < route.size() - 1) sol.succ[c] = globalIds[route[i+1]];
            else sol.succ[c] = 0;
            
            sol.routePosition[c] = i + 1;
            sol.cumLoad[c] = load;
        }
        sol.routeLoad[r] = load;
    }

    return sol;
}
