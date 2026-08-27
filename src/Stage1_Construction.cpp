#include "Stage1_Construction.hpp"
#include "UnionFind.hpp"
#include <algorithm>
#include <functional>

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

    // T6 (docs/reports/009_plan_beating_filo2.md): Clarke & Wright savings construction,
    // ported from FILO2's solution/savings.hpp. Returns routes in LOCAL chunk ids so it
    // drops straight into the same Solution-building code the MST path already uses.
    //
    // Why: measured, our MST+randomized-DFS construction starts materially worse than CW --
    // 22,522,444 vs 22,231,600 at Valle-D-Aosta (1.3%), and 3,208,434,488 vs 3,177,770,000
    // at Lazio (0.96%). At Lazio that matters enormously: FILO2's CW output alone
    // (3,177,770,000 / 40,252 routes) beats our entire 315 s 4-core FINAL answer
    // (3,182,981,663 / 40,431 routes), on both cost and route count. Report 009 estimated
    // this technique at "0-0.15%"; measurement says ~1%.
    //
    // Savings s(i,j) = d(0,i) + d(0,j) - lambda*d(i,j): how much is saved by serving i and j
    // on one route (i's tail joined to j's head) instead of two separate out-and-back trips.
    // Merges are applied greedily in descending savings order, subject to i being its route's
    // tail, j its route's head, different routes, and combined load fitting capacity.
    //
    // Route identity under merging is tracked with a local union-find (route id == the local
    // id of the customer that started it) so "which route is i in?" stays near-O(1) as routes
    // combine; the sequence itself is a `next` linked list so a merge is O(1) pointer work
    // rather than copying one route's customers into another.
    std::vector<std::vector<NodeId>> clarke_wright_routes(int chunkId, const Instance& inst,
                                                           const Stage0Result& partitionInfo,
                                                           const NeighborLists& neighborLists,
                                                           int chunkSize,
                                                           const std::vector<int64_t>& demand_local,
                                                           int cw_neighbors) {
        const auto& globalIds = partitionInfo.globalId[chunkId];

        std::vector<NodeId> next(chunkSize + 1, 0);
        std::vector<int> parent(chunkSize + 1), usize(chunkSize + 1, 1);
        std::vector<NodeId> head(chunkSize + 1), tail(chunkSize + 1);
        std::vector<Cost> load(chunkSize + 1, 0);
        for (int i = 0; i <= chunkSize; ++i) {
            parent[i] = i;
            head[i] = (NodeId)i;
            tail[i] = (NodeId)i;
            load[i] = (i == 0) ? 0 : demand_local[i];
        }
        std::function<int(int)> find = [&](int x) {
            while (parent[x] != x) { parent[x] = parent[parent[x]]; x = parent[x]; }
            return x;
        };

        struct Saving { NodeId i, j; Cost value; };
        std::vector<Saving> savings;
        savings.reserve((size_t)chunkSize * std::min(cw_neighbors, 16));

        for (int i = 1; i <= chunkSize; ++i) {
            NodeId global_i = globalIds[i];
            Cost d0i = dist(inst, 0, global_i);
            int added = 0;
            for (NodeId global_j : neighborLists.nbr[global_i]) {
                if (added >= cw_neighbors) break;
                if (partitionInfo.chunkOf[global_j] != chunkId) continue;
                NodeId local_j = partitionInfo.localId[global_j];
                if (local_j <= 0 || local_j > chunkSize) continue;
                // i < j only: the saving is symmetric, so generating both directions would
                // just double the sort cost for identical merge decisions.
                if ((NodeId)i >= local_j) continue;
                Cost value = d0i + dist(inst, 0, global_j) - dist(inst, global_i, global_j);
                savings.push_back({(NodeId)i, local_j, value});
                ++added;
            }
        }

        std::sort(savings.begin(), savings.end(),
                  [](const Saving& a, const Saving& b) { return a.value > b.value; });

        for (const auto& s : savings) {
            int ri = find(s.i), rj = find(s.j);
            if (ri == rj) continue;
            if (load[ri] + load[rj] > inst.Q) continue;

            NodeId newHead, newTail;
            if (tail[ri] == s.i && head[rj] == s.j) {
                next[s.i] = s.j;
                newHead = head[ri]; newTail = tail[rj];
            } else if (tail[rj] == s.j && head[ri] == s.i) {
                next[s.j] = s.i;
                newHead = head[rj]; newTail = tail[ri];
            } else {
                continue; // neither orientation joins an end to an end
            }

            Cost merged = load[ri] + load[rj];
            int root = ri, other = rj;
            if (usize[ri] < usize[rj]) { root = rj; other = ri; }
            parent[other] = root;
            usize[root] += usize[other];
            head[root] = newHead; tail[root] = newTail; load[root] = merged;
        }

        std::vector<std::vector<NodeId>> routes;
        for (int i = 1; i <= chunkSize; ++i) {
            if (find(i) != i) continue; // not a route root
            std::vector<NodeId> r;
            for (NodeId c = head[i]; c != 0; c = next[c]) {
                r.push_back(c);
                if ((int)r.size() > chunkSize) break; // cycle guard
            }
            if (!r.empty()) routes.push_back(std::move(r));
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

Solution stage1_construct(int chunkId, const Instance& inst, const Stage0Result& partitionInfo,
                          const NeighborLists& neighborLists, std::mt19937& rng,
                          const NeighborLists* cw_neighborLists) {
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

    Cost best_cost = -1;
    std::vector<std::vector<NodeId>> best_routes;

    // T6: Clarke & Wright instead of MST+randomized-DFS when enabled -- see
    // clarke_wright_routes above for the measured justification. The MST path below is kept
    // (not deleted) so the two are A/B-comparable on the same build.
    extern int g_use_clarke_wright;
    extern int g_cw_neighbors;
    if (g_use_clarke_wright) {
        // Deliberately the wide list when one is supplied -- see the header comment.
        const NeighborLists& cwLists = cw_neighborLists ? *cw_neighborLists : neighborLists;
        best_routes = clarke_wright_routes(chunkId, inst, partitionInfo, cwLists,
                                           chunkSize, demand_local, g_cw_neighbors);
        best_cost = compute_total_cost(best_routes, globalIds, inst);
    }

    int rho = 100;

    for (int attempt = 0; g_use_clarke_wright == 0 && attempt < rho; ++attempt) {
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
    sol.costToPred.assign(inst.n + 1, 0);
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
            load += inst.demand[c];

            // cumLoad[c] must include c's own demand (load incremented BEFORE assignment) --
            // matching update_route_info's convention (Stage2_ILS.cpp) exactly. This order
            // was swapped by an earlier commit, understating every node's cumLoad by its own
            // demand at construction time (worst-visible on route heads: cumLoad=0 instead of
            // demand[head]). That let eval_2opt_star's capacity check pass genuinely
            // over-capacity 2-opt* moves on any route local_search hadn't yet touched (nothing
            // else ever recomputes cumLoad for an untouched route), producing real capacity
            // violations days into Stage 2/5 -- confirmed via a reproducible VDA crash trail.
            sol.routePosition[c] = i + 1;
            sol.cumLoad[c] = load;

            if (i > 0) sol.pred[c] = globalIds[route[i-1]];
            else sol.pred[c] = 0;

            if (i < route.size() - 1) sol.succ[c] = globalIds[route[i+1]];
            else sol.succ[c] = 0;

            sol.costToPred[c] = dist(inst, sol.pred[c], c);
        }
        sol.routeLoad[r] = load;
    }

    return sol;
}
