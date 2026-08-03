#include "Stage3_MergeHealing.hpp"
#include <set>
#include <thread>
#include <vector>
#include <algorithm>
#include <random>
#include <iostream>

void stage3_healing_ils_pass(Solution& globalSolution, ThreadArena& arena, SVCCache& cache, 
                             const Instance& inst, const NeighborLists& neighborLists, 
                             const Stage0Result& partitionInfo,
                             const std::vector<int>& boundaryList, 
                             int t1, int t2, std::mt19937& rng,
                             const std::vector<int>* routeToChunk = nullptr);

void merge_all_chunks_into_global(Solution& globalSolution, 
                                  const std::vector<Solution>& chunkSolutions, 
                                  const Stage0Result& partitionInfo,
                                  const Instance& inst) {
    int n = inst.n;
    globalSolution.pred.assign(n + 1, 0);
    globalSolution.succ.assign(n + 1, 0);
    globalSolution.routeOf.assign(n + 1, -1);
    globalSolution.routePosition.assign(n + 1, 0);
    globalSolution.cumLoad.assign(n + 1, 0);
    globalSolution.totalCost = 0;
    globalSolution.numRoutes = 0;
    
    int current_route_index = 0;
    
    for (size_t t = 0; t < chunkSolutions.size(); ++t) {
        const auto& sol = chunkSolutions[t];
        const auto& globalIds = partitionInfo.globalId[t];
        
        globalSolution.totalCost += sol.totalCost;
        
        for (size_t r = 0; r < (size_t)sol.numRoutes; ++r) {
            globalSolution.routeHead.push_back(sol.routeHead[r]);
            globalSolution.routeTail.push_back(sol.routeTail[r]);
            globalSolution.routeLoad.push_back(sol.routeLoad[r]);
        }
        
        for (size_t i = 1; i < globalIds.size(); ++i) {
            NodeId global_id = globalIds[i];
            
            globalSolution.pred[global_id] = sol.pred[global_id];
            globalSolution.succ[global_id] = sol.succ[global_id];
            
            if (sol.routeOf[global_id] != -1) {
                globalSolution.routeOf[global_id] = sol.routeOf[global_id] + current_route_index;
            }
        }
        globalSolution.numRoutes += sol.numRoutes;
        current_route_index = globalSolution.numRoutes;
    }
    
    for (int r = 0; r < globalSolution.numRoutes; ++r) {
        NodeId curr = globalSolution.routeHead[r];
        int pos = 1;
        Cost current_load = 0;
        while (curr != 0) {
            globalSolution.routePosition[curr] = pos++;
            current_load += inst.demand[curr];
            globalSolution.cumLoad[curr] = current_load;
            curr = globalSolution.succ[curr];
        }
    }
}

void run_stage3_healing(Solution& globalSolution, 
                        const Instance& inst, 
                        const Stage0Result& partitionInfo,
                        const NeighborLists& neighborLists) {
    
    // Pre-allocate extra routes to prevent reallocation data races in concurrent healing
    int max_possible_routes = inst.n + 100;
    if ((int)globalSolution.routeHead.size() < max_possible_routes) {
        globalSolution.routeHead.resize(max_possible_routes, 0);
        globalSolution.routeTail.resize(max_possible_routes, 0);
        globalSolution.routeLoad.resize(max_possible_routes, 0);
    }
    
    // Build routeToChunk mapping for boundary restriction
    std::vector<int> routeToChunk(max_possible_routes, -1);
    int curr_r = 0;
    for (size_t t = 0; t < partitionInfo.numChunks; ++t) {
        // If chunk size array was available, this would be easier.
        // Instead, we scan globalSolution to map routes back to chunks based on nodes
    }
    // Better way: Just assign during the loop
    for (int i = 1; i <= inst.n; ++i) {
        if (globalSolution.routeOf[i] != -1) {
            routeToChunk[globalSolution.routeOf[i]] = partitionInfo.chunkOf[i];
        }
    }
    
    std::set<std::pair<int, int>> edges;
    for (int i : partitionInfo.boundaryList) {
        int c_i = partitionInfo.chunkOf[i];
        for (int c_j : partitionInfo.boundaryChunkPair[i]) {
            int u = std::min(c_i, c_j);
            int v = std::max(c_i, c_j);
            edges.insert({u, v});
        }
    }
    
    std::vector<std::pair<int, int>> edge_list(edges.begin(), edges.end());
    int num_edges = edge_list.size();
    
    std::vector<int> edge_color(num_edges, -1);
    int max_color = 0;
    
    for (int e = 0; e < num_edges; ++e) {
        int u = edge_list[e].first;
        int v = edge_list[e].second;
        
        std::set<int> used_colors;
        for (int i = 0; i < e; ++i) {
            int u2 = edge_list[i].first;
            int v2 = edge_list[i].second;
            if (u == u2 || u == v2 || v == u2 || v == v2) {
                used_colors.insert(edge_color[i]);
            }
        }
        
        int c = 0;
        while (used_colors.count(c)) c++;
        edge_color[e] = c;
        if (c > max_color) max_color = c;
    }
    
    std::vector<std::vector<std::pair<int, int>>> color_classes(max_color + 1);
    for (int e = 0; e < num_edges; ++e) {
        color_classes[edge_color[e]].push_back(edge_list[e]);
    }
    
    for (int c = 0; c <= max_color; ++c) {
        std::vector<std::thread> threads;
        const auto& class_edges = color_classes[c];
        
        for (size_t t_idx = 0; t_idx < class_edges.size(); ++t_idx) {
            int t1 = class_edges[t_idx].first;
            int t2 = class_edges[t_idx].second;
            
            std::vector<int> pair_boundary;
            for (int i : partitionInfo.boundaryList) {
                int c_i = partitionInfo.chunkOf[i];
                if (c_i == t1) {
                    for (int c_j : partitionInfo.boundaryChunkPair[i]) {
                        if (c_j == t2) { pair_boundary.push_back(i); break; }
                    }
                } else if (c_i == t2) {
                    for (int c_j : partitionInfo.boundaryChunkPair[i]) {
                        if (c_j == t1) { pair_boundary.push_back(i); break; }
                    }
                }
            }
            
            // deduplicate
            std::sort(pair_boundary.begin(), pair_boundary.end());
            pair_boundary.erase(std::unique(pair_boundary.begin(), pair_boundary.end()), pair_boundary.end());
            
            if (pair_boundary.empty()) continue;
            
            // heal boundaries in parallel
            threads.emplace_back([&, pair_boundary, t1, t2]() mutable {
                ThreadArena arena;
                arena.reserve_fixed_capacity(inst.n);
                SVCCache cache;
                cache.init(inst.n);
                std::mt19937 rng(1337 + t1 * 1000 + t2);
                std::cout << "Healing chunk pair (" << t1 << "," << t2 << ")" << std::endl;
                for (int cust : pair_boundary) cache.insert(cust);
                stage3_healing_ils_pass(globalSolution, arena, cache, inst, neighborLists, partitionInfo, pair_boundary, t1, t2, rng, &routeToChunk);
            });
        }
        
        for (auto& th : threads) {
            th.join();
        }
    }
}
