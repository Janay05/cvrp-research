#include "Stage3_MergeHealing.hpp"
#include <set>
#include <thread>
#include <vector>
#include <algorithm>
#include <random>
#include <iostream>

Cost stage3_healing_ils_pass(Solution& globalSolution, ThreadArena& arena, SVCCache& cache,
                             const Instance& inst, const NeighborLists& neighborLists,
                             const Stage0Result& partitionInfo,
                             const std::vector<int>& boundaryList,
                             int t1, int t2, std::mt19937& rng,
                             std::vector<int>* routeToChunk = nullptr);

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
    globalSolution.costToPred.assign(n + 1, 0);
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
            globalSolution.costToPred[curr] = dist(inst, globalSolution.pred[curr], curr);
            curr = globalSolution.succ[curr];
        }
        globalSolution.routeLoad[r] = current_load; // Recompute to be absolutely safe
    }
}

void run_stage3_healing(Solution& globalSolution, 
                        const Instance& inst, 
                        const Stage0Result& partitionInfo,
                        const NeighborLists& neighborLists) {
    
    // Pre-allocate extra routes to prevent reallocation data races in concurrent healing.
    // This must be a genuinely provable bound, not a cushion: local_search's evaluation
    // phase (get_top3_insertions, eval_2opt_star, etc.) reads globalSolution.routeHead/
    // Tail/Load WITHOUT the mutex (see the comment at that call site), so if recreate()'s
    // own route-creation path (mutex-protected) ever needs to actually reallocate one of
    // these vectors while another thread holds a bare, unlocked pointer/reference into it,
    // that's a real use-after-free / access violation, not just stale data -- confirmed via
    // crashes during Tier-1 stress testing (see docs/reports/005_cost_optimization.md Phase
    // 1). `inst.n + 100` is not that bound: a route can never hold more than inst.n
    // customers total (so live routes alone can't exceed inst.n), but every route CREATED
    // during a rejected iteration leaks its slot permanently (see routeToChunk's -1 label
    // for healing-created routes, below) -- and since removing the racy numRoutes
    // snapshot/restore (Phase 1 fix), numRoutes is monotonically non-decreasing for the rest
    // of this function, so those leaked slots keep accumulating instead of resetting each
    // rejected iteration. 2x + a large flat slack comfortably covers even every iteration of
    // every chunk-pair thread leaking a route (each pass removes only a handful of customers
    // per iteration, capped at 1000 iterations/pair), while costing a few tens of MB at most.
    int max_possible_routes = 2 * inst.n + 10000;
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

    // Color classes run strictly sequentially (see the arena-reuse comment below), but each
    // pair's healing pass (stage3_healing_ils_pass, Stage2_ILS.cpp) reads --stage3-ms's value
    // fresh off this global and treats it as its own full deadline. Left alone, that gives
    // every one of the (max_color+1) color classes the full requested budget back to back --
    // report 010 §0.10 measured 36.9s of real Stage 3 wall time against a 12s ask at Lazio (3
    // color classes from K4's edge-chromatic number). Rescale here so --stage3-ms means "Stage
    // 3 total", matching --stage2-ms/--stage5-ms's semantics: divide across classes, restore
    // the original value on every exit path (including exceptions) via RAII so a later stage
    // reusing this global isn't silently left with the rescaled figure.
    extern int g_stage3_time_budget_ms;
    struct BudgetRestore {
        int& budget;
        int original;
        ~BudgetRestore() { budget = original; }
    } budgetRestore{g_stage3_time_budget_ms, g_stage3_time_budget_ms};
    if (g_stage3_time_budget_ms > 0) {
        g_stage3_time_budget_ms = std::max(1, g_stage3_time_budget_ms / (max_color + 1));
    }

    // Pool of arenas, one per thread SLOT within a color class (indexed by t_idx, which
    // ranges 0..class_edges.size()-1 and restarts each class) -- reused across every color
    // class instead of a fresh ThreadArena constructed inside the lambda per pair. A fresh
    // arena at n=1,000,000 is ~110-230 MB to allocate and zero-fill (see
    // docs/reports/006_throughput_and_parallelism.md Phase 2.6); this repeated for every
    // chunk pair in every color class was a large, pure-overhead cost. Safe to reuse: color
    // classes run strictly sequentially (all of class c's threads are joined below before
    // class c+1's threads are spawned), so slot t_idx is never touched by two threads at
    // once, and each pair's own SA loop already resets doCount/undoCount/pendingDelta at the
    // top of its first iteration regardless of what a previous pair left behind.
    size_t max_class_size = 0;
    for (const auto& cc : color_classes) max_class_size = std::max(max_class_size, cc.size());
    std::vector<ThreadArena> arena_pool(max_class_size);
    for (auto& a : arena_pool) a.reserve_fixed_capacity(inst.n, max_possible_routes);

    for (int c = 0; c <= max_color; ++c) {
        std::vector<std::thread> threads;
        const auto& class_edges = color_classes[c];
        // Each thread writes to its own slot (indexed by t_idx), so no two threads ever
        // touch the same element concurrently -- summed single-threaded after join below.
        std::vector<Cost> classDeltas(class_edges.size(), 0);

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
            threads.emplace_back([&, pair_boundary, t1, t2, t_idx]() mutable {
                ThreadArena& arena = arena_pool[t_idx];
                SVCCache cache;
                cache.init(inst.n);
                extern int g_seed; // overridable via --seed; default 1337 preserves prior behavior exactly
                std::mt19937 rng(g_seed + t1 * 1000 + t2);
                std::cout << "Healing chunk pair (" << t1 << "," << t2 << ")" << std::endl;
                for (int cust : pair_boundary) cache.insert(cust);
                classDeltas[t_idx] = stage3_healing_ils_pass(globalSolution, arena, cache, inst, neighborLists, partitionInfo, pair_boundary, t1, t2, rng, &routeToChunk);
            });
        }

        for (auto& th : threads) {
            th.join();
        }

        // Safe to touch the shared scalar here: all threads in this color class have
        // joined, so this runs single-threaded before the next class's threads start.
        for (Cost d : classDeltas) {
            globalSolution.totalCost += d;
        }
    }
}
