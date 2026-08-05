#include "Stage2_ILS.hpp"
#include <algorithm>
#include <cmath>
#include <mutex>
#include <chrono>
#include <iostream>

namespace {
    static std::mutex route_creation_mutex;
    
    void update_route_info(Solution& sol, int route, const Instance& inst) {
        if (route == -1 || route >= sol.numRoutes) return;
        
        NodeId curr = sol.routeHead[route];
        int pos = 1;
        Cost current_load = 0;
        while (curr != 0) {
            sol.routePosition[curr] = pos++;
            current_load += inst.demand[curr];
            sol.cumLoad[curr] = current_load;
            curr = sol.succ[curr];
        }
    }
    
    void remove_customer(Solution& sol, NodeId c, ThreadArena& arena, const Instance& inst) {
        NodeId p = sol.pred[c];
        NodeId s = sol.succ[c];
        
        DoUndoEntry undo_entry;
        undo_entry.type = DoUndoEntry::INSERT;
        undo_entry.customer = c;
        undo_entry.prevPred = p; undo_entry.prevSucc = s;
        undo_entry.newPred = p; undo_entry.newSucc = s;
        undo_entry.prevRoute = sol.routeOf[c]; undo_entry.newRoute = sol.routeOf[c];
        
        Cost delta = dist(inst, p, s) - dist(inst, p, c) - dist(inst, c, s);
        undo_entry.costDelta = -delta; 
        arena.undoList[arena.undoCount++] = undo_entry;

        DoUndoEntry do_entry;
        do_entry.type = DoUndoEntry::REMOVE;
        do_entry.customer = c;
        do_entry.prevPred = p; do_entry.prevSucc = s;
        do_entry.newPred = p; do_entry.newSucc = s;
        do_entry.prevRoute = sol.routeOf[c]; do_entry.newRoute = -1;
        do_entry.costDelta = delta;
        
        arena.doList[arena.doCount++] = do_entry;
        arena.pendingDelta += delta;

        sol.succ[p] = s;
        sol.pred[s] = p;
        sol.routeLoad[sol.routeOf[c]] -= inst.demand[c];
        
        if (p == 0) sol.routeHead[sol.routeOf[c]] = s;
        if (s == 0) sol.routeTail[sol.routeOf[c]] = p;
        
        sol.routeOf[c] = -1;
        sol.pred[c] = 0; sol.succ[c] = 0;
    }

    void insert_customer(Solution& sol, NodeId c, NodeId p, NodeId s, int route, ThreadArena& arena, const Instance& inst) {
        DoUndoEntry undo_entry;
        undo_entry.type = DoUndoEntry::REMOVE;
        undo_entry.customer = c;
        undo_entry.prevPred = p; undo_entry.prevSucc = s;
        undo_entry.newPred = p; undo_entry.newSucc = s;
        undo_entry.prevRoute = route; undo_entry.newRoute = -1;
        
        Cost delta = dist(inst, p, c) + dist(inst, c, s) - dist(inst, p, s);
        undo_entry.costDelta = -delta; 
        arena.undoList[arena.undoCount++] = undo_entry;

        DoUndoEntry do_entry;
        do_entry.type = DoUndoEntry::INSERT;
        do_entry.customer = c;
        do_entry.prevPred = p; do_entry.prevSucc = s;
        do_entry.newPred = p; do_entry.newSucc = s;
        do_entry.prevRoute = -1; do_entry.newRoute = route;
        do_entry.costDelta = delta;
        
        arena.doList[arena.doCount++] = do_entry;
        arena.pendingDelta += delta;

        sol.succ[p] = c;
        sol.pred[c] = p;
        sol.succ[c] = s;
        sol.pred[s] = c;
        sol.routeOf[c] = route;
        sol.routeLoad[route] += inst.demand[c];
        
        if (p == 0) sol.routeHead[route] = c;
        if (s == 0) sol.routeTail[route] = c;
    }

    void apply_undo_list(Solution& sol, ThreadArena& arena, const Instance& inst, std::mutex* mtx = nullptr) {
        if (mtx) mtx->lock();
        
        // Track unique routes modified
        int modified_routes[10];
        int num_mod = 0;
        
        for (int i = arena.undoCount - 1; i >= 0; --i) {
            const auto& entry = arena.undoList[i];
            int route_changed = -1;
            if (entry.type == DoUndoEntry::INSERT) {
                NodeId c = entry.customer; NodeId p = entry.newPred; NodeId s = entry.newSucc; int route = entry.newRoute;
                sol.succ[p] = c; sol.pred[c] = p; sol.succ[c] = s; sol.pred[s] = c;
                sol.routeOf[c] = route; sol.routeLoad[route] += inst.demand[c];
                if (p == 0) sol.routeHead[route] = c;
                if (s == 0) sol.routeTail[route] = c;
                route_changed = route;
            } else {
                // Undoing an INSERT means we must REMOVE it
                NodeId c = entry.customer; int route = entry.prevRoute;
                NodeId p = sol.pred[c]; NodeId s = sol.succ[c];
                if (p != 0) sol.succ[p] = s;
                if (s != 0) sol.pred[s] = p;
                sol.pred[c] = 0; sol.succ[c] = 0;
                sol.routeLoad[route] -= inst.demand[c];
                if (p == 0) sol.routeHead[route] = s;
                if (s == 0) sol.routeTail[route] = p;
                sol.routeOf[c] = -1;
                route_changed = route;
            }
            
            bool found = false;
            for (int j = 0; j < num_mod; ++j) if (modified_routes[j] == route_changed) found = true;
            if (!found && num_mod < 10) modified_routes[num_mod++] = route_changed;
        }
        
        for (int j = 0; j < num_mod; ++j) {
            update_route_info(sol, modified_routes[j], inst);
        }
        
        arena.doCount = 0; arena.undoCount = 0; arena.pendingDelta = 0;
        if (mtx) mtx->unlock();
    }

    void invalidate_svc(SVCCache& cache, NodeId i, NodeId j, NodeId p_i, NodeId s_i, NodeId p_j, NodeId s_j) {
        if (i != 0) cache.insert(i);
        if (j != 0) cache.insert(j);
        if (p_i != 0) cache.insert(p_i);
        if (s_i != 0) cache.insert(s_i);
        if (p_j != 0) cache.insert(p_j);
        if (s_j != 0) cache.insert(s_j);
    }

    void ruin(Solution& sol, NodeId seed, ThreadArena& arena, SVCCache& cache, std::mt19937& rng, int chunkSize, const NeighborLists& granular_lists, const Instance& inst, std::mutex* mtx = nullptr, int t1 = -1, int t2 = -1, const std::vector<int>* routeToChunk = nullptr) {
        if (mtx) mtx->lock();
        arena.removed_count = 0;
        if (sol.routeOf[seed] == -1) {
            if (mtx) mtx->unlock();
            return;
        }
        if (routeToChunk && t1 != -1) {
            int c_r = (*routeToChunk)[sol.routeOf[seed]];
            if (c_r != t1 && c_r != t2) {
                if (mtx) mtx->unlock();
                return;
            }
        }
        
        NodeId current = seed;
        remove_customer(sol, current, arena, inst);
        cache.insert(current);
        arena.removed_customers[arena.removed_count++] = current;
        
        int walk_length = (int)std::ceil(std::log(chunkSize));
        if (walk_length < 1) walk_length = 1;
        
        for (int step = 1; step < walk_length; ++step) {
            std::vector<NodeId> candidates;
            NodeId p = arena.doList[arena.doCount - 1].prevPred;
            NodeId s = arena.doList[arena.doCount - 1].prevSucc;
            if (p != 0 && sol.routeOf[p] != -1) {
                if (!routeToChunk || t1 == -1 || ((*routeToChunk)[sol.routeOf[p]] == t1 || (*routeToChunk)[sol.routeOf[p]] == t2)) {
                    candidates.push_back(p);
                }
            }
            if (s != 0 && sol.routeOf[s] != -1) {
                if (!routeToChunk || t1 == -1 || ((*routeToChunk)[sol.routeOf[s]] == t1 || (*routeToChunk)[sol.routeOf[s]] == t2)) {
                    candidates.push_back(s);
                }
            }
            
            int k = std::min((int)granular_lists.nbr[current].size(), granular_lists.k);
            for (int i = 0; i < k; ++i) {
                NodeId j = granular_lists.nbr[current][i];
                if (sol.routeOf[j] != -1) {
                    if (!routeToChunk || t1 == -1 || ((*routeToChunk)[sol.routeOf[j]] == t1 || (*routeToChunk)[sol.routeOf[j]] == t2)) {
                        candidates.push_back(j);
                    }
                }
            }
            if (candidates.empty()) break;
            
            std::uniform_int_distribution<int> dist_c(0, candidates.size() - 1);
            NodeId next = candidates[dist_c(rng)];
            
            remove_customer(sol, next, arena, inst);
            cache.insert(next);
            arena.removed_customers[arena.removed_count++] = next;
            current = next;
        }
        if (mtx) mtx->unlock();
    }

    void recreate(Solution& sol, ThreadArena& arena, SVCCache& cache, const Instance& inst, const NeighborLists& granular_lists, std::mutex* mtx = nullptr, int t1 = -1, int t2 = -1, const std::vector<int>* routeToChunk = nullptr) {
        if (mtx) mtx->lock();
        std::sort(arena.removed_customers.begin(), arena.removed_customers.begin() + arena.removed_count,
            [&inst](NodeId a, NodeId b) { return inst.demand[a] > inst.demand[b]; });
            
        for (int i = 0; i < arena.removed_count; ++i) {
            NodeId c = arena.removed_customers[i];
            Cost bestDelta = 999999999;
            NodeId bestPred = 0, bestSucc = 0;
            int bestRoute = -1;
            
            int k = std::min((int)granular_lists.nbr[c].size(), granular_lists.k);
            for (int j_idx = 0; j_idx < k; ++j_idx) {
                NodeId j = granular_lists.nbr[c][j_idx];
                int r = sol.routeOf[j];
                if (r == -1) continue;
                if (routeToChunk && t1 != -1) {
                    if (r >= (int)routeToChunk->size()) continue;
                    int chunk = (*routeToChunk)[r];
                    if (chunk != t1 && chunk != t2) continue;
                }
                if (sol.routeLoad[r] + inst.demand[c] > inst.Q) continue;
                
                NodeId p = sol.pred[j];
                Cost delta1 = dist(inst, p, c) + dist(inst, c, j) - dist(inst, p, j);
                if (delta1 < bestDelta) { bestDelta = delta1; bestPred = p; bestSucc = j; bestRoute = r; }
                
                NodeId s = sol.succ[j];
                Cost delta2 = dist(inst, j, c) + dist(inst, c, s) - dist(inst, j, s);
                if (delta2 < bestDelta) { bestDelta = delta2; bestPred = j; bestSucc = s; bestRoute = r; }
            }
            
            if (bestRoute != -1) {
                insert_customer(sol, c, bestPred, bestSucc, bestRoute, arena, inst);
                cache.insert(c);
            } else {
                int r = -1;
                for (int empty_r = 0; empty_r < sol.numRoutes; ++empty_r) {
                    if (sol.routeHead[empty_r] == 0) {
                        // In chunked mode, verify the empty route belongs to the correct chunk
                        if (routeToChunk && t1 != -1) {
                            if (empty_r >= (int)routeToChunk->size()) continue;
                            int chunk = (*routeToChunk)[empty_r];
                            if (chunk != t1 && chunk != t2) continue;
                        }
                        r = empty_r;
                        break;
                    }
                }
                
                if (r == -1) {
                    r = sol.numRoutes++;
                    if (r >= (int)sol.routeHead.size()) {
                        sol.routeHead.resize(r + 100, 0); 
                        sol.routeTail.resize(r + 100, 0); 
                        sol.routeLoad.resize(r + 100, 0);
                    }
                }
                sol.routeLoad[r] = 0;
                // Important: if we're in stage3, we need to map the new route to the chunk so we can insert into it again!
                // But we don't mutate routeToChunk here as it's const. However, it's fine since we just added it.
                // Wait! To be safe, we just allow new routes if they are empty, but here it's empty so it's fine.
                insert_customer(sol, c, 0, 0, r, arena, inst);
                update_route_info(sol, r, inst);
                cache.insert(c);
            }
        }
        if (mtx) mtx->unlock();
    }

    Cost eval_relocate(const Solution& sol, const Instance& inst, NodeId i, NodeId j) {
        if (i == 0 || j == 0) return 0; // Forbid depot as primary operand
        if (i == j || sol.pred[i] == j || sol.succ[i] == j) return 0; // Adjacency double-count protection
        
        int r_i = sol.routeOf[i], r_j = sol.routeOf[j];
        if (r_i == -1 || r_j == -1) return 0;
        // Capacity short-circuit BEFORE distance lookups
        if (r_i != r_j && sol.routeLoad[r_j] + inst.demand[i] > inst.Q) return 0;
        
        NodeId p_i = sol.pred[i], s_i = sol.succ[i], s_j = sol.succ[j];
        return -dist(inst, p_i, i) - dist(inst, i, s_i) + dist(inst, p_i, s_i)
               -dist(inst, j, s_j) + dist(inst, j, i) + dist(inst, i, s_j);
    }
    
    Cost eval_swap(const Solution& sol, const Instance& inst, NodeId i, NodeId j) {
        if (i == 0 || j == 0 || i == j) return 0;
        int r_i = sol.routeOf[i], r_j = sol.routeOf[j];
        if (r_i == -1 || r_j == -1) return 0;
        
        // Capacity short-circuit BEFORE distance lookups
        if (r_i != r_j) {
            if (sol.routeLoad[r_i] - inst.demand[i] + inst.demand[j] > inst.Q) return 0;
            if (sol.routeLoad[r_j] - inst.demand[j] + inst.demand[i] > inst.Q) return 0;
        }
        
        NodeId p_i = sol.pred[i], s_i = sol.succ[i], p_j = sol.pred[j], s_j = sol.succ[j];
        
        // Explicit adjacency branches to prevent double-subtracting edges
        if (s_i == j) {
            return -dist(inst, p_i, i) - dist(inst, j, s_j)
                   +dist(inst, p_i, j) + dist(inst, i, s_j); // distance i,j cancels out
        } else if (s_j == i) {
            return -dist(inst, p_j, j) - dist(inst, i, s_i)
                   +dist(inst, p_j, i) + dist(inst, j, s_i); // distance j,i cancels out
        } else {
            return -dist(inst, p_i, i) - dist(inst, i, s_i) - dist(inst, p_j, j) - dist(inst, j, s_j)
                   +dist(inst, p_i, j) + dist(inst, j, s_i) + dist(inst, p_j, i) + dist(inst, i, s_j);
        }
    }

    bool is_before(const Solution& sol, NodeId a, NodeId b) {
        return sol.routePosition[a] < sol.routePosition[b];
    }

    Cost eval_2opt(const Solution& sol, const Instance& inst, NodeId i, NodeId j) {
        if (i == 0 || j == 0) return 0;
        int r_i = sol.routeOf[i], r_j = sol.routeOf[j];
        if (r_i == -1 || r_j == -1) return 0;
        if (r_i != r_j) return 0;
        
        // Adjacency double-count protection: reversing a segment of length 1 or 2 is handled by SWAP.
        if (i == j || sol.succ[i] == j || sol.succ[j] == i) return 0; 
        
        if (!is_before(sol, i, j)) std::swap(i, j);
        
        NodeId s_i = sol.succ[i], s_j = sol.succ[j];
        return -dist(inst, i, s_i) - dist(inst, j, s_j) + dist(inst, i, j) + dist(inst, s_i, s_j);
    }

    Cost eval_2opt_star(const Solution& sol, const Instance& inst, NodeId i, NodeId j) {
        if (i == 0 || j == 0) return 0; // Explicitly forbid depot intersections
        int r_i = sol.routeOf[i], r_j = sol.routeOf[j];
        if (r_i == -1 || r_j == -1) return 0;
        if (r_i == r_j) return 0;
        
        Cost load_tail_i = sol.routeLoad[r_i] - sol.cumLoad[i];
        Cost load_tail_j = sol.routeLoad[r_j] - sol.cumLoad[j];
        
        // Capacity short-circuit BEFORE distance lookups
        if (sol.routeLoad[r_i] - load_tail_i + load_tail_j > inst.Q) return 0;
        if (sol.routeLoad[r_j] - load_tail_j + load_tail_i > inst.Q) return 0;
        
        NodeId s_i = sol.succ[i], s_j = sol.succ[j];
        return -dist(inst, i, s_i) - dist(inst, j, s_j) + dist(inst, i, s_j) + dist(inst, j, s_i);
    }

    void get_top3_insertions(const Solution& sol, const Instance& inst, NodeId v, int target_route, Top3Insertions& top3) {
        top3.count = 0;
        NodeId p = 0;
        NodeId s = sol.routeHead[target_route];
        
        struct PosDelta { Cost delta; NodeId p; NodeId s; };
        std::vector<PosDelta> cands;
        
        while (true) {
            Cost delta = dist(inst, p, v) + dist(inst, v, s) - dist(inst, p, s);
            cands.push_back({delta, p, s});
            if (s == 0) break;
            p = s;
            s = sol.succ[s];
        }
        
        std::sort(cands.begin(), cands.end(), [](const PosDelta& a, const PosDelta& b) { return a.delta < b.delta; });
        
        top3.count = std::min((int)cands.size(), 3);
        for (int k = 0; k < top3.count; ++k) {
            top3.delta[k] = cands[k].delta;
            top3.pos_pred[k] = cands[k].p;
            top3.pos_succ[k] = cands[k].s;
        }
    }

    Cost eval_swap_star_dir(const Solution& sol, const Instance& inst, NodeId v, NodeId v_prime, const Top3Insertions& top3, NodeId& out_p, NodeId& out_s) {
        NodeId p_v_prime = sol.pred[v_prime], s_v_prime = sol.succ[v_prime];
        Cost best_delta = dist(inst, p_v_prime, v) + dist(inst, v, s_v_prime) - dist(inst, p_v_prime, s_v_prime);
        out_p = p_v_prime; out_s = s_v_prime;
        
        for (int k = 0; k < top3.count; ++k) {
            NodeId p = top3.pos_pred[k], s = top3.pos_succ[k];
            if (p != v_prime && s != v_prime) {
                if (top3.delta[k] < best_delta) {
                    best_delta = top3.delta[k];
                    out_p = p; out_s = s;
                }
            }
        }
        return best_delta;
    }

    Cost eval_swap_star(const Solution& sol, const Instance& inst, NodeId i, NodeId j, 
                        NodeId& out_p_i, NodeId& out_s_i, NodeId& out_p_j, NodeId& out_s_j) {
        if (i == 0 || j == 0) return 0; // Forbid depot
        int r_i = sol.routeOf[i], r_j = sol.routeOf[j];
        if (r_i == -1 || r_j == -1) return 0;
        if (r_i == r_j) return 0;
        
        // Capacity short-circuit BEFORE distance lookups
        if (sol.routeLoad[r_i] - inst.demand[i] + inst.demand[j] > inst.Q) return 0;
        if (sol.routeLoad[r_j] - inst.demand[j] + inst.demand[i] > inst.Q) return 0;
        
        Top3Insertions top3_i, top3_j;
        get_top3_insertions(sol, inst, i, r_j, top3_i);
        get_top3_insertions(sol, inst, j, r_i, top3_j);
        
        Cost ins_i = eval_swap_star_dir(sol, inst, i, j, top3_i, out_p_i, out_s_i);
        Cost ins_j = eval_swap_star_dir(sol, inst, j, i, top3_j, out_p_j, out_s_j);
        
        Cost rem_i = dist(inst, sol.pred[i], i) + dist(inst, i, sol.succ[i]) - dist(inst, sol.pred[i], sol.succ[i]);
        Cost rem_j = dist(inst, sol.pred[j], j) + dist(inst, j, sol.succ[j]) - dist(inst, sol.pred[j], sol.succ[j]);
        
        return ins_i + ins_j - rem_i - rem_j;
    }
    
    Cost eval_swap_star_fast(const Solution& sol, const Instance& inst, NodeId i, NodeId j, 
                             const Top3Insertions& top3_i, const Top3Insertions& top3_j,
                             NodeId& out_p_i, NodeId& out_s_i, NodeId& out_p_j, NodeId& out_s_j) {
        Cost ins_i = eval_swap_star_dir(sol, inst, i, j, top3_i, out_p_i, out_s_i);
        Cost ins_j = eval_swap_star_dir(sol, inst, j, i, top3_j, out_p_j, out_s_j);
        
        Cost rem_i = dist(inst, sol.pred[i], i) + dist(inst, i, sol.succ[i]) - dist(inst, sol.pred[i], sol.succ[i]);
        Cost rem_j = dist(inst, sol.pred[j], j) + dist(inst, j, sol.succ[j]) - dist(inst, sol.pred[j], sol.succ[j]);
        
        return ins_i + ins_j - rem_i - rem_j;
    }

    void apply_relocate(Solution& sol, ThreadArena& arena, const Instance& inst, NodeId i, NodeId j, SVCCache& cache) {
        NodeId p_i = sol.pred[i], s_i = sol.succ[i], s_j = sol.succ[j];
        int r_i = sol.routeOf[i];
        int r_j = sol.routeOf[j];
        remove_customer(sol, i, arena, inst);
        insert_customer(sol, i, j, s_j, r_j, arena, inst);
        update_route_info(sol, r_i, inst);
        if (r_i != r_j) update_route_info(sol, r_j, inst);
        invalidate_svc(cache, i, j, p_i, s_i, j, s_j);
    }
    
    void apply_swap(Solution& sol, ThreadArena& arena, const Instance& inst, NodeId i, NodeId j, SVCCache& cache) {
        NodeId p_i = sol.pred[i], s_i = sol.succ[i], p_j = sol.pred[j], s_j = sol.succ[j];
        int r_i = sol.routeOf[i], r_j = sol.routeOf[j];
        
        remove_customer(sol, i, arena, inst);
        remove_customer(sol, j, arena, inst);
        
        if (s_i == j) {
            insert_customer(sol, j, p_i, s_j, r_i, arena, inst);
            insert_customer(sol, i, j, s_j, r_j, arena, inst);
        } else if (s_j == i) {
            insert_customer(sol, i, p_j, s_i, r_j, arena, inst);
            insert_customer(sol, j, i, s_i, r_i, arena, inst);
        } else {
            insert_customer(sol, j, p_i, s_i, r_i, arena, inst);
            insert_customer(sol, i, p_j, s_j, r_j, arena, inst);
        }
        update_route_info(sol, r_i, inst);
        if (r_i != r_j) update_route_info(sol, r_j, inst);
        invalidate_svc(cache, i, j, p_i, s_i, p_j, s_j);
    }

    void apply_2opt(Solution& sol, ThreadArena& arena, const Instance& inst, NodeId i, NodeId j, SVCCache& cache) {
        if (!is_before(sol, i, j)) std::swap(i, j);
        
        std::vector<NodeId> seg;
        NodeId curr = sol.succ[i];
        while (curr != sol.succ[j]) { seg.push_back(curr); curr = sol.succ[curr]; }
        
        int route = sol.routeOf[i];
        for (NodeId v : seg) remove_customer(sol, v, arena, inst);
        
        NodeId insert_after = i;
        for (auto it = seg.rbegin(); it != seg.rend(); ++it) {
            NodeId v = *it;
            NodeId s = sol.succ[insert_after];
            insert_customer(sol, v, insert_after, s, route, arena, inst);
            insert_after = v;
            cache.insert(v);
        }
        update_route_info(sol, route, inst);
        cache.insert(i); cache.insert(j);
    }

    void apply_2opt_star(Solution& sol, ThreadArena& arena, const Instance& inst, NodeId i, NodeId j, SVCCache& cache) {
        int r_i = sol.routeOf[i], r_j = sol.routeOf[j];
        
        std::vector<NodeId> tail_i, tail_j;
        NodeId curr = sol.succ[i];
        while (curr != 0) { tail_i.push_back(curr); curr = sol.succ[curr]; }
        
        curr = sol.succ[j];
        while (curr != 0) { tail_j.push_back(curr); curr = sol.succ[curr]; }
        
        for (NodeId v : tail_i) remove_customer(sol, v, arena, inst);
        for (NodeId v : tail_j) remove_customer(sol, v, arena, inst);
        
        NodeId insert_after = j;
        for (NodeId v : tail_i) {
            NodeId s = sol.succ[insert_after];
            insert_customer(sol, v, insert_after, s, r_j, arena, inst);
            insert_after = v; cache.insert(v);
        }
        
        insert_after = i;
        for (NodeId v : tail_j) {
            NodeId s = sol.succ[insert_after];
            insert_customer(sol, v, insert_after, s, r_i, arena, inst);
            insert_after = v; cache.insert(v);
        }
        update_route_info(sol, r_i, inst);
        update_route_info(sol, r_j, inst);
        cache.insert(i); cache.insert(j);
    }
    
    void apply_swap_star(Solution& sol, ThreadArena& arena, const Instance& inst, NodeId i, NodeId j, 
                         NodeId p_i, NodeId s_i, NodeId p_j, NodeId s_j, SVCCache& cache) {
        NodeId orig_p_i = sol.pred[i], orig_s_i = sol.succ[i];
        NodeId orig_p_j = sol.pred[j], orig_s_j = sol.succ[j];
        int r_i = sol.routeOf[i], r_j = sol.routeOf[j];
        
        remove_customer(sol, i, arena, inst);
        remove_customer(sol, j, arena, inst);
        
        insert_customer(sol, i, p_i, s_i, r_j, arena, inst);
        insert_customer(sol, j, p_j, s_j, r_i, arena, inst);
        
        update_route_info(sol, r_i, inst);
        update_route_info(sol, r_j, inst);
        
        invalidate_svc(cache, i, j, orig_p_i, orig_s_i, orig_p_j, orig_s_j);
        cache.insert(p_i); cache.insert(s_i);
        cache.insert(p_j); cache.insert(s_j);
    }

    bool local_search(Solution& sol, ThreadArena& arena, SVCCache& cache, const Instance& inst, const NeighborLists& granular_lists, int chunkSize, std::mutex* mtx = nullptr, int t1 = -1, int t2 = -1, const std::vector<int>* routeToChunk = nullptr) {
        bool improved = false;
        int ls_iter = 0;
        
        while (cache.count > 0) {
            ls_iter++;
            NodeId i = cache.pop();
            
            if (sol.routeOf[i] == -1) continue;
            
            Cost bestDelta = 0;
            int bestOp = -1; // 0=Relocate, 1=Swap, 2=2-Opt, 3=2-Opt*, 4=Swap*
            NodeId best_j = 0;
            NodeId best_p_i = 0, best_s_i = 0, best_p_j = 0, best_s_j = 0;
            
            int k = std::min((int)granular_lists.nbr[i].size(), granular_lists.k);
            int r_i = sol.routeOf[i];
            if (routeToChunk && t1 != -1) {
                if (r_i >= (int)routeToChunk->size()) continue;
                int c_r_i = (*routeToChunk)[r_i];
                if (c_r_i != t1 && c_r_i != t2) continue;
            }
            
            // Step 1: Precompute top-3 insertions for SWAP*
            for (int j_idx = 0; j_idx < k; ++j_idx) {
                NodeId j = granular_lists.nbr[i][j_idx];
                int r_j = sol.routeOf[j];
                if (r_j == -1 || r_i == r_j) continue;
                if (routeToChunk && t1 != -1) {
                    if (r_j >= (int)routeToChunk->size()) continue;
                    int c_r_j = (*routeToChunk)[r_j];
                    if (c_r_j != t1 && c_r_j != t2) continue;
                }
                
                if (arena.route_visited_iter[r_j] != ls_iter) {
                    arena.route_visited_iter[r_j] = ls_iter;
                    get_top3_insertions(sol, inst, i, r_j, arena.top3_i_to_V[r_j]);
                }
                if (arena.node_visited_iter[j] != ls_iter) {
                    arena.node_visited_iter[j] = ls_iter;
                    get_top3_insertions(sol, inst, j, r_i, arena.top3_j_to_U[j]);
                }
            }
            
            for (int j_idx = 0; j_idx < k; ++j_idx) {
                NodeId j = granular_lists.nbr[i][j_idx];
                int r_j = sol.routeOf[j];
                if (r_j == -1) continue;
                if (routeToChunk && t1 != -1) {
                    if (r_j >= (int)routeToChunk->size()) continue;
                    int c_r_j = (*routeToChunk)[r_j];
                    if (c_r_j != t1 && c_r_j != t2) continue;
                }
                
                Cost delta_relocate = eval_relocate(sol, inst, i, j);
                if (delta_relocate < bestDelta) { bestDelta = delta_relocate; bestOp = 0; best_j = j; }
                
                Cost delta_swap = eval_swap(sol, inst, i, j);
                if (delta_swap < bestDelta) { bestDelta = delta_swap; bestOp = 1; best_j = j; }
                
                Cost delta_2opt = eval_2opt(sol, inst, i, j);
                if (delta_2opt < bestDelta) { bestDelta = delta_2opt; bestOp = 2; best_j = j; }
                
                Cost delta_2opt_star = eval_2opt_star(sol, inst, i, j);
                if (delta_2opt_star < bestDelta) { bestDelta = delta_2opt_star; bestOp = 3; best_j = j; }
                
                NodeId p_i, s_i, p_j, s_j;
                Cost delta_swap_star = 0;
                if (r_i != r_j) {
                    // Capacity short-circuit BEFORE distance lookups
                    if (sol.routeLoad[r_i] - inst.demand[i] + inst.demand[j] <= inst.Q &&
                        sol.routeLoad[r_j] - inst.demand[j] + inst.demand[i] <= inst.Q) {
                        delta_swap_star = eval_swap_star_fast(sol, inst, i, j, arena.top3_i_to_V[r_j], arena.top3_j_to_U[j], p_i, s_i, p_j, s_j);
                        if (delta_swap_star < bestDelta) { 
                            bestDelta = delta_swap_star; bestOp = 4; best_j = j; 
                            best_p_i = p_i; best_s_i = s_i; best_p_j = p_j; best_s_j = s_j;
                        }
                    }
                }
            }
            
            if (bestDelta < -1e-6) {
                if (mtx) mtx->lock(); // Lock acquired before changing sol!
                
                // Re-evaluate inside lock to prevent concurrency bugs
                Cost verify_delta = 0;
                if (bestOp == 0) verify_delta = eval_relocate(sol, inst, i, best_j);
                else if (bestOp == 1) verify_delta = eval_swap(sol, inst, i, best_j);
                else if (bestOp == 2) verify_delta = eval_2opt(sol, inst, i, best_j);
                else if (bestOp == 3) verify_delta = eval_2opt_star(sol, inst, i, best_j);
                else if (bestOp == 4) verify_delta = eval_swap_star(sol, inst, i, best_j, best_p_i, best_s_i, best_p_j, best_s_j);
                
                if (verify_delta < -1e-6) {
                    int old_r_i = sol.routeOf[i];
                    int old_r_j = best_j != 0 ? sol.routeOf[best_j] : -1;
                    
                    if (bestOp == 0) apply_relocate(sol, arena, inst, i, best_j, cache);
                    else if (bestOp == 1) apply_swap(sol, arena, inst, i, best_j, cache);
                    else if (bestOp == 2) apply_2opt(sol, arena, inst, i, best_j, cache);
                    else if (bestOp == 3) apply_2opt_star(sol, arena, inst, i, best_j, cache);
                    else if (bestOp == 4) apply_swap_star(sol, arena, inst, i, best_j, best_p_i, best_s_i, best_p_j, best_s_j, cache);
                    
                    if (old_r_i != -1) update_route_info(sol, old_r_i, inst);
                    if (old_r_j != -1 && old_r_j != old_r_i) update_route_info(sol, old_r_j, inst);
                    
                    improved = true;
                }
                
                if (mtx) mtx->unlock();
            }
        }
        return improved;
    }

    bool accept_delta(Cost delta, double temperature, std::mt19937& rng) {
        if (delta <= 0) return true;
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        return dist(rng) < std::exp(-(double)delta / temperature);
    }
}

Solution stage2_ils(Solution sol, ThreadArena& arena, SVCCache& cache, 
                    const Instance& inst, const Stage0Result& partitionInfo, 
                    const NeighborLists& neighborLists, int chunkId, std::mt19937& rng) {
    int chunkSize = partitionInfo.globalId[chunkId].size() - 1;
    cache.init(inst.n);
    cache.clear();
    
    // Build local granular list from the global one
    NeighborLists local_granular_lists;
    local_granular_lists.k = neighborLists.k;
    local_granular_lists.nbr.assign(inst.n + 1, std::vector<NodeId>());
    for (int i = 1; i <= chunkSize; ++i) {
        NodeId global_i = partitionInfo.globalId[chunkId][i];
        for (NodeId global_j : neighborLists.nbr[global_i]) {
            if (partitionInfo.chunkOf[global_j] == chunkId) {
                local_granular_lists.nbr[global_i].push_back(global_j);
            }
        }
    }

    double avg_arc_cost_estimate = 100.0;
    double T0 = 0.1 * avg_arc_cost_estimate; 
    if (T0 < 1e-6) T0 = 1.0;
    double Tf = 0.01 * T0;
    int max_iterations = chunkSize * 50; 
    double cooling_rate = std::pow(Tf / T0, 1.0 / max_iterations);
    double temperature = T0;
    
    Solution bestSol = sol;

    for (int iter = 0; iter < max_iterations; ++iter) {
        arena.doCount = 0;
        arena.undoCount = 0;
        arena.pendingDelta = 0;
        
        int prevNumRoutes = sol.numRoutes;
        
        std::uniform_int_distribution<int> dist_cust(1, chunkSize);
        NodeId seed_cust = dist_cust(rng);
        ruin(sol, seed_cust, arena, cache, rng, chunkSize, local_granular_lists, inst);
        
        recreate(sol, arena, cache, inst, local_granular_lists);
        
        for (int r = 0; r < sol.numRoutes; ++r) {
            update_route_info(sol, r, inst);
        }
        
        bool local_search_improved = true;
        while(local_search_improved) {
            local_search_improved = local_search(sol, arena, cache, inst, local_granular_lists, chunkSize);
        }
        
        Cost delta = arena.pendingDelta;
        if (accept_delta(delta, temperature, rng)) {
            sol.totalCost += delta;
            if (sol.totalCost < bestSol.totalCost) {
                bestSol = sol;
            }
        } else {
            apply_undo_list(sol, arena, inst);
            sol.numRoutes = prevNumRoutes;
            for (int r = 0; r < sol.numRoutes; ++r) {
                update_route_info(sol, r, inst);
            }
        }
        
        temperature *= cooling_rate;
    }
    
    return bestSol;
}

void stage3_healing_ils_pass(Solution& globalSolution, ThreadArena& arena, SVCCache& cache, 
                             const Instance& inst, const NeighborLists& neighborLists, 
                             const Stage0Result& partitionInfo,
                             const std::vector<int>& boundaryList, 
                             int t1, int t2, std::mt19937& rng,
                             const std::vector<int>* routeToChunk = nullptr) {
    if (boundaryList.empty()) return;
    
    cache.init(inst.n);
    cache.clear();
    
    for (int c : boundaryList) {
        cache.insert(c);
    }
    
    NeighborLists local_granular_lists;
    local_granular_lists.k = neighborLists.k;
    local_granular_lists.nbr.assign(inst.n + 1, std::vector<NodeId>());
    for (int i : boundaryList) {
        for (NodeId j : neighborLists.nbr[i]) {
            int c_j = partitionInfo.chunkOf[j];
            if (c_j == t1 || c_j == t2) {
                local_granular_lists.nbr[i].push_back(j);
            }
        }
    }
    
    double avg_arc_cost_estimate = 100.0;
    double T0 = 0.1 * avg_arc_cost_estimate; 
    if (T0 < 1e-6) T0 = 1.0;
    double Tf = 0.01 * T0;
    int max_iterations = std::min(1000, (int)boundaryList.size() * 50); 
    if (max_iterations == 0) return;
    double cooling_rate = std::pow(Tf / T0, 1.0 / max_iterations);
    double temperature = T0;
    
    for (int iter = 0; iter < max_iterations; ++iter) {
        if (iter >= max_iterations - 50) temperature = 0.0; // Force greedy descent at the end
        auto t_start = std::chrono::high_resolution_clock::now();
        arena.doCount = 0;
        arena.undoCount = 0;
        arena.pendingDelta = 0;
        int prevNumRoutes = globalSolution.numRoutes;
        
        std::uniform_int_distribution<int> dist_cust(0, boundaryList.size() - 1);
        NodeId seed_cust = boundaryList[dist_cust(rng)];
        int virtual_chunk_size = boundaryList.size(); // for log-scaled ruin walk
        ruin(globalSolution, seed_cust, arena, cache, rng, virtual_chunk_size, local_granular_lists, inst, &route_creation_mutex, t1, t2, routeToChunk);
        
        auto t_ruin = std::chrono::high_resolution_clock::now();
        
        recreate(globalSolution, arena, cache, inst, local_granular_lists, &route_creation_mutex, t1, t2, routeToChunk);
        
        for (int r = 0; r < globalSolution.numRoutes; ++r) {
            if (!routeToChunk || t1 == -1 || ((*routeToChunk)[r] == t1 || (*routeToChunk)[r] == t2)) {
                update_route_info(globalSolution, r, inst);
            }
        }
        
        auto t_recr = std::chrono::high_resolution_clock::now();
        bool local_search_improved = true;
        int ls_loops = 0;
        while(local_search_improved) {
            local_search_improved = local_search(globalSolution, arena, cache, inst, local_granular_lists, inst.n, &route_creation_mutex, t1, t2, routeToChunk);
            ls_loops++;
            if (ls_loops > 1000) {
                std::cout << "[HANG WARNING] local_search stuck in loop! iter: " << iter << " ls_loops: " << ls_loops << std::endl;
            }
        }
        
        auto t_end = std::chrono::high_resolution_clock::now();
        
        double ruin_ms = std::chrono::duration<double, std::milli>(t_ruin - t_start).count();
        double recr_ms = std::chrono::duration<double, std::milli>(t_recr - t_ruin).count();
        double ls_ms = std::chrono::duration<double, std::milli>(t_end - t_recr).count();
        double total_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        
        if (total_ms > 100.0) {
            std::cout << "[PROFILE] Pair(" << t1 << "," << t2 << ") Iter " << iter 
                      << " Total=" << total_ms << "ms (Ruin=" << ruin_ms 
                      << ", Recr=" << recr_ms << ", LS=" << ls_ms << ") loops=" << ls_loops << std::endl;
        }
        
        Cost delta = arena.pendingDelta;
        if (accept_delta(delta, temperature, rng)) {
            // accepted
        } else {
            apply_undo_list(globalSolution, arena, inst, &route_creation_mutex);
            globalSolution.numRoutes = prevNumRoutes;
            for (int r = 0; r < globalSolution.numRoutes; ++r) {
                if (!routeToChunk || t1 == -1 || ((*routeToChunk)[r] == t1 || (*routeToChunk)[r] == t2)) {
                    update_route_info(globalSolution, r, inst);
                }
            }
        }
        
        temperature *= cooling_rate;
    }
}

void stage5_serial_polish(Solution& globalSolution, ThreadArena& arena, const Instance& inst, const NeighborLists& neighborLists) {
    if (globalSolution.numRoutes == 0) return;
    
    SVCCache cache;
    cache.init(inst.n);
    cache.clear();
    for (int i = 1; i <= inst.n; ++i) {
        cache.insert(i);
    }
    
    std::mt19937 rng(424242);
    
    double avg_arc_cost_estimate = 100.0;
    double T0 = 0.1 * avg_arc_cost_estimate; 
    if (T0 < 1e-6) T0 = 1.0;
    double Tf = 0.01 * T0;
    int max_iterations = 500;
    int stagnation_limit = 150;
    int stagnation = 0;
    double cooling_rate = std::pow(Tf / T0, 1.0 / max_iterations);
    double temperature = T0;
    
    Solution bestSol = globalSolution;
    
    for (int iter = 0; iter < max_iterations; ++iter) {
        arena.doCount = 0;
        arena.undoCount = 0;
        arena.pendingDelta = 0;
        
        int prevNumRoutes = globalSolution.numRoutes;
        
        std::uniform_int_distribution<int> dist_cust(1, inst.n);
        NodeId seed_cust = dist_cust(rng);
        
        ruin(globalSolution, seed_cust, arena, cache, rng, inst.n, neighborLists, inst, nullptr);
        
        recreate(globalSolution, arena, cache, inst, neighborLists, nullptr);
        
        bool local_search_improved = true;
        while(local_search_improved) {
            local_search_improved = local_search(globalSolution, arena, cache, inst, neighborLists, inst.n, nullptr);
        }
        
        Cost delta = arena.pendingDelta;
        if (accept_delta(delta, temperature, rng)) {
            globalSolution.totalCost += delta;
            if (globalSolution.totalCost < bestSol.totalCost - 1e-6) {
                bestSol = globalSolution;
                stagnation = 0;
            } else {
                stagnation++;
            }
        } else {
            apply_undo_list(globalSolution, arena, inst, nullptr);
            globalSolution.numRoutes = prevNumRoutes;
            for (int r = 0; r < globalSolution.numRoutes; ++r) {
                update_route_info(globalSolution, r, inst);
            }
            stagnation++;
        }
        
        if (stagnation >= stagnation_limit) break;
        
        temperature *= cooling_rate;
    }
    globalSolution = bestSol;
}
