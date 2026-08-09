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

    // Copies everything except routePosition/cumLoad, which are pure derived caches (see
    // update_route_info) fully reconstructible from pred/succ/routeHead. The per-improvement
    // "new best" snapshot in stage2_ils/stage5_serial_polish used to be a full `bestSol =
    // sol` (24 bytes/node at scale -- see docs/reports/006_throughput_and_parallelism.md
    // Phase 2.3), which is one of the most expensive things in the loop since early in the
    // anneal nearly every iteration is a new best. Dropping the two derived fields here cuts
    // that to 12 bytes/node; finalize_solution_derived_fields (below) regenerates them once,
    // authoritatively, after the loop ends -- byte-identical result by construction, since
    // update_route_info is the same function that would have produced whatever values a full
    // copy carried along at each snapshot anyway.
    void snapshot_essential(Solution& dst, const Solution& src) {
        dst.pred = src.pred;
        dst.succ = src.succ;
        dst.routeOf = src.routeOf;
        dst.routeHead = src.routeHead;
        dst.routeTail = src.routeTail;
        dst.routeLoad = src.routeLoad;
        dst.numRoutes = src.numRoutes;
        dst.totalCost = src.totalCost;
    }

    void finalize_solution_derived_fields(Solution& sol, const Instance& inst) {
        for (int r = 0; r < sol.numRoutes; ++r) update_route_info(sol, r, inst);
    }

    // Refreshes routePosition/cumLoad for exactly the routes touched by this SA iteration's
    // do/undo log so far, instead of an unconditional full O(numRoutes) sweep -- a ruin +
    // recreate + local_search cascade typically touches a handful of routes, not all of
    // them, and at large N (Lazio: ~1,000,000 nodes) a per-iteration full-graph rescan
    // dominates actual search time. Uses the same generation-marker scratch buffer as
    // apply_undo_list's own rescan (Phase 1.2) -- see docs/reports/005_cost_optimization.md
    // Phase 5. doList and undoList are always pushed in lockstep (remove_customer/
    // insert_customer each push exactly one of each per call), so either list identifies the
    // same touched-route set; this reads doList since it's meaningful even mid-iteration,
    // before any rollback decision has been made.
    void rescan_touched_routes(Solution& sol, ThreadArena& arena, const Instance& inst) {
        arena.modified_routes_gen++;
        int gen = arena.modified_routes_gen;
        int num_mod = 0;
        int gen_cap = (int)arena.route_modified_gen.size();

        for (int i = 0; i < arena.doCount; ++i) {
            const auto& entry = arena.doList[i];
            int route_changed = (entry.type == DoUndoEntry::REMOVE) ? entry.prevRoute : entry.newRoute;
            if (route_changed != -1 && route_changed < gen_cap &&
                arena.route_modified_gen[route_changed] != gen) {
                arena.route_modified_gen[route_changed] = gen;
                if (num_mod < (int)arena.modified_routes_list.size()) {
                    arena.modified_routes_list[num_mod++] = route_changed;
                }
            }
        }
        for (int j = 0; j < num_mod; ++j) {
            update_route_info(sol, arena.modified_routes_list[j], inst);
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
        // Bounds-checked: doList/undoList are sized generously (up to 500,000 entries,
        // ThreadArena.hpp) but not unboundedly, and a corrupted/cyclic route (e.g. via the
        // stale-routePosition failure mode Phase 1.2 fixes) could otherwise drive an
        // out-of-bounds write here -- confirmed as a real access-violation crash during
        // Tier-1 stress testing (docs/reports/005_cost_optimization.md Phase 1.5). Dropping
        // the log entry when full still lets the actual move proceed (the arena being full
        // at all indicates something else is already badly wrong), but a crash is strictly
        // worse than a warning, so this fails loud instead of silent.
        if (arena.undoCount < (int)arena.undoList.size()) {
            arena.undoList[arena.undoCount++] = undo_entry;
        } else {
            static thread_local bool warned = false;
            if (!warned) { std::cout << "[WARNING] undoList arena exhausted -- rollback will be incomplete" << std::endl; warned = true; }
        }

        DoUndoEntry do_entry;
        do_entry.type = DoUndoEntry::REMOVE;
        do_entry.customer = c;
        do_entry.prevPred = p; do_entry.prevSucc = s;
        do_entry.newPred = p; do_entry.newSucc = s;
        do_entry.prevRoute = sol.routeOf[c]; do_entry.newRoute = -1;
        do_entry.costDelta = delta;

        if (arena.doCount < (int)arena.doList.size()) {
            arena.doList[arena.doCount++] = do_entry;
        }
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
        if (arena.undoCount < (int)arena.undoList.size()) {
            arena.undoList[arena.undoCount++] = undo_entry;
        } else {
            static thread_local bool warned = false;
            if (!warned) { std::cout << "[WARNING] undoList arena exhausted -- rollback will be incomplete" << std::endl; warned = true; }
        }

        DoUndoEntry do_entry;
        do_entry.type = DoUndoEntry::INSERT;
        do_entry.customer = c;
        do_entry.prevPred = p; do_entry.prevSucc = s;
        do_entry.newPred = p; do_entry.newSucc = s;
        do_entry.prevRoute = -1; do_entry.newRoute = route;
        do_entry.costDelta = delta;

        if (arena.doCount < (int)arena.doList.size()) {
            arena.doList[arena.doCount++] = do_entry;
        }
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

        for (int i = arena.undoCount - 1; i >= 0; --i) {
            const auto& entry = arena.undoList[i];
            if (entry.type == DoUndoEntry::INSERT) {
                NodeId c = entry.customer; NodeId p = entry.newPred; NodeId s = entry.newSucc; int route = entry.newRoute;
                sol.succ[p] = c; sol.pred[c] = p; sol.succ[c] = s; sol.pred[s] = c;
                sol.routeOf[c] = route; sol.routeLoad[route] += inst.demand[c];
                if (p == 0) sol.routeHead[route] = c;
                if (s == 0) sol.routeTail[route] = c;
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
            }
        }

        // doList is still intact here (cleared below) and identifies exactly the same
        // touched-route set undoList does -- see rescan_touched_routes's comment.
        rescan_touched_routes(sol, arena, inst);

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

    // Was a std::vector<PosDelta> + push_back over the entire route + std::sort, just to
    // keep the best 3 -- called up to 60x per local_search node pop, the innermost loop of
    // the whole solver (docs/reports/006_throughput_and_parallelism.md Phase 2.2). This is a
    // zero-allocation, no-sort running top-3 (insertion into a length<=3 sorted array) that
    // visits every candidate exactly once. Uses strict '<' throughout so an earlier-visited
    // position wins any exact delta tie -- matching what a *stable* sort would have done,
    // which std::sort itself never actually guaranteed.
    void get_top3_insertions(const Solution& sol, const Instance& inst, NodeId v, int target_route, Top3Insertions& top3) {
        top3.count = 0;
        NodeId p = 0;
        NodeId s = sol.routeHead[target_route];

        while (true) {
            Cost delta = dist(inst, p, v) + dist(inst, v, s) - dist(inst, p, s);
            if (top3.count < 3 || delta < top3.delta[2]) {
                int pos = std::min(top3.count, 2);
                while (pos > 0 && top3.delta[pos - 1] > delta) {
                    top3.delta[pos] = top3.delta[pos - 1];
                    top3.pos_pred[pos] = top3.pos_pred[pos - 1];
                    top3.pos_succ[pos] = top3.pos_succ[pos - 1];
                    --pos;
                }
                top3.delta[pos] = delta;
                top3.pos_pred[pos] = p;
                top3.pos_succ[pos] = s;
                if (top3.count < 3) top3.count++;
            }
            if (s == 0) break;
            p = s;
            s = sol.succ[s];
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
                // route_visited_iter/top3_i_to_V are route-indexed (ThreadArena.hpp); the
                // routeToChunk size check below already implies this bound when routeToChunk
                // is set (Stage 3, both sized to the same 2*inst.n+10000), but Stage 5 has no
                // routeToChunk at all -- guard unconditionally rather than rely on the sizes
                // happening to line up (docs/reports/006_throughput_and_parallelism.md Phase 4.1).
                if (r_j >= (int)arena.route_visited_iter.size()) continue;
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
                // top3_i_to_V is route-indexed (ThreadArena.hpp) -- see the identical guard
                // and comment on the precompute loop above (Phase 4.1).
                if (r_i != r_j && r_j < (int)arena.top3_i_to_V.size()) {
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

    // SVCCache is a fixed-size (CAPACITY=50) ring buffer that evicts oldest-on-overflow
    // (ThreadArena.hpp), so a loop that inserts every node in a large instance and then
    // starts searching only actually queues the last 50 ids inserted -- the earlier ones are
    // silently evicted before local_search ever pops them. stage5_serial_polish and
    // stage3_healing_ils_pass both used to do exactly this trying to seed a full sweep over
    // (respectively) every customer or every boundary customer. This runs the same
    // local-search-to-convergence loop used elsewhere, but in batches of CAPACITY nodes at a
    // time, so every node in `nodes` genuinely gets a turn instead of only the last 50 --
    // see docs/reports/005_cost_optimization.md Phase 1.4.
    //
    // Returns the total accumulated cost delta (always <= 0, since local_search only ever
    // applies strictly-improving moves) via arena.pendingDelta, which -- unlike the main SA
    // loops -- nothing else resets or applies here: local_search's apply_* path mutates
    // pred/succ/routeOf/etc immediately and unconditionally, but leaves sol.totalCost itself
    // untouched, so the caller MUST fold this return value into sol.totalCost (directly for
    // single-threaded Stage 5, or via the same acceptedDelta-summed-after-join pattern
    // Stage 3 already uses for every other cost change, to avoid a shared-scalar race) --
    // omitting that step desyncs totalCost from the real route costs, caught by verifier.py
    // as a "reported cost doesn't match recomputed cost" failure.
    //
    // `deadline`: checked once per batch (every CAPACITY=50 nodes), not per node -- this was
    // completely unbounded until docs/reports/006_throughput_and_parallelism.md Phase 1.1.
    // Both call sites used to run this BEFORE their stage's stageStart was even captured, so
    // it sat entirely outside --stage3-ms/--stage5-ms; report 005 measured this directly as
    // "Stage 4+5 took 56s against a nominal 20s budget". Default (time_point::max()) keeps
    // legacy/iteration-count mode's existing unbounded behavior unchanged.
    Cost full_sweep_local_search(Solution& sol, ThreadArena& arena, SVCCache& cache, const Instance& inst,
                                  const NeighborLists& granular_lists, int chunkSize,
                                  const std::vector<int>& nodes, std::mutex* mtx = nullptr,
                                  int t1 = -1, int t2 = -1, const std::vector<int>* routeToChunk = nullptr,
                                  std::chrono::steady_clock::time_point deadline = std::chrono::steady_clock::time_point::max()) {
        arena.pendingDelta = 0;
        size_t idx = 0;
        while (idx < nodes.size()) {
            if (std::chrono::steady_clock::now() >= deadline) break;
            size_t batch_end = std::min(idx + (size_t)SVCCache::CAPACITY, nodes.size());
            cache.clear();
            for (size_t k = idx; k < batch_end; ++k) cache.insert(nodes[k]);
            bool improved = true;
            while (improved) {
                improved = local_search(sol, arena, cache, inst, granular_lists, chunkSize, mtx, t1, t2, routeToChunk);
            }
            idx = batch_end;
        }
        return arena.pendingDelta;
    }

    bool accept_delta(Cost delta, double temperature, std::mt19937& rng) {
        if (delta <= 0) return true;
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        return dist(rng) < std::exp(-(double)delta / temperature);
    }
}

Solution stage2_ils(Solution sol, ThreadArena& arena, SVCCache& cache,
                    const Instance& inst, const Stage0Result& partitionInfo,
                    const NeighborLists& neighborLists, int chunkId, std::mt19937& rng,
                    int* out_iterations_completed) {
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

    // Instance-scaled T0 (was a hardcoded 100.0 regardless of instance -- fine at N=2,000's
    // small coordinate range but ~11x too cold at Valle-D-Aosta and ~32x too cold at Lazio,
    // where exp(-delta/T) collapses to ~0 for any typical worsening move, degenerating the
    // SA into pure hill-climbing exactly at the scales report 004 measured the cost gap on.
    // See docs/reports/005_cost_optimization.md Phase 1.3; mirrors FILO2's own
    // T0 = factor * mean-arc-cost rule (baselines/filo2/main.cpp).
    double avg_arc_cost_estimate = partitionInfo.medianKnnEdgeLen;
    double T0 = 0.1 * avg_arc_cost_estimate;
    if (T0 < 1e-6) T0 = 1.0;
    double Tf = 0.01 * T0;
    // Iteration budget is per-node-of-the-FULL-instance, not per-node-of-this-chunk: giving
    // each chunk only chunkSize*50 iterations meant P=4 threads did 4x less absolute search
    // than P=1, which is why the old default was fast but not a fair use of the parallelism
    // headroom (see docs/reports/001_p1_p4_filo2_baseline.md). Each thread now gets the same
    // absolute budget P=1 spends on the whole graph, spending the parallelism dividend on
    // search instead of pure idle time.
    extern int g_iters_per_node; // overridable via --iters-per-node, default 50
    extern int g_max_iterations_override; // overridable via --max-iterations (absolute, per thread)
    extern int g_stage2_time_budget_ms; // overridable via --stage2-ms; >0 switches to time-budget mode
    // Time-budget mode (see docs/reports/004_time_budget_scheduling.md) drives the cooling
    // schedule from elapsed-time fraction instead of iteration fraction -- same pattern as
    // FILO2's TimeBasedSimulatedAnnealing (baselines/filo2/opt/SimulatedAnnealing.hpp). This
    // is what lets the same -p/config choices behave sensibly whether N is 2,000 or
    // 1,000,000 without a hand-picked --max-iterations per instance. The legacy
    // iteration-count path is kept for exact backward compatibility when no time budget is set.
    bool useTimeBudget = g_stage2_time_budget_ms > 0;
    int max_iterations = g_max_iterations_override > 0 ? g_max_iterations_override : inst.n * g_iters_per_node;
    double cooling_rate = useTimeBudget ? 1.0 : std::pow(Tf / T0, 1.0 / max_iterations);
    double temperature = T0;

    Solution bestSol = sol;

    int iter = 0;
    auto stageStart = std::chrono::steady_clock::now();
    for (; useTimeBudget || iter < max_iterations; ++iter) {
        if (useTimeBudget) {
            double elapsed_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - stageStart).count();
            if (elapsed_ms >= g_stage2_time_budget_ms) break;
            temperature = T0 * std::pow(Tf / T0, elapsed_ms / g_stage2_time_budget_ms);
        }

        arena.doCount = 0;
        arena.undoCount = 0;
        arena.pendingDelta = 0;

        int prevNumRoutes = sol.numRoutes;

        // seed_cust must be a global NodeId (routeOf/pred/succ are all global-indexed --
        // see Stage1_Construction.cpp), not a local 1..chunkSize count. Drawing the local
        // count directly and using it as if it were global (the previous behavior) means
        // routeOf[seed_cust] almost always misses this chunk's nodes -- ruin() no-ops
        // immediately (docs/reports/005_cost_optimization.md Phase 1.1) for roughly
        // (P-1)/P of all Stage 2 iterations, and even on a hit only ever seeds from global
        // ids 1..chunkSize, so most of a chunk's actual customers were never reachable as a
        // seed at all. partitionInfo.globalId[chunkId] is exactly the local->global map
        // Stage 1 used to build this chunk's solution in the first place.
        std::uniform_int_distribution<int> dist_cust(1, chunkSize);
        NodeId seed_cust = partitionInfo.globalId[chunkId][dist_cust(rng)];
        ruin(sol, seed_cust, arena, cache, rng, chunkSize, local_granular_lists, inst);

        recreate(sol, arena, cache, inst, local_granular_lists);

        // Only rescan routes ruin/recreate actually touched (see rescan_touched_routes,
        // docs/reports/005_cost_optimization.md Phase 5) -- chunkSize-bounded here, so less
        // urgent than Stage 5's full-N version, but cheap and keeps the pattern consistent.
        rescan_touched_routes(sol, arena, inst);

        bool local_search_improved = true;
        while(local_search_improved) {
            local_search_improved = local_search(sol, arena, cache, inst, local_granular_lists, chunkSize);
        }

        Cost delta = arena.pendingDelta;
        if (accept_delta(delta, temperature, rng)) {
            sol.totalCost += delta;
            if (sol.totalCost < bestSol.totalCost) {
                snapshot_essential(bestSol, sol);
            }
        } else {
            // apply_undo_list already rescans every route it touched (Phase 1.2/5) -- no
            // separate full-route rescan needed here anymore.
            apply_undo_list(sol, arena, inst);
            sol.numRoutes = prevNumRoutes;
        }

        if (!useTimeBudget) {
            temperature *= cooling_rate;
        }
    }
    finalize_solution_derived_fields(bestSol, inst);

    if (out_iterations_completed) *out_iterations_completed = iter;
    return bestSol;
}

Cost stage3_healing_ils_pass(Solution& globalSolution, ThreadArena& arena, SVCCache& cache,
                             const Instance& inst, const NeighborLists& neighborLists,
                             const Stage0Result& partitionInfo,
                             const std::vector<int>& boundaryList,
                             int t1, int t2, std::mt19937& rng,
                             const std::vector<int>* routeToChunk = nullptr) {
    if (boundaryList.empty()) return 0;
    Cost acceptedDelta = 0;
    
    cache.init(inst.n);
    cache.clear();

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

    // Refresh routePosition/cumLoad for this pair's routes before the full sweep below --
    // full_sweep_local_search is the first thing in this pass to call local_search (which
    // depends on both being current, e.g. eval_2opt_star's capacity check reads cumLoad),
    // and unlike the per-iteration refreshes further down this function, nothing has
    // necessarily just rebuilt them fresh yet. Matches the existing (lock-free -- routes
    // labeled t1/t2 are this thread's exclusively within this color class, per the
    // graph-coloring disjointness guarantee) refresh convention used later in this function.
    for (int r = 0; r < globalSolution.numRoutes; ++r) {
        if (!routeToChunk || t1 == -1 || ((*routeToChunk)[r] == t1 || (*routeToChunk)[r] == t2)) {
            update_route_info(globalSolution, r, inst);
        }
    }

    // See the identical comment in stage2_ils above (docs/reports/005_cost_optimization.md
    // Phase 1.3) -- same instance-scaling fix, same rationale.
    double avg_arc_cost_estimate = partitionInfo.medianKnnEdgeLen;
    double T0 = 0.1 * avg_arc_cost_estimate;
    if (T0 < 1e-6) T0 = 1.0;
    double Tf = 0.01 * T0;
    extern int g_stage3_time_budget_ms; // overridable via --stage3-ms; >0 switches to time-budget mode
    bool useTimeBudget = g_stage3_time_budget_ms > 0;
    int max_iterations = std::min(1000, (int)boundaryList.size() * 50);
    if (!useTimeBudget && max_iterations == 0) return 0;
    double cooling_rate = useTimeBudget ? 1.0 : std::pow(Tf / T0, 1.0 / max_iterations);
    double temperature = T0;

    // stageStart captured here, before the full sweep, so the sweep's deadline (Phase 1.1)
    // and the main loop's own elapsed-time check below share one clock and one budget --
    // time the sweep spends is time the main loop below has less of, not extra on top.
    auto stageStart = std::chrono::steady_clock::now();
    auto sweepDeadline = useTimeBudget
        ? stageStart + std::chrono::milliseconds(g_stage3_time_budget_ms)
        : std::chrono::steady_clock::time_point::max();

    // Real full-sweep local-search descent over every boundary customer before the ILS loop
    // starts (see full_sweep_local_search's comment) -- the previous "insert every boundary
    // customer" loop here silently only queued the last 50 ids (SVCCache's ring-buffer
    // capacity) once local_granular_lists existed, this runs the descent for real.
    // Folded into acceptedDelta (not globalSolution.totalCost directly): this runs
    // concurrently with other chunk-pair threads sharing globalSolution, and acceptedDelta
    // is already the established race-free mechanism -- summed into totalCost once,
    // single-threaded, after this color class's threads all join (Stage3_MergeHealing.cpp).
    acceptedDelta += full_sweep_local_search(globalSolution, arena, cache, inst, local_granular_lists, inst.n,
                                              boundaryList, &route_creation_mutex, t1, t2, routeToChunk, sweepDeadline);

    for (int iter = 0; useTimeBudget || iter < max_iterations; ++iter) {
        double elapsed_ms = 0.0;
        if (useTimeBudget) {
            elapsed_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - stageStart).count();
            if (elapsed_ms >= g_stage3_time_budget_ms) break;
            temperature = T0 * std::pow(Tf / T0, elapsed_ms / g_stage3_time_budget_ms);
            // Force greedy descent in the last ~5% of the budget, same intent as the
            // legacy path's "last 50 iterations" cutoff.
            if (elapsed_ms >= 0.95 * g_stage3_time_budget_ms) temperature = 0.0;
        } else if (iter >= max_iterations - 50) {
            temperature = 0.0; // Force greedy descent at the end
        }
        auto t_start = std::chrono::high_resolution_clock::now();
        arena.doCount = 0;
        arena.undoCount = 0;
        arena.pendingDelta = 0;
        // No prevNumRoutes snapshot here (unlike stage2_ils/stage5_serial_polish): this pass
        // shares one globalSolution across multiple concurrently-running chunk-pair threads
        // (disjoint routes, same object), so a snapshot-then-unconditional-restore of a
        // process-wide numRoutes counter races with any other thread's concurrent route
        // creation -- one thread's rejection could silently roll numRoutes back below a
        // route another thread just legitimately created, making its customers vanish from
        // every later scan/output (see docs/reports/005_cost_optimization.md Phase 1). On
        // reject, apply_undo_list below already fully unwinds this iteration's own
        // pred/succ/routeOf/routeLoad/routeHead/routeTail changes -- a route it created ends
        // up merely empty (routeHead[r]==0), which is the same harmless dead-slot state
        // recreate()'s empty-route reuse scan and Stage 4's cleanup already handle elsewhere
        // in this codebase, so simply not touching numRoutes here is both race-free and safe.

        std::uniform_int_distribution<int> dist_cust(0, boundaryList.size() - 1);
        NodeId seed_cust = boundaryList[dist_cust(rng)];
        int virtual_chunk_size = boundaryList.size(); // for log-scaled ruin walk
        ruin(globalSolution, seed_cust, arena, cache, rng, virtual_chunk_size, local_granular_lists, inst, &route_creation_mutex, t1, t2, routeToChunk);
        
        auto t_ruin = std::chrono::high_resolution_clock::now();
        
        recreate(globalSolution, arena, cache, inst, local_granular_lists, &route_creation_mutex, t1, t2, routeToChunk);

        // Only rescan routes ruin/recreate actually touched (docs/reports/005_cost_optimization.md
        // Phase 5) -- these are already guaranteed to be within t1/t2 since ruin/recreate
        // refuse to touch any route outside that filter, so no separate routeToChunk check
        // is needed here (unlike the old unconditional full-numRoutes loop this replaces).
        rescan_touched_routes(globalSolution, arena, inst);

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
            // Accumulate locally rather than writing globalSolution.totalCost directly:
            // multiple healing threads run this pass concurrently (on disjoint chunk-pairs
            // per the graph-coloring schedule), so a shared scalar read-modify-write here
            // would race even though the routes each thread touches are disjoint.
            acceptedDelta += delta;
        } else {
            // apply_undo_list already rescans every route it touched (Phase 1.2/5) -- no
            // separate rescan needed here anymore.
            apply_undo_list(globalSolution, arena, inst, &route_creation_mutex);
        }

        if (!useTimeBudget) {
            temperature *= cooling_rate;
        }
    }
    return acceptedDelta;
}

void stage5_serial_polish(Solution& globalSolution, ThreadArena& arena, const Instance& inst, const NeighborLists& neighborLists, double avgArcCostEstimate) {
    if (globalSolution.numRoutes == 0) return;
    
    SVCCache cache;
    cache.init(inst.n);
    cache.clear();

    // Refresh routePosition/cumLoad for every route before the full sweep below.
    // stage4_route_cleanup (main.cpp, runs immediately before this) relocates customers but
    // never calls update_route_info, so routes it touched can still be stale here; the old
    // code never noticed because Stage 5's own ILS loop always refreshed everything itself
    // after its first recreate() call. full_sweep_local_search now runs before that and
    // depends on both being current (e.g. eval_2opt_star's capacity check reads cumLoad) --
    // confirmed as a real cause of capacity violations during Tier-1 stress testing.
    for (int r = 0; r < globalSolution.numRoutes; ++r) {
        update_route_info(globalSolution, r, inst);
    }

    extern int g_seed; // overridable via --seed; offset chosen so default (1337) reproduces the
                        // prior hardcoded 424242 exactly (424242 - 1337 = 422905)
    std::mt19937 rng(g_seed + 422905);

    // See the identical comment in stage2_ils above (docs/reports/005_cost_optimization.md
    // Phase 1.3) -- same instance-scaling fix, same rationale.
    double avg_arc_cost_estimate = avgArcCostEstimate;
    double T0 = 0.1 * avg_arc_cost_estimate;
    if (T0 < 1e-6) T0 = 1.0;
    double Tf = 0.01 * T0;
    extern int g_stage5_time_budget_ms; // overridable via --stage5-ms; >0 switches to time-budget mode
    bool useTimeBudget = g_stage5_time_budget_ms > 0;
    int max_iterations = 500;
    int stagnation_limit = 150;
    int stagnation = 0;
    double cooling_rate = useTimeBudget ? 1.0 : std::pow(Tf / T0, 1.0 / max_iterations);

    // stageStart captured here, before the full sweep, so the sweep's deadline (Phase 1.1)
    // and the main loop's own elapsed-time check share one clock and one budget -- time the
    // sweep spends is time the main loop below has less of, not extra on top of --stage5-ms.
    auto sweepStart = std::chrono::steady_clock::now();
    auto sweepDeadline = useTimeBudget
        ? sweepStart + std::chrono::milliseconds(g_stage5_time_budget_ms)
        : std::chrono::steady_clock::time_point::max();

    // A real full-graph local-search descent before the ILS loop starts (see
    // full_sweep_local_search's comment) -- the previous "insert every customer" loop here
    // silently only queued the last 50 ids (SVCCache's ring-buffer capacity), so Stage 5 was
    // never actually polishing anything but a handful of highest-numbered nodes before this.
    {
        std::vector<int> all_customers(inst.n);
        for (int i = 1; i <= inst.n; ++i) all_customers[i - 1] = i;
        // Single-threaded here, so folding the delta straight into totalCost is safe (no
        // shared-scalar race to worry about, unlike the Stage 3 call site above).
        globalSolution.totalCost += full_sweep_local_search(globalSolution, arena, cache, inst, neighborLists, inst.n, all_customers, nullptr, -1, -1, nullptr, sweepDeadline);
    }
    double temperature = T0;

    Solution bestSol = globalSolution;

    // stagnation_limit (150) is disabled in time-budget mode: empirically, at N=2,000 it
    // triggers within ~150ms regardless of a much larger requested budget (150 consecutive
    // non-improving iterations happens almost immediately at this scale), silently
    // defeating the point of asking for more Stage 5 time. With a real time budget already
    // bounding worst-case runtime, continued ruin/recreate exploration after a stagnant
    // streak is low-risk (bounded time cost) and can still occasionally escape a local
    // optimum, so time-budget mode relies on the time cutoff alone. Legacy mode (no time
    // budget) keeps stagnation_limit as its only safety valve, unchanged.
    auto stageStart = std::chrono::steady_clock::now();
    for (int iter = 0; useTimeBudget || iter < max_iterations; ++iter) {
        if (useTimeBudget) {
            double elapsed_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - stageStart).count();
            if (elapsed_ms >= g_stage5_time_budget_ms) break;
            temperature = T0 * std::pow(Tf / T0, elapsed_ms / g_stage5_time_budget_ms);
        }
        arena.doCount = 0;
        arena.undoCount = 0;
        arena.pendingDelta = 0;
        
        int prevNumRoutes = globalSolution.numRoutes;
        
        std::uniform_int_distribution<int> dist_cust(1, inst.n);
        NodeId seed_cust = dist_cust(rng);
        
        ruin(globalSolution, seed_cust, arena, cache, rng, inst.n, neighborLists, inst, nullptr);

        recreate(globalSolution, arena, cache, inst, neighborLists, nullptr);

        // recreate() only calls update_route_info() for its new-empty-route fallback path;
        // a normal insertion into an existing route leaves that route's cumLoad/routePosition
        // stale. eval_2opt_star's capacity check reads cumLoad, so without this rescan it can
        // pass a move against stale (too-low) load data and produce an over-capacity route.
        // Only rescans routes ruin/recreate actually touched (docs/reports/005_cost_optimization.md
        // Phase 5) instead of the previous unconditional full O(N) sweep -- at Lazio's
        // ~1,000,000-node scale, a per-iteration full-graph rescan to service a ruin that
        // touches ~14 nodes dominated actual search time.
        rescan_touched_routes(globalSolution, arena, inst);

        bool local_search_improved = true;
        while(local_search_improved) {
            local_search_improved = local_search(globalSolution, arena, cache, inst, neighborLists, inst.n, nullptr);
        }

        Cost delta = arena.pendingDelta;
        if (accept_delta(delta, temperature, rng)) {
            globalSolution.totalCost += delta;
            if (globalSolution.totalCost < bestSol.totalCost - 1e-6) {
                snapshot_essential(bestSol, globalSolution);
                stagnation = 0;
            } else {
                stagnation++;
            }
        } else {
            // apply_undo_list already rescans every route it touched (Phase 1.2/5) -- no
            // separate full-route rescan needed here anymore.
            apply_undo_list(globalSolution, arena, inst, nullptr);
            globalSolution.numRoutes = prevNumRoutes;
            stagnation++;
        }
        
        if (!useTimeBudget && stagnation >= stagnation_limit) break;

        if (!useTimeBudget) {
            temperature *= cooling_rate;
        }
    }
    finalize_solution_derived_fields(bestSol, inst);
    globalSolution = bestSol;
}
