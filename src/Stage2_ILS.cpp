#include "Stage2_ILS.hpp"
#include <algorithm>
#include <cmath>
#include <mutex>
#include <chrono>
#include <iostream>
#include <atomic>

thread_local const char* current_op = "unknown";
thread_local char debug_info[256] = {0};

namespace {
    static std::mutex route_creation_mutex;
}

void update_route_info(Solution& sol, int route, const Instance& inst) {
        if (route == -1 || route >= sol.numRoutes) return;

        Cost load = 0;
        NodeId curr = sol.routeHead[route];
        int pos = 1;
        int count = 0;
        int max_nodes = inst.n + 2;
        while (curr != 0) {
            count++;
            if (count > max_nodes) {
                printf("CYCLE DETECTED IN update_route_info for route %d!\n", route); fflush(stdout);
                break;
            }
            load += inst.demand[curr];
            sol.cumLoad[curr] = load;
            sol.routePosition[curr] = pos++;
            curr = sol.succ[curr];
        }
        if (sol.routeLoad[route] != load) {
            printf("[FATAL] routeLoad desync for route %d: tracked=%lld, true=%lld\n", route, (long long)sol.routeLoad[route], (long long)load);
            fflush(stdout);
            exit(1);
        }
        sol.routeLoad[route] = load;
        if (load > inst.Q) {
            printf("[FATAL] update_route_info load %lld > Q for route %d (OP: %s)\n", (long long)load, route, ::current_op);
            fflush(stdout);
            exit(1);
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
namespace {
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
        for (int r = 0; r < sol.numRoutes; ++r) {
            if (sol.routeLoad[r] == 0) continue;
            Cost load = 0;
            NodeId curr = sol.routeHead[r];
            int pos = 1;
            int max_nodes = inst.n + 2;
            int count = 0;
            while (curr != 0) {
                count++;
                if (count > max_nodes) {
                    printf("CYCLE DETECTED IN ROUTE %d!\n", r); fflush(stdout);
                    break;
                }
                load += inst.demand[curr];
                sol.cumLoad[curr] = load;
                sol.routePosition[curr] = pos++;
                // T1: costToPred is dropped by snapshot_essential just like cumLoad/
                // routePosition (same rationale -- it's a pure function of pred/succ), so
                // regenerate it here too.
                sol.costToPred[curr] = dist(inst, sol.pred[curr], curr);
                curr = sol.succ[curr];
            }
            sol.routeLoad[r] = load;
        }
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
        
        // T1 (docs/reports/009_plan_beating_filo2.md): dist(p,c) and dist(c,s) are both
        // *existing* edges, already sitting in costToPred[c]/[s] from whatever previously
        // made c's neighbors what they are -- no need to recompute them. Only dist(p,s), the
        // edge this removal newly creates, is genuinely new. costToPred[0] (the depot slot)
        // is never maintained (see Solution.hpp), so s==0 (c was the route tail) still needs
        // a real dist() call for costCS.
        Cost costPC = sol.costToPred[c];
        Cost costCS = (s != 0) ? sol.costToPred[s] : dist(inst, c, 0);
        Cost costPS = dist(inst, p, s);
        Cost delta = costPS - costPC - costCS;
        undo_entry.costDelta = -delta;
        // Stash the pre-removal costToPred[c]/[s] so undo can restore them with no extra
        // dist() calls -- see the DoUndoEntry comment in ThreadArena.hpp.
        undo_entry.undoCostC = costPC;
        undo_entry.undoCostS = costCS;
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
        if (s != 0) sol.costToPred[s] = costPS;
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
        
        // T1: dist(p,s) is the *existing* edge being split by this insertion -- already
        // cached in costToPred[s] (invariant: s's pred is p right up until the mutation
        // below). dist(p,c) and dist(c,s) are new edges the insertion creates.
        Cost costPS = (s != 0) ? sol.costToPred[s] : dist(inst, p, 0);
        Cost costPC = dist(inst, p, c);
        Cost costCS = dist(inst, c, s);
        Cost delta = costPC + costCS - costPS;
        undo_entry.costDelta = -delta;
        // Restore target for undo (a later remove of c must put costToPred[s] back to
        // costPS) -- see the DoUndoEntry comment in ThreadArena.hpp. undoCostC is unused for
        // REMOVE-type entries (c leaves the route on undo, so its costToPred is moot).
        undo_entry.undoCostS = costPS;
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
        sol.costToPred[c] = costPC;
        if (s != 0) sol.costToPred[s] = costCS;
        sol.routeOf[c] = route;
        sol.routeLoad[route] += inst.demand[c];
        if (sol.routeLoad[route] > inst.Q) {
            printf("[FATAL] insert_customer load %lld > Q for route %d, inserting node %d (OP: %s)\n%s\n", (long long)sol.routeLoad[route], route, c, ::current_op, ::debug_info);
            printf("DUMP OF ROUTE %d:\n", route);
            NodeId curr = sol.routeHead[route];
            int count = 0;
            Cost true_load = 0;
            while (curr != 0) {
                printf("  Node %d (demand %lld, routeOf %d)\n", curr, (long long)inst.demand[curr], sol.routeOf[curr]);
                true_load += inst.demand[curr];
                curr = sol.succ[curr];
                count++;
                if (count > inst.n + 2) { printf("  [CYCLE DETECTED]\n"); break; }
            }
            printf("True load by traversal: %lld\n", (long long)true_load);
            fflush(stdout);
            exit(1);
        }
        
        if (p == 0) sol.routeHead[route] = c;
        if (s == 0) sol.routeTail[route] = c;
    }

    void apply_undo_list(Solution& sol, ThreadArena& arena, const Instance& inst, std::mutex* mtx = nullptr) {
        current_op = "apply_undo_list";
        if (mtx) mtx->lock();

        std::vector<int> touched_routes;
        for (int i = arena.undoCount - 1; i >= 0; --i) {
            const auto& entry = arena.undoList[i];
            if (entry.type == DoUndoEntry::INSERT) {
                NodeId c = entry.customer; NodeId p = entry.newPred; NodeId s = entry.newSucc; int route = entry.newRoute;
                sol.succ[p] = c; sol.pred[c] = p; sol.succ[c] = s; sol.pred[s] = c;
                // T1: restore the costToPred values this customer/successor had right before
                // the removal being undone -- stashed by remove_customer at the time, so no
                // dist() call needed here (see the DoUndoEntry comment in ThreadArena.hpp).
                sol.costToPred[c] = entry.undoCostC;
                if (s != 0) sol.costToPred[s] = entry.undoCostS;
                sol.routeOf[c] = route; sol.routeLoad[route] += inst.demand[c];

                if (p == 0) sol.routeHead[route] = c;
                if (s == 0) sol.routeTail[route] = c;
                touched_routes.push_back(route);
            } else {
                // Undoing an INSERT means we must REMOVE it
                NodeId c = entry.customer; int route = entry.prevRoute;
                NodeId p = sol.pred[c]; NodeId s = sol.succ[c];

                if (p != 0) sol.succ[p] = s;
                if (s != 0) { sol.pred[s] = p; sol.costToPred[s] = entry.undoCostS; }
                sol.pred[c] = 0; sol.succ[c] = 0;
                
                if (route != -1) {
                    sol.routeOf[c] = -1;
                    sol.routeLoad[route] -= inst.demand[c];
                    if (p == 0) sol.routeHead[route] = s;
                    if (s == 0) sol.routeTail[route] = p;
                    touched_routes.push_back(route);
                }
            }
        }

        // doList is still intact here (cleared below) and identifies exactly the same
        // touched-route set undoList does -- see rescan_touched_routes's comment.
        rescan_touched_routes(sol, arena, inst);
        
        for (int r : touched_routes) {
            if (sol.routeHead[r] != 0) {
                update_route_info(sol, r, inst);
            } else {
                sol.routeLoad[r] = 0;
            }
        }

        arena.doCount = 0; arena.undoCount = 0; arena.pendingDelta = 0;
        if (mtx) mtx->unlock();
    }

    // T1: dist(X, Y) where Y is currently in a route and X == sol.pred[Y] (i.e. an edge that
    // exists in the solution *right now*) is exactly costToPred[Y] -- no dist() call needed.
    // costToPred[0] is never maintained (0 is the depot sentinel, not a tracked customer), so
    // callers must fall back to a real dist() call whenever Y could be 0 (a route tail's
    // successor). Centralizing the Y==0 check here instead of guarding every call site by
    // hand removes the main way this optimization could silently read a stale slot.
    inline Cost curEdgeCost(const Solution& sol, const Instance& inst, NodeId X, NodeId Y) {
        return (Y != 0) ? sol.costToPred[Y] : dist(inst, X, 0);
    }

    // T2-lite pair cache invalidation (see PairCacheEntry, ThreadArena.hpp): for each touched
    // vertex v, invalidate v's own row (v's context changed, so every cached (v,*) entry is
    // stale) and every (u, j_idx) entry in reverseIdx[v] (v is one of u's candidates, so u's
    // cached delta for that specific pair may be stale too -- other entries in u's row are
    // untouched and stay valid). No-op when pairCache is null (every caller except
    // stage2_ils's main SA loop passes null, leaving them at their exact pre-T2 behavior).
    void invalidate_pair_cache_one(PairCacheEntry* pairCache, int k_max,
                                    const NeighborLists* reverseIdx_lists, NodeId v) {
        if (!pairCache || v == 0) return;
        for (int j_idx = 0; j_idx < k_max; ++j_idx) {
            pairCache[(size_t)v * k_max + j_idx].gen = -1;
        }
        if (!reverseIdx_lists) return;
        for (const auto& pr : reverseIdx_lists->reverseIdx[v]) {
            NodeId u = pr.first;
            int j_idx = pr.second;
            pairCache[(size_t)u * k_max + j_idx].gen = -1;
        }
    }

    void invalidate_svc(SVCCache& cache, NodeId i, NodeId j, NodeId p_i, NodeId s_i, NodeId p_j, NodeId s_j,
                         PairCacheEntry* pairCache = nullptr, int k_max = 0, const NeighborLists* reverseIdx_lists = nullptr) {
        if (i != 0) cache.insert(i);
        if (j != 0) cache.insert(j);
        if (p_i != 0) cache.insert(p_i);
        if (s_i != 0) cache.insert(s_i);
        if (p_j != 0) cache.insert(p_j);
        if (s_j != 0) cache.insert(s_j);

        invalidate_pair_cache_one(pairCache, k_max, reverseIdx_lists, i);
        invalidate_pair_cache_one(pairCache, k_max, reverseIdx_lists, j);
        invalidate_pair_cache_one(pairCache, k_max, reverseIdx_lists, p_i);
        invalidate_pair_cache_one(pairCache, k_max, reverseIdx_lists, s_i);
        invalidate_pair_cache_one(pairCache, k_max, reverseIdx_lists, p_j);
        invalidate_pair_cache_one(pairCache, k_max, reverseIdx_lists, s_j);
    }

    // omega: optional per-vertex adaptive ruin-walk length (T4.2, mirrors FILO2's omega
    // array in main.cpp) -- indexed by seed (a global NodeId). When null, falls back to the
    // previous fixed ceil(log(chunkSize)) walk length for every seed.
    void ruin(Solution& sol, NodeId seed, ThreadArena& arena, SVCCache& cache, std::mt19937& rng, int chunkSize, const NeighborLists& granular_lists, const Instance& inst, std::mutex* mtx = nullptr, int t1 = -1, int t2 = -1, std::vector<int>* routeToChunk = nullptr, bool append = false, const std::vector<int>* omega = nullptr) {
        if (mtx) mtx->lock();
        if (!append) arena.removed_count = 0;
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
        
        int walk_length = omega ? (*omega)[seed] : (int)std::ceil(std::log(chunkSize));
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

    void recreate(Solution& sol, ThreadArena& arena, SVCCache& cache, const Instance& inst, const NeighborLists& granular_lists, std::mt19937& rng, std::mutex* mtx = nullptr, int t1 = -1, int t2 = -1, std::vector<int>* routeToChunk = nullptr) {
        if (mtx) mtx->lock();
        // Vary reinsertion order across FILO2's 4 rules instead of always descending-demand,
        // so the recreate phase doesn't always resolve the same removal-order ties the same way.
        auto begin = arena.removed_customers.begin();
        auto end = arena.removed_customers.begin() + arena.removed_count;
        switch (std::uniform_int_distribution<int>(0, 3)(rng)) {
            case 0:
                std::shuffle(begin, end, rng);
                break;
            case 1:
                std::sort(begin, end, [&inst](NodeId a, NodeId b) { return inst.demand[a] > inst.demand[b]; });
                break;
            case 2:
                std::sort(begin, end, [&inst](NodeId a, NodeId b) { return dist(inst, a, 0) > dist(inst, b, 0); });
                break;
            case 3:
                std::sort(begin, end, [&inst](NodeId a, NodeId b) { return dist(inst, a, 0) < dist(inst, b, 0); });
                break;
        }

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
                    // No inner mtx lock here: recreate() already holds mtx for its entire
                    // call (locked at function entry, unlocked at function exit -- one
                    // critical section), and std::mutex is non-recursive, so a second
                    // lock() by the same thread deadlocks forever instead of blocking on
                    // another thread. This was the Stage 3 hang (reproducible even at
                    // N=2000, P=4: any time recreate() needed to create a genuinely new
                    // route, the thread locked route_creation_mutex against itself and
                    // never returned) -- confirmed by Stage 2's calls (mtx=nullptr) never
                    // hitting it, only Stage 3's (mtx=&route_creation_mutex) did.
                    r = sol.numRoutes++;
                    if (r >= (int)sol.routeHead.size()) {
                        sol.routeHead.resize(r + 100, 0);
                        sol.routeTail.resize(r + 100, 0);
                        sol.routeLoad.resize(r + 100, 0);
                    }
                    if (routeToChunk && t1 != -1) {
                        if (r >= (int)routeToChunk->size()) {
                            routeToChunk->resize(r + 100, -1);
                        }
                        (*routeToChunk)[r] = t1;
                    }
                }
                sol.routeLoad[r] = 0;
                insert_customer(sol, c, 0, 0, r, arena, inst);
                update_route_info(sol, r, inst);
                cache.insert(c);
            }
        }
        if (mtx) mtx->unlock();
    }

    Cost eval_relocate2(const Solution& sol, const Instance& inst, NodeId i, NodeId j) {
        if (i == 0 || j == 0) return 0;
        int r_i = sol.routeOf[i], r_j = sol.routeOf[j];
        if (r_i == -1 || r_j == -1) return 0;
        
        NodeId s_i = sol.succ[i];
        if (s_i == 0) return 0; // Cannot relocate depot
        NodeId s_s_i = sol.succ[s_i];
        
        if (j == i || j == sol.pred[i] || j == s_i || j == s_s_i) return 0;
        
        if (r_i != r_j && sol.routeLoad[r_j] + inst.demand[i] + inst.demand[s_i] > inst.Q) return 0;
        
        NodeId p_i = sol.pred[i], s_j = sol.succ[j];

        // T1: dist(p_i,i), dist(s_i,s_s_i), dist(j,s_j) are all *current* edges (unmutated at
        // this read-only eval point) -- already sitting in costToPred. dist(p_i,s_s_i) is the
        // edge the removal would newly create, not cached.
        Cost rem = dist(inst, p_i, s_s_i) - sol.costToPred[i] - curEdgeCost(sol, inst, s_i, s_s_i);
        Cost ins = dist(inst, j, i) + dist(inst, s_i, s_j) - curEdgeCost(sol, inst, j, s_j);

        return rem + ins;
    }

    Cost eval_relocate3(const Solution& sol, const Instance& inst, NodeId i, NodeId j) {
        if (i == 0 || j == 0) return 0;
        int r_i = sol.routeOf[i], r_j = sol.routeOf[j];
        if (r_i == -1 || r_j == -1) return 0;
        
        NodeId s_i = sol.succ[i];
        if (s_i == 0) return 0;
        NodeId s_s_i = sol.succ[s_i];
        if (s_s_i == 0) return 0;
        NodeId s_s_s_i = sol.succ[s_s_i];
        
        if (j == i || j == sol.pred[i] || j == s_i || j == s_s_i || j == s_s_s_i) return 0;
        
        if (r_i != r_j && sol.routeLoad[r_j] + inst.demand[i] + inst.demand[s_i] + inst.demand[s_s_i] > inst.Q) return 0;
        
        NodeId p_i = sol.pred[i], s_j = sol.succ[j];

        Cost rem = dist(inst, p_i, s_s_s_i) - sol.costToPred[i] - curEdgeCost(sol, inst, s_s_i, s_s_s_i);
        Cost ins = dist(inst, j, i) + dist(inst, s_s_i, s_j) - curEdgeCost(sol, inst, j, s_j);

        return rem + ins;
    }

    // Reversed-insertion variants of relocate2/relocate3: same segment, same removal cost
    // (removal doesn't depend on insertion orientation), but inserted at the destination in
    // the opposite order (j -> s_i -> i -> s_j instead of j -> i -> s_i -> s_j). The
    // segment's own internal edges (i-s_i, s_i-s_s_i) are unchanged by reversal -- distance is
    // symmetric -- so only the two boundary edges touching j/s_j differ, matching FILO2's
    // RevTwoZeroExchange/RevThreeZeroExchange (docs: report-006-era operator inventory).
    Cost eval_relocate2_rev(const Solution& sol, const Instance& inst, NodeId i, NodeId j) {
        if (i == 0 || j == 0) return 0;
        int r_i = sol.routeOf[i], r_j = sol.routeOf[j];
        if (r_i == -1 || r_j == -1) return 0;

        NodeId s_i = sol.succ[i];
        if (s_i == 0) return 0;
        NodeId s_s_i = sol.succ[s_i];

        if (j == i || j == sol.pred[i] || j == s_i || j == s_s_i) return 0;

        if (r_i != r_j && sol.routeLoad[r_j] + inst.demand[i] + inst.demand[s_i] > inst.Q) return 0;

        NodeId p_i = sol.pred[i], s_j = sol.succ[j];

        Cost rem = dist(inst, p_i, s_s_i) - sol.costToPred[i] - curEdgeCost(sol, inst, s_i, s_s_i);
        Cost ins = dist(inst, j, s_i) + dist(inst, i, s_j) - curEdgeCost(sol, inst, j, s_j);

        return rem + ins;
    }

    Cost eval_relocate3_rev(const Solution& sol, const Instance& inst, NodeId i, NodeId j) {
        if (i == 0 || j == 0) return 0;
        int r_i = sol.routeOf[i], r_j = sol.routeOf[j];
        if (r_i == -1 || r_j == -1) return 0;

        NodeId s_i = sol.succ[i];
        if (s_i == 0) return 0;
        NodeId s_s_i = sol.succ[s_i];
        if (s_s_i == 0) return 0;
        NodeId s_s_s_i = sol.succ[s_s_i];

        if (j == i || j == sol.pred[i] || j == s_i || j == s_s_i || j == s_s_s_i) return 0;

        if (r_i != r_j && sol.routeLoad[r_j] + inst.demand[i] + inst.demand[s_i] + inst.demand[s_s_i] > inst.Q) return 0;

        NodeId p_i = sol.pred[i], s_j = sol.succ[j];

        Cost rem = dist(inst, p_i, s_s_s_i) - sol.costToPred[i] - curEdgeCost(sol, inst, s_s_i, s_s_s_i);
        Cost ins = dist(inst, j, s_s_i) + dist(inst, i, s_j) - curEdgeCost(sol, inst, j, s_j);

        return rem + ins;
    }

    Cost eval_relocate(const Solution& sol, const Instance& inst, NodeId i, NodeId j) {
        if (i == 0 || j == 0) return 0; // Forbid depot as primary operand
        if (i == j || sol.pred[i] == j || sol.succ[i] == j) return 0; // Adjacency double-count protection
        
        int r_i = sol.routeOf[i], r_j = sol.routeOf[j];
        if (r_i == -1 || r_j == -1) return 0;
        // Capacity short-circuit BEFORE distance lookups
        if (r_i != r_j && sol.routeLoad[r_j] + inst.demand[i] > inst.Q) return 0;
        
        NodeId p_i = sol.pred[i], s_i = sol.succ[i], s_j = sol.succ[j];
        return -sol.costToPred[i] - curEdgeCost(sol, inst, i, s_i) + dist(inst, p_i, s_i)
               -curEdgeCost(sol, inst, j, s_j) + dist(inst, j, i) + dist(inst, i, s_j);
    }
    
    // T5.2 (docs/reports/009_plan_beating_filo2.md): E21 -- swaps the 2-customer segment
    // (p_i, i) with the 1-customer segment (p_j), preserving each segment's internal order.
    // Ported from FILO2's TwoOneExchange.hpp (localsearch/). Generalizes eval_swap (E11,
    // 1-for-1) to a 2-for-1 exchange. Scoped to cross-route only (r_i != r_j): FILO2 also
    // supports a same-route variant, but that needs careful adjacency exclusions (segments
    // overlapping/touching) that are a real source of bugs elsewhere in this file when gotten
    // wrong -- cross-route-only avoids that class entirely at the cost of some same-route
    // moves 2-opt/other operators may already cover partially.
    Cost eval_E21(const Solution& sol, const Instance& inst, NodeId i, NodeId j) {
        if (i == 0 || j == 0) return 0;
        int r_i = sol.routeOf[i], r_j = sol.routeOf[j];
        if (r_i == -1 || r_j == -1 || r_i == r_j) return 0;

        NodeId p_i = sol.pred[i];
        if (p_i == 0) return 0; // need a real 2-segment (p_i,i); p_i can't be the depot
        NodeId p_j = sol.pred[j];
        if (p_j == 0) return 0; // need a real 1-segment (p_j) to swap out

        if (sol.routeLoad[r_j] - inst.demand[p_j] + inst.demand[p_i] + inst.demand[i] > inst.Q) return 0;
        if (sol.routeLoad[r_i] + inst.demand[p_j] - inst.demand[p_i] - inst.demand[i] > inst.Q) return 0;

        NodeId s_i = sol.succ[i];
        NodeId pp_i = sol.pred[p_i];
        NodeId pp_j = sol.pred[p_j];

        Cost rem = -sol.costToPred[p_i] - curEdgeCost(sol, inst, i, s_i) - sol.costToPred[p_j] - sol.costToPred[j];
        Cost add = dist(inst, pp_j, p_i) + dist(inst, i, j) + dist(inst, pp_i, p_j) + dist(inst, p_j, s_i);
        return add + rem;
    }

    void apply_E21(Solution& sol, ThreadArena& arena, const Instance& inst, NodeId i, NodeId j, SVCCache& cache,
                   PairCacheEntry* pairCache = nullptr, int k_max = 0, const NeighborLists* reverseIdx_lists = nullptr) {
        current_op = "apply_E21";
        NodeId p_i = sol.pred[i], s_i = sol.succ[i], pp_i = sol.pred[p_i];
        NodeId p_j = sol.pred[j], pp_j = sol.pred[p_j];
        int r_i = sol.routeOf[i], r_j = sol.routeOf[j];

        remove_customer(sol, i, arena, inst);
        remove_customer(sol, p_i, arena, inst);
        remove_customer(sol, p_j, arena, inst);
        // iRoute now: pp_i -> s_i. jRoute now: pp_j -> j.

        insert_customer(sol, p_i, pp_j, j, r_j, arena, inst);
        insert_customer(sol, i, p_i, j, r_j, arena, inst);
        insert_customer(sol, p_j, pp_i, s_i, r_i, arena, inst);

        update_route_info(sol, r_i, inst);
        update_route_info(sol, r_j, inst);

        invalidate_svc(cache, i, j, pp_i, s_i, pp_j, p_j, pairCache, k_max, reverseIdx_lists);
        cache.insert(p_i);
        invalidate_pair_cache_one(pairCache, k_max, reverseIdx_lists, p_i);
    }

    // T5.2: E22 -- swaps the 2-customer segment (p_i, i) with the 2-customer segment
    // (pp_j, p_j), preserving each segment's internal order. Ported from FILO2's
    // TwoTwoExchange.hpp. Cross-route only, same rationale as eval_E21 above.
    Cost eval_E22(const Solution& sol, const Instance& inst, NodeId i, NodeId j) {
        if (i == 0 || j == 0) return 0;
        int r_i = sol.routeOf[i], r_j = sol.routeOf[j];
        if (r_i == -1 || r_j == -1 || r_i == r_j) return 0;

        NodeId p_i = sol.pred[i];
        if (p_i == 0) return 0;
        NodeId p_j = sol.pred[j];
        if (p_j == 0) return 0;
        NodeId pp_j = sol.pred[p_j];
        if (pp_j == 0) return 0; // need a real 2-segment (pp_j,p_j)

        if (sol.routeLoad[r_j] - inst.demand[p_j] - inst.demand[pp_j] + inst.demand[i] + inst.demand[p_i] > inst.Q) return 0;
        if (sol.routeLoad[r_i] + inst.demand[p_j] + inst.demand[pp_j] - inst.demand[i] - inst.demand[p_i] > inst.Q) return 0;

        NodeId s_i = sol.succ[i];
        NodeId pp_i = sol.pred[p_i];
        NodeId ppp_j = sol.pred[pp_j];

        Cost rem = -sol.costToPred[p_i] - curEdgeCost(sol, inst, i, s_i) - sol.costToPred[pp_j] - sol.costToPred[j];
        Cost add = dist(inst, ppp_j, p_i) + dist(inst, i, j) + dist(inst, pp_i, pp_j) + dist(inst, p_j, s_i);
        return add + rem;
    }

    void apply_E22(Solution& sol, ThreadArena& arena, const Instance& inst, NodeId i, NodeId j, SVCCache& cache,
                   PairCacheEntry* pairCache = nullptr, int k_max = 0, const NeighborLists* reverseIdx_lists = nullptr) {
        current_op = "apply_E22";
        NodeId p_i = sol.pred[i], s_i = sol.succ[i], pp_i = sol.pred[p_i];
        NodeId p_j = sol.pred[j], pp_j = sol.pred[p_j], ppp_j = sol.pred[pp_j];
        int r_i = sol.routeOf[i], r_j = sol.routeOf[j];

        remove_customer(sol, i, arena, inst);
        remove_customer(sol, p_i, arena, inst);
        remove_customer(sol, p_j, arena, inst);
        remove_customer(sol, pp_j, arena, inst);
        // iRoute now: pp_i -> s_i. jRoute now: ppp_j -> j.

        insert_customer(sol, p_i, ppp_j, j, r_j, arena, inst);
        insert_customer(sol, i, p_i, j, r_j, arena, inst);
        insert_customer(sol, pp_j, pp_i, s_i, r_i, arena, inst);
        insert_customer(sol, p_j, pp_j, s_i, r_i, arena, inst);

        update_route_info(sol, r_i, inst);
        update_route_info(sol, r_j, inst);

        invalidate_svc(cache, i, j, pp_i, s_i, ppp_j, p_j, pairCache, k_max, reverseIdx_lists);
        cache.insert(p_i);
        cache.insert(pp_j);
        invalidate_pair_cache_one(pairCache, k_max, reverseIdx_lists, p_i);
        invalidate_pair_cache_one(pairCache, k_max, reverseIdx_lists, pp_j);
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
            return -sol.costToPred[i] - curEdgeCost(sol, inst, j, s_j)
                   +dist(inst, p_i, j) + dist(inst, i, s_j); // distance i,j cancels out
        } else if (s_j == i) {
            return -sol.costToPred[j] - curEdgeCost(sol, inst, i, s_i)
                   +dist(inst, p_j, i) + dist(inst, j, s_i); // distance j,i cancels out
        } else {
            return -sol.costToPred[i] - curEdgeCost(sol, inst, i, s_i) - sol.costToPred[j] - curEdgeCost(sol, inst, j, s_j)
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
        return -curEdgeCost(sol, inst, i, s_i) - curEdgeCost(sol, inst, j, s_j) + dist(inst, i, j) + dist(inst, s_i, s_j);
    }

    Cost eval_2opt_star(const Solution& sol, const Instance& inst, NodeId i, NodeId j) {
        if (i == 0 || j == 0) return 0; // Explicitly forbid depot intersections
        int r_i = sol.routeOf[i], r_j = sol.routeOf[j];
        if (r_i == -1 || r_j == -1) return 0;
        if (r_i == r_j) return 0;

        // Deriving both "kept" and "tail" loads from routeLoad[r] - cumLoad[node] (rather
        // than an independent fresh walk for tail, as this used to do) is deliberate:
        // routeLoad[r] is kept exactly up to date in real time by every remove_customer/
        // insert_customer call, so kept+tail always sums to the true routeLoad[r] even if
        // cumLoad[node] itself is somewhat stale (only refreshed by update_route_info,
        // called once per touched route per SA iteration, not after every single move).
        // A previous rewrite computed load_tail_i/j via a fresh sol.succ walk while still
        // reading load_kept_i/j from cumLoad directly (plus an erroneous extra
        // + inst.demand[i]/[j] on top) -- that decoupling meant a stale cumLoad[j] could
        // under-report route j's "kept" load with nothing forcing the numbers back toward
        // the true routeLoad total, letting a genuinely-infeasible move pass this check and
        // then blow capacity for real in apply_2opt_star's insert_customer. Confirmed via a
        // reproducible VDA crash: route sitting at exactly Q, one more customer's demand
        // (2-3 units) let through by this check, then insert_customer's own (always-current)
        // capacity check correctly caught it and aborted -- i.e. this check was the one that
        // should have rejected the move earlier but didn't.
        Cost load_tail_i = sol.routeLoad[r_i] - sol.cumLoad[i];
        Cost load_tail_j = sol.routeLoad[r_j] - sol.cumLoad[j];

        // Capacity short-circuit BEFORE distance lookups
        if (sol.routeLoad[r_i] - load_tail_i + load_tail_j > inst.Q) return 0;
        if (sol.routeLoad[r_j] - load_tail_j + load_tail_i > inst.Q) return 0;

        NodeId s_i = sol.succ[i], s_j = sol.succ[j];
        Cost delta = -curEdgeCost(sol, inst, i, s_i) - curEdgeCost(sol, inst, j, s_j) + dist(inst, i, s_j) + dist(inst, j, s_i);

        return delta;
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

        int loops = 0;
        int max_nodes = inst.n + 2;
        while (true) {
            loops++;
            if (loops > max_nodes) {
                printf("[FATAL] HANG IN get_top3_insertions! route=%d\n", target_route); fflush(stdout);
                exit(1);
            }
            // T1: dist(p,s) is the current edge between this route's consecutive p/s pair
            // (unmutated during this scan) -- already cached, saves a dist() call on every
            // one of the O(routeLen) positions this loop visits.
            Cost delta = dist(inst, p, v) + dist(inst, v, s) - curEdgeCost(sol, inst, p, s);
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
        
        Cost rem_i = sol.costToPred[i] + curEdgeCost(sol, inst, i, sol.succ[i]) - dist(inst, sol.pred[i], sol.succ[i]);
        Cost rem_j = sol.costToPred[j] + curEdgeCost(sol, inst, j, sol.succ[j]) - dist(inst, sol.pred[j], sol.succ[j]);

        return ins_i + ins_j - rem_i - rem_j;
    }

    Cost eval_swap_star_fast(const Solution& sol, const Instance& inst, NodeId i, NodeId j, 
                             const Top3Insertions& top3_i, const Top3Insertions& top3_j,
                             NodeId& out_p_i, NodeId& out_s_i, NodeId& out_p_j, NodeId& out_s_j) {
        Cost ins_i = eval_swap_star_dir(sol, inst, i, j, top3_i, out_p_i, out_s_i);
        Cost ins_j = eval_swap_star_dir(sol, inst, j, i, top3_j, out_p_j, out_s_j);
        
        Cost rem_i = sol.costToPred[i] + curEdgeCost(sol, inst, i, sol.succ[i]) - dist(inst, sol.pred[i], sol.succ[i]);
        Cost rem_j = sol.costToPred[j] + curEdgeCost(sol, inst, j, sol.succ[j]) - dist(inst, sol.pred[j], sol.succ[j]);

        return ins_i + ins_j - rem_i - rem_j;
    }

    void apply_relocate2(Solution& sol, ThreadArena& arena, const Instance& inst, NodeId i, NodeId j, SVCCache& cache,
                         PairCacheEntry* pairCache = nullptr, int k_max = 0, const NeighborLists* reverseIdx_lists = nullptr) {
        current_op = "apply_relocate2";
        NodeId s_i = sol.succ[i];
        NodeId p_i = sol.pred[i], s_s_i = sol.succ[s_i], s_j = sol.succ[j];
        int r_i = sol.routeOf[i], r_j = sol.routeOf[j];

        remove_customer(sol, s_i, arena, inst);
        remove_customer(sol, i, arena, inst);

        insert_customer(sol, i, j, s_j, r_j, arena, inst);
        insert_customer(sol, s_i, i, s_j, r_j, arena, inst);

        update_route_info(sol, r_i, inst);
        update_route_info(sol, r_j, inst);

        invalidate_svc(cache, i, j, p_i, s_s_i, j, s_j, pairCache, k_max, reverseIdx_lists);
        cache.insert(s_i);
        invalidate_pair_cache_one(pairCache, k_max, reverseIdx_lists, s_i);
    }

    void apply_relocate3(Solution& sol, ThreadArena& arena, const Instance& inst, NodeId i, NodeId j, SVCCache& cache,
                         PairCacheEntry* pairCache = nullptr, int k_max = 0, const NeighborLists* reverseIdx_lists = nullptr) {
        current_op = "apply_relocate3";
        NodeId s_i = sol.succ[i], s_s_i = sol.succ[s_i];
        NodeId p_i = sol.pred[i], s_s_s_i = sol.succ[s_s_i], s_j = sol.succ[j];
        int r_i = sol.routeOf[i], r_j = sol.routeOf[j];

        remove_customer(sol, s_s_i, arena, inst);
        remove_customer(sol, s_i, arena, inst);
        remove_customer(sol, i, arena, inst);

        insert_customer(sol, i, j, s_j, r_j, arena, inst);
        insert_customer(sol, s_i, i, s_j, r_j, arena, inst);
        insert_customer(sol, s_s_i, s_i, s_j, r_j, arena, inst);

        update_route_info(sol, r_i, inst);
        update_route_info(sol, r_j, inst);

        invalidate_svc(cache, i, j, p_i, s_s_s_i, j, s_j, pairCache, k_max, reverseIdx_lists);
        cache.insert(s_i);
        cache.insert(s_s_i);
        invalidate_pair_cache_one(pairCache, k_max, reverseIdx_lists, s_i);
        invalidate_pair_cache_one(pairCache, k_max, reverseIdx_lists, s_s_i);
    }

    void apply_relocate2_rev(Solution& sol, ThreadArena& arena, const Instance& inst, NodeId i, NodeId j, SVCCache& cache,
                             PairCacheEntry* pairCache = nullptr, int k_max = 0, const NeighborLists* reverseIdx_lists = nullptr) {
        current_op = "apply_relocate2_rev";
        NodeId s_i = sol.succ[i];
        NodeId p_i = sol.pred[i], s_s_i = sol.succ[s_i], s_j = sol.succ[j];
        int r_i = sol.routeOf[i], r_j = sol.routeOf[j];

        remove_customer(sol, s_i, arena, inst);
        remove_customer(sol, i, arena, inst);

        insert_customer(sol, s_i, j, s_j, r_j, arena, inst);
        insert_customer(sol, i, s_i, s_j, r_j, arena, inst);

        update_route_info(sol, r_i, inst);
        update_route_info(sol, r_j, inst);

        invalidate_svc(cache, i, j, p_i, s_s_i, j, s_j, pairCache, k_max, reverseIdx_lists);
        cache.insert(s_i);
        invalidate_pair_cache_one(pairCache, k_max, reverseIdx_lists, s_i);
    }

    void apply_relocate3_rev(Solution& sol, ThreadArena& arena, const Instance& inst, NodeId i, NodeId j, SVCCache& cache,
                             PairCacheEntry* pairCache = nullptr, int k_max = 0, const NeighborLists* reverseIdx_lists = nullptr) {
        current_op = "apply_relocate3_rev";
        NodeId s_i = sol.succ[i], s_s_i = sol.succ[s_i];
        NodeId p_i = sol.pred[i], s_s_s_i = sol.succ[s_s_i], s_j = sol.succ[j];
        int r_i = sol.routeOf[i], r_j = sol.routeOf[j];

        remove_customer(sol, s_s_i, arena, inst);
        remove_customer(sol, s_i, arena, inst);
        remove_customer(sol, i, arena, inst);

        insert_customer(sol, s_s_i, j, s_j, r_j, arena, inst);
        insert_customer(sol, s_i, s_s_i, s_j, r_j, arena, inst);
        insert_customer(sol, i, s_i, s_j, r_j, arena, inst);

        update_route_info(sol, r_i, inst);
        update_route_info(sol, r_j, inst);

        invalidate_svc(cache, i, j, p_i, s_s_s_i, j, s_j, pairCache, k_max, reverseIdx_lists);
        cache.insert(s_i);
        cache.insert(s_s_i);
        invalidate_pair_cache_one(pairCache, k_max, reverseIdx_lists, s_i);
        invalidate_pair_cache_one(pairCache, k_max, reverseIdx_lists, s_s_i);
    }

    void apply_relocate(Solution& sol, ThreadArena& arena, const Instance& inst, NodeId i, NodeId j, SVCCache& cache,
                        PairCacheEntry* pairCache = nullptr, int k_max = 0, const NeighborLists* reverseIdx_lists = nullptr) {
        current_op = "apply_relocate";
        NodeId p_i = sol.pred[i], s_i = sol.succ[i], s_j = sol.succ[j];
        int r_i = sol.routeOf[i];
        int r_j = sol.routeOf[j];
        remove_customer(sol, i, arena, inst);
        insert_customer(sol, i, j, s_j, r_j, arena, inst);
        update_route_info(sol, r_i, inst);
        if (r_i != r_j) update_route_info(sol, r_j, inst);
        invalidate_svc(cache, i, j, p_i, s_i, j, s_j, pairCache, k_max, reverseIdx_lists);
    }

    void apply_swap(Solution& sol, ThreadArena& arena, const Instance& inst, NodeId i, NodeId j, SVCCache& cache,
                    PairCacheEntry* pairCache = nullptr, int k_max = 0, const NeighborLists* reverseIdx_lists = nullptr) {
        current_op = "apply_swap";
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
        invalidate_svc(cache, i, j, p_i, s_i, p_j, s_j, pairCache, k_max, reverseIdx_lists);
    }

    void apply_2opt(Solution& sol, ThreadArena& arena, const Instance& inst, NodeId i, NodeId j, SVCCache& cache,
                    PairCacheEntry* pairCache = nullptr, int k_max = 0, const NeighborLists* reverseIdx_lists = nullptr) {
        if (!is_before(sol, i, j)) std::swap(i, j);

        std::vector<NodeId> seg;
        NodeId curr = sol.succ[i];
        int loops = 0;
        while (curr != sol.succ[j]) { 
            loops++;
            if (loops > inst.n + 2) {
                printf("[FATAL] HANG IN apply_2opt! i=%d (routeOf=%d) j=%d (routeOf=%d)\n", i, sol.routeOf[i], j, sol.routeOf[j]);
                printf("DUMP OF ROUTE %d:\n", sol.routeOf[i]);
                NodeId dbg_curr = sol.routeHead[sol.routeOf[i]];
                int dbg_count = 0;
                while(dbg_curr != 0) {
                    printf("  Node %d (routeOf %d)\n", dbg_curr, sol.routeOf[dbg_curr]);
                    dbg_curr = sol.succ[dbg_curr];
                    if (++dbg_count > inst.n + 2) { printf("  [CYCLE DETECTED]\n"); break; }
                }
                fflush(stdout);
                exit(1);
            }
            seg.push_back(curr); 
            curr = sol.succ[curr]; 
        }
        
        int route = sol.routeOf[i];
        for (NodeId v : seg) remove_customer(sol, v, arena, inst);
        
        NodeId insert_after = i;
        for (auto it = seg.rbegin(); it != seg.rend(); ++it) {
            NodeId v = *it;
            NodeId s = sol.succ[insert_after];
            insert_customer(sol, v, insert_after, s, route, arena, inst);
            insert_after = v;
            cache.insert(v);
            invalidate_pair_cache_one(pairCache, k_max, reverseIdx_lists, v);
        }
        update_route_info(sol, route, inst);
        cache.insert(i); cache.insert(j);
        invalidate_pair_cache_one(pairCache, k_max, reverseIdx_lists, i);
        invalidate_pair_cache_one(pairCache, k_max, reverseIdx_lists, j);
    }

    void apply_2opt_star(Solution& sol, ThreadArena& arena, const Instance& inst, NodeId i, NodeId j, SVCCache& cache,
                         PairCacheEntry* pairCache = nullptr, int k_max = 0, const NeighborLists* reverseIdx_lists = nullptr) {
        current_op = "apply_2opt_star";
        int r_i = sol.routeOf[i], r_j = sol.routeOf[j];
        
        std::vector<NodeId> tail_i, tail_j;
        NodeId curr = sol.succ[i];
        int loops = 0;
        while (curr != 0) { 
            loops++;
            if (loops > inst.n + 2) {
                printf("[FATAL] HANG IN apply_2opt_star! i=%d\n", i); fflush(stdout);
                exit(1);
            }
            tail_i.push_back(curr); 
            curr = sol.succ[curr]; 
        }
        
        curr = sol.succ[j];
        loops = 0;
        while (curr != 0) { 
            loops++;
            if (loops > inst.n + 2) {
                printf("[FATAL] HANG IN apply_2opt_star! j=%d\n", j); fflush(stdout);
                exit(1);
            }
            tail_j.push_back(curr); 
            curr = sol.succ[curr]; 
        }
        
        for (NodeId v : tail_i) remove_customer(sol, v, arena, inst);
        for (NodeId v : tail_j) remove_customer(sol, v, arena, inst);
        
        NodeId insert_after = j;
        for (NodeId v : tail_i) {
            NodeId s = sol.succ[insert_after];
            insert_customer(sol, v, insert_after, s, r_j, arena, inst);
            insert_after = v; cache.insert(v);
            invalidate_pair_cache_one(pairCache, k_max, reverseIdx_lists, v);
        }

        insert_after = i;
        for (NodeId v : tail_j) {
            NodeId s = sol.succ[insert_after];
            insert_customer(sol, v, insert_after, s, r_i, arena, inst);
            insert_after = v; cache.insert(v);
            invalidate_pair_cache_one(pairCache, k_max, reverseIdx_lists, v);
        }
        update_route_info(sol, r_i, inst);
        update_route_info(sol, r_j, inst);
        cache.insert(i); cache.insert(j);
        invalidate_pair_cache_one(pairCache, k_max, reverseIdx_lists, i);
        invalidate_pair_cache_one(pairCache, k_max, reverseIdx_lists, j);
    }

    void apply_swap_star(Solution& sol, ThreadArena& arena, const Instance& inst, NodeId i, NodeId j,
                         NodeId p_i, NodeId s_i, NodeId p_j, NodeId s_j, SVCCache& cache,
                         PairCacheEntry* pairCache = nullptr, int k_max = 0, const NeighborLists* reverseIdx_lists = nullptr) {
        current_op = "apply_swap_star";
        NodeId orig_p_i = sol.pred[i], orig_s_i = sol.succ[i];
        NodeId orig_p_j = sol.pred[j], orig_s_j = sol.succ[j];
        int r_i = sol.routeOf[i], r_j = sol.routeOf[j];

        remove_customer(sol, i, arena, inst);
        remove_customer(sol, j, arena, inst);

        insert_customer(sol, i, p_i, s_i, r_j, arena, inst);
        insert_customer(sol, j, p_j, s_j, r_i, arena, inst);

        update_route_info(sol, r_i, inst);
        update_route_info(sol, r_j, inst);

        invalidate_svc(cache, i, j, orig_p_i, orig_s_i, orig_p_j, orig_s_j, pairCache, k_max, reverseIdx_lists);
        cache.insert(p_i); cache.insert(s_i);
        cache.insert(p_j); cache.insert(s_j);
        invalidate_pair_cache_one(pairCache, k_max, reverseIdx_lists, p_i);
        invalidate_pair_cache_one(pairCache, k_max, reverseIdx_lists, s_i);
        invalidate_pair_cache_one(pairCache, k_max, reverseIdx_lists, p_j);
        invalidate_pair_cache_one(pairCache, k_max, reverseIdx_lists, s_j);
    }

    bool local_search(Solution& sol, ThreadArena& arena, SVCCache& cache, const Instance& inst, const NeighborLists& granular_lists, int chunkSize, std::mutex* mtx = nullptr, int t1 = -1, int t2 = -1, std::vector<int>* routeToChunk = nullptr,
                      PairCacheEntry* pairCache = nullptr, int k_max = 0, const NeighborLists* reverseIdx_lists = nullptr) {
        bool improved = false;
        int ls_iter = 0;
        
        while (cache.count > 0) {
            ls_iter++;
            if (ls_iter > 50000000) break; // Safety net
            NodeId i = cache.pop();
            
            if (sol.routeOf[i] == -1) continue;
            
            Cost bestDelta = 0;
            int bestOp = -1; // 0=Relocate, 1=Swap, 2=2-Opt, 3=2-Opt*, 4=Swap*, 5=Relocate2, 6=Relocate3, 7=Relocate2Rev, 8=Relocate3Rev, 9=E21, 10=E22
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

                Cost pairDelta;
                int pairOp;

                // T2-lite (docs/reports/009_plan_beating_filo2.md, ThreadArena.hpp
                // PairCacheEntry): reuse the cached best-of-9-operators result for this exact
                // (i,j) pair if it's still valid, instead of redoing all 9 eval_* calls. Only
                // stage2_ils's main SA loop passes a non-null pairCache; every other caller
                // (stage3/5/routemin) takes the pairCache==nullptr branch below unconditionally,
                // i.e. their behavior is byte-for-byte what it was before T2-lite.
                PairCacheEntry* slot = pairCache ? &pairCache[(size_t)i * k_max + j_idx] : nullptr;
                if (slot && slot->gen == arena.pairCacheGen &&
                    slot->loadRi == sol.routeLoad[r_i] && slot->loadRj == sol.routeLoad[r_j]) {
                    pairDelta = slot->delta;
                    pairOp = slot->op;
                } else {
                    pairDelta = 0;
                    pairOp = -1;

                    Cost delta_relocate = eval_relocate(sol, inst, i, j);
                    if (delta_relocate < pairDelta) { pairDelta = delta_relocate; pairOp = 0; }

                    Cost delta_relocate2 = eval_relocate2(sol, inst, i, j);
                    if (delta_relocate2 < pairDelta) { pairDelta = delta_relocate2; pairOp = 5; }

                    Cost delta_relocate3 = eval_relocate3(sol, inst, i, j);
                    if (delta_relocate3 < pairDelta) { pairDelta = delta_relocate3; pairOp = 6; }

                    Cost delta_relocate2_rev = eval_relocate2_rev(sol, inst, i, j);
                    if (delta_relocate2_rev < pairDelta) { pairDelta = delta_relocate2_rev; pairOp = 7; }

                    Cost delta_relocate3_rev = eval_relocate3_rev(sol, inst, i, j);
                    if (delta_relocate3_rev < pairDelta) { pairDelta = delta_relocate3_rev; pairOp = 8; }

                    Cost delta_swap = eval_swap(sol, inst, i, j);
                    if (delta_swap < pairDelta) { pairDelta = delta_swap; pairOp = 1; }

                    Cost delta_E21 = eval_E21(sol, inst, i, j);
                    if (delta_E21 < pairDelta) { pairDelta = delta_E21; pairOp = 9; }

                    Cost delta_E22 = eval_E22(sol, inst, i, j);
                    if (delta_E22 < pairDelta) { pairDelta = delta_E22; pairOp = 10; }

                    Cost delta_2opt = eval_2opt(sol, inst, i, j);
                    if (delta_2opt < pairDelta) { pairDelta = delta_2opt; pairOp = 2; }

                    Cost delta_2opt_star = eval_2opt_star(sol, inst, i, j);
                    if (delta_2opt_star < pairDelta) { pairDelta = delta_2opt_star; pairOp = 3; }

                    // top3_i_to_V is route-indexed (ThreadArena.hpp) -- see the identical guard
                    // and comment on the precompute loop above (Phase 4.1).
                    if (r_i != r_j && r_j < (int)arena.top3_i_to_V.size()) {
                        // Capacity short-circuit BEFORE distance lookups
                        if (sol.routeLoad[r_i] - inst.demand[i] + inst.demand[j] <= inst.Q &&
                            sol.routeLoad[r_j] - inst.demand[j] + inst.demand[i] <= inst.Q) {
                            NodeId p_i, s_i, p_j, s_j;
                            Cost delta_swap_star = eval_swap_star_fast(sol, inst, i, j, arena.top3_i_to_V[r_j], arena.top3_j_to_U[j], p_i, s_i, p_j, s_j);
                            if (delta_swap_star < pairDelta) { pairDelta = delta_swap_star; pairOp = 4; }
                        }
                    }

                    if (slot) {
                        slot->delta = pairDelta;
                        slot->op = (int8_t)pairOp;
                        slot->gen = arena.pairCacheGen;
                        slot->loadRi = sol.routeLoad[r_i];
                        slot->loadRj = sol.routeLoad[r_j];
                    }
                }

                // swap_star's p_i/s_i/p_j/s_j are never cached (see PairCacheEntry comment) --
                // when pairOp==4 ends up chosen as the overall best move below, the existing
                // verify-inside-lock step (unchanged) recomputes them fresh via eval_swap_star
                // before applying, exactly as it already did before T2-lite.
                if (pairDelta < bestDelta) { bestDelta = pairDelta; bestOp = pairOp; best_j = j; }
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
                else if (bestOp == 5) verify_delta = eval_relocate2(sol, inst, i, best_j);
                else if (bestOp == 6) verify_delta = eval_relocate3(sol, inst, i, best_j);
                else if (bestOp == 7) verify_delta = eval_relocate2_rev(sol, inst, i, best_j);
                else if (bestOp == 8) verify_delta = eval_relocate3_rev(sol, inst, i, best_j);
                else if (bestOp == 9) verify_delta = eval_E21(sol, inst, i, best_j);
                else if (bestOp == 10) verify_delta = eval_E22(sol, inst, i, best_j);

                if (verify_delta < -1e-6) {
                    int old_r_i = sol.routeOf[i];
                    int old_r_j = best_j != 0 ? sol.routeOf[best_j] : -1;

                    if (bestOp == 0) apply_relocate(sol, arena, inst, i, best_j, cache, pairCache, k_max, reverseIdx_lists);
                    else if (bestOp == 1) apply_swap(sol, arena, inst, i, best_j, cache, pairCache, k_max, reverseIdx_lists);
                    else if (bestOp == 2) apply_2opt(sol, arena, inst, i, best_j, cache, pairCache, k_max, reverseIdx_lists);
                    else if (bestOp == 3) apply_2opt_star(sol, arena, inst, i, best_j, cache, pairCache, k_max, reverseIdx_lists);
                    else if (bestOp == 4) apply_swap_star(sol, arena, inst, i, best_j, best_p_i, best_s_i, best_p_j, best_s_j, cache, pairCache, k_max, reverseIdx_lists);
                    else if (bestOp == 5) apply_relocate2(sol, arena, inst, i, best_j, cache, pairCache, k_max, reverseIdx_lists);
                    else if (bestOp == 6) apply_relocate3(sol, arena, inst, i, best_j, cache, pairCache, k_max, reverseIdx_lists);
                    else if (bestOp == 7) apply_relocate2_rev(sol, arena, inst, i, best_j, cache, pairCache, k_max, reverseIdx_lists);
                    else if (bestOp == 8) apply_relocate3_rev(sol, arena, inst, i, best_j, cache, pairCache, k_max, reverseIdx_lists);
                    else if (bestOp == 9) apply_E21(sol, arena, inst, i, best_j, cache, pairCache, k_max, reverseIdx_lists);
                    else if (bestOp == 10) apply_E22(sol, arena, inst, i, best_j, cache, pairCache, k_max, reverseIdx_lists);

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
                                  int t1 = -1, int t2 = -1, std::vector<int>* routeToChunk = nullptr,
                                  std::chrono::steady_clock::time_point deadline = std::chrono::steady_clock::time_point::max()) {
        arena.pendingDelta = 0;
        size_t idx = 0;
        while (idx < nodes.size()) {
            if (std::chrono::steady_clock::now() >= deadline) break;
        int batch_end = std::min((int)idx + (int)SVCCache::CAPACITY, (int)nodes.size());
            cache.clear();
            for (size_t k = idx; k < batch_end; ++k) cache.insert(nodes[k]);
            bool improved = true;
            while (improved) {
                if (std::chrono::steady_clock::now() >= deadline) break;
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
    int chunkSize = (int)partitionInfo.globalId[chunkId].size() - 1;
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
    // T2-lite (docs/reports/009_plan_beating_filo2.md) -- disabled. Measured net-negative on
    // VDA: 68837/96331 iterations vs 93928/142751 without it (same 40s budget) -- the pair
    // cache itself was verified exactly correctness-neutral (byte-identical cost vs. the
    // uncached path on repeat runs), but a dist()-call-count comparison showed it doesn't
    // reduce dist() calls/iteration at all (~85k/67k either way), meaning the caching
    // overhead (generation checks, routeLoad snapshot comparisons, reverse-index invalidation
    // walks) is pure loss. Root cause: Step 1 above (get_top3_insertions, SWAP*'s route-walk
    // precompute) is the actual dominant cost, not the Step 2 9-operator evals this cache
    // targets -- Step 1 is NOT cached by this design at all. A follow-up would need to extend
    // caching to Step 1 (a differently-shaped, node x route cache, not node x candidate-idx)
    // to have a chance of paying off. Implementation left in place (verified correct) behind
    // this flag rather than deleted.
    constexpr bool kEnablePairCache = false;
    if (kEnablePairCache) local_granular_lists.build_reverse_index();
    int pair_cache_k_max = arena.pairCacheKMax;

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

    // Measured net-negative on VDA (see the ruin() call below) -- kept as a compile-time
    // toggle rather than deleted in case a cheaper adaptation scheme is worth revisiting.
    constexpr bool kEnableOmegaAdaptation = false;

    // T4.2: adaptive per-vertex ruin-walk length, mirroring FILO2's omega array
    // (baselines/filo2/main.cpp:180-181, 375-405). Base value matches the previous fixed
    // walk length (ceil(log(chunkSize))) so this starts identical to the old behavior and
    // only diverges as iterations adapt it. shaking_lb/ub are absolute cost thresholds
    // derived from the mean arc-length estimate, same as FILO2's shaking_lb_factor/
    // shaking_ub_factor * mean_solution_arc_cost.
    int omega_base = std::max(1, (int)std::ceil(std::log(chunkSize)));
    // Capped at 2x base rather than 4x: an isolation test (comparing T1 alone against T1+T4.2
    // on VDA) showed the 4x cap let omega drift high enough that the extra per-iteration ruin/
    // recreate/local_search cost outweighed T1's per-operation savings, *reducing* net
    // iteration throughput (94845/137989 iters/worker with T4.2 off vs 45319/57337 with it on,
    // same 40s budget). 2x keeps the adaptive-destructiveness idea T4.2 is chasing without
    // letting a single vertex's walk length balloon far past what the SA temperature schedule
    // can actually afford to search at.
    int omega_cap = std::max(omega_base * 2, 15);
    std::vector<int> omega(inst.n + 1, omega_base);
    double shaking_lb = 0.375 * avg_arc_cost_estimate;
    double shaking_ub = 0.85 * avg_arc_cost_estimate;
    std::uniform_int_distribution<int> nudge_dist(0, 1); // 0 -> -1, 1 -> +1

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
        // T2-lite: bumping this (O(1)) invalidates every pair-cache entry left over from the
        // previous SA iteration's local_search-to-convergence cascade, without touching the
        // array itself -- see PairCacheEntry's comment in ThreadArena.hpp.
        arena.pairCacheGen++;

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
        // T4.2 disabled (docs/reports/009_plan_beating_filo2.md): measured on VDA, the
        // adaptive omega array costs 35-47% of iteration throughput (larger ruin walks mean
        // proportionally more recreate/local_search work per iteration) for a cost outcome
        // statistically indistinguishable from noise across every cap tried (4x base: cost
        // WORSE than baseline; 2x base: flat vs baseline) -- it fails the plan's own Stage 2
        // gate ("did time drop >=15%?") outright. T1's per-operation savings alone deliver a
        // clean, unconfounded +19-23% iteration throughput win (94845/137989 vs 79709/112249
        // baseline, same 40s budget) and that's what's kept. The omega machinery is left in
        // place (unused) rather than deleted in case a cheaper adaptation scheme is worth
        // revisiting later.
        ruin(sol, seed_cust, arena, cache, rng, chunkSize, local_granular_lists, inst, nullptr, -1, -1, nullptr, false, nullptr);

        recreate(sol, arena, cache, inst, local_granular_lists, rng);

        // Only rescan routes ruin/recreate actually touched (see rescan_touched_routes,
        // docs/reports/005_cost_optimization.md Phase 5) -- chunkSize-bounded here, so less
        // urgent than Stage 5's full-N version, but cheap and keeps the pattern consistent.
        rescan_touched_routes(sol, arena, inst);

        bool local_search_improved = true;
        while(local_search_improved) {
            // pair_cache_k_max guard: only pass the cache if the arena's fixed-size array is
            // actually large enough for this chunk's k -- pairCache is indexed
            // nodeId*pairCacheKMax+j_idx, so k > pairCacheKMax would write out of bounds.
            // Falls back to nullptr (T2-lite's exact pre-existing uncached behavior) rather
            // than risk that; in practice local_granular_lists.k always matches the 30 both
            // are built with (main.cpp's neighborLists.build call / reserve_fixed_capacity's
            // default), so this should never actually trigger.
            if (kEnablePairCache && local_granular_lists.k <= pair_cache_k_max) {
                local_search_improved = local_search(sol, arena, cache, inst, local_granular_lists, chunkSize,
                                                      nullptr, -1, -1, nullptr,
                                                      arena.pairCache.data(), pair_cache_k_max, &local_granular_lists);
            } else {
                local_search_improved = local_search(sol, arena, cache, inst, local_granular_lists, chunkSize);
            }
        }

        Cost delta = arena.pendingDelta;

        // T4.2 omega adaptation -- disabled (see the ruin() call above for why), so this
        // block is skipped rather than deleted: omega is unused by ruin() now, and running
        // the adaptation anyway would just burn an RNG draw and a removed_count-sized loop
        // for no effect every single iteration.
        if (kEnableOmegaAdaptation) {
            double newCost = (double)sol.totalCost + (double)delta;
            double refCost = (double)sol.totalCost;
            int adj;
            if (newCost > refCost + shaking_ub) {
                adj = -1; // too destructive
            } else if (newCost < refCost + shaking_lb) {
                adj = +1; // too timid
            } else {
                adj = nudge_dist(rng) ? +1 : -1; // in the sweet spot: random nudge
            }
            for (int i = 0; i < arena.removed_count; ++i) {
                NodeId c = arena.removed_customers[i];
                int v = omega[c] + adj;
                if (v < 1) v = 1;
                if (v > omega_cap) v = omega_cap;
                omega[c] = v;
            }
        }

        if (accept_delta(delta, temperature, rng)) {
            sol.totalCost += delta;
            if (sol.totalCost < bestSol.totalCost) {
                snapshot_essential(bestSol, sol);
            }
            arena.doCount = 0; arena.undoCount = 0; arena.pendingDelta = 0;
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

namespace {
    // Greedy first-fit-decreasing bin packing lower bound on route count, restricted to this
    // chunk's own customers (FILO2's opt/bpp.hpp, ported per-chunk to match our architecture).
    int greedy_ffd_kmin(const Instance& inst, const std::vector<int>& chunkGlobalIds, int chunkSize) {
        std::vector<NodeId> customers(chunkGlobalIds.begin() + 1, chunkGlobalIds.begin() + 1 + chunkSize);
        std::sort(customers.begin(), customers.end(), [&inst](NodeId a, NodeId b) { return inst.demand[a] > inst.demand[b]; });

        std::vector<Cost> bins(chunkSize, 0);
        int used_bins = 0;
        for (NodeId c : customers) {
            Cost demand = inst.demand[c];
            for (int p = 0; p < (int)bins.size(); ++p) {
                if (bins[p] + demand <= inst.Q) {
                    bins[p] += demand;
                    if (p + 1 > used_bins) used_bins = p + 1;
                    break;
                }
            }
        }
        return used_bins;
    }

    // Opens (or reuses) an empty route slot. No mutex: T3 runs per-chunk, single-threaded
    // within that chunk, same as stage2_ils's own iteration loop.
    // Number of routes actually carrying customers. Solution::numRoutes is an allocation
    // high-water mark, not a live count: remove_customer sets routeHead[r]=0 when a route
    // empties but nothing ever decrements numRoutes, and open_route only ever increments it.
    // main.cpp reports the LIVE count (its liveRouteCount loop), so numRoutes and the number
    // the solver actually reports are different quantities. ROUTEMIN's whole job is driving
    // the live count down toward kmin, so every route-count decision in stage1_5_routemin
    // must use this, not numRoutes -- see the comment there.
    int count_live_routes(const Solution& sol) {
        int live = 0;
        for (int r = 0; r < sol.numRoutes; ++r) {
            if (sol.routeHead[r] != 0) ++live;
        }
        return live;
    }

    int open_route(Solution& sol) {
        for (int r = 0; r < sol.numRoutes; ++r) {
            if (sol.routeHead[r] == 0) return r;
        }
        int r = sol.numRoutes++;
        if (r >= (int)sol.routeHead.size()) {
            sol.routeHead.resize(r + 100, 0);
            sol.routeTail.resize(r + 100, 0);
            sol.routeLoad.resize(r + 100, 0);
        }
        return r;
    }
}

// T3 (docs/reports/009_plan_beating_filo2.md): FILO2's ROUTEMIN heuristic, ported per-chunk.
// Runs once per chunk, right after construction and before that chunk's SA/ILS loop -- our
// earlier "let a vehicle serve a different set of customers" attempts failed because they
// differed from FILO2 on every axis (see the plan's Stage 3 table): this destroys *two whole
// routes* per iteration (not one customer's local walk, which can never eliminate a route),
// tolerates temporarily-unserved customers with annealed probability (a genuine partial
// solution, not just "any feasible" — routeOf[c]==-1 already means "unserved" throughout this
// codebase, so no new infrastructure is needed), and targets a bin-packing FFD lower bound
// instead of "one fewer than we have now". Simplification vs. FILO2: reuses the existing
// per-chunk granular neighbor list (k=30) rather than building a separate wider "full
// neighborhood" (FILO2 uses up to 1500) candidate set for this pass specifically -- acceptable
// since k=30 is already the set used for every other operator throughout the run.
Solution stage1_5_routemin(Solution sol, ThreadArena& arena, SVCCache& cache,
                            const Instance& inst, const Stage0Result& partitionInfo,
                            const NeighborLists& neighborLists, int chunkId, std::mt19937& rng,
                            int max_iter) {
    int chunkSize = (int)partitionInfo.globalId[chunkId].size() - 1;
    if (chunkSize < 3) return sol; // not enough customers for a meaningful FFD/route-pair destroy

    int kmin = greedy_ffd_kmin(inst, partitionInfo.globalId[chunkId], chunkSize);
    // Live count, NOT sol.numRoutes -- see count_live_routes above. Using numRoutes here (as
    // this function originally did, across all four route-count decisions below) meant
    // ROUTEMIN was steering on a number that can only increase, so it could never observe a
    // route reduction, never hit its kmin stop condition, and its accept-fewer-routes
    // tiebreak never fired. Measured effect of that defect at P=1 on Valle-D'Aosta: routes
    // went 810 -> 831 (up), where FILO2's ROUTEMIN on the same instance goes 810 -> 801.
    if (kmin >= count_live_routes(sol)) return sol; // already at (or below) the estimated minimum

    cache.init(inst.n);
    cache.clear();

    // Two lists, deliberately different widths.
    //
    // local_granular_lists (WIDE, whatever --routemin-k gives): used for choosing the second
    // route to destroy and, critically, for the reinsertion candidate scan. Width is what
    // makes reinsertion succeed -- it decides how many distinct routes we can even consider
    // when looking for residual capacity to absorb a destroyed route's customers. Too narrow
    // and every candidate route is full, so we open a new route and route count RISES
    // (measured on VDA at P=1: k=30 -> 810->814, k=300 -> 810->805, k=1000 -> 810->800=kmin).
    //
    // local_narrow_lists (capped at 30): used ONLY for the local_search calls below. Our
    // local_search is O(k x route_length) per node pop (11 operators x k neighbours, and
    // get_top3_insertions walks a whole route), so feeding it the wide list is catastrophic
    // -- measured 69-124 s per worker for 2000 ROUTEMIN iterations at k=1000, versus ~11 s
    // to build the wide list itself. FILO2 can afford gamma=1.0 here because its local
    // search is incremental (SMD + per-operator heaps, recomputing only the moves an applied
    // move invalidates); ours is not, so it pays full width on every pop. Narrowing just the
    // local_search input keeps the reinsertion benefit at a fraction of the cost.
    // nbr[] is sorted by ascending distance, so truncating is exactly "the 30 nearest".
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

    constexpr int kRoutemimLocalSearchK = 30;
    NeighborLists local_narrow_lists;
    local_narrow_lists.k = std::min(local_granular_lists.k, kRoutemimLocalSearchK);
    local_narrow_lists.nbr.assign(inst.n + 1, std::vector<NodeId>());
    for (int i = 1; i <= chunkSize; ++i) {
        NodeId global_i = partitionInfo.globalId[chunkId][i];
        const auto& wide = local_granular_lists.nbr[global_i];
        int take = std::min((int)wide.size(), kRoutemimLocalSearchK);
        local_narrow_lists.nbr[global_i].assign(wide.begin(), wide.begin() + take);
    }

    Solution bestSol = sol;
    Solution current = sol;
    int bestLive = count_live_routes(bestSol);

    const double t_base = 1.0, t_end = 0.01;
    double t = t_base;
    double cool = std::pow(t_end / t_base, 1.0 / max_iter);

    std::vector<NodeId> removed, still_removed;
    removed.reserve(chunkSize);
    still_removed.reserve(chunkSize);

    std::uniform_int_distribution<int> dist_cust(1, chunkSize);
    std::uniform_real_distribution<double> uniform01(0.0, 1.0);
    std::vector<int> candidate_routes;

    for (int iter = 0; iter < max_iter; ++iter) {
        cache.clear();

        // Seed customer -> its route, plus (if found) one neighboring route.
        NodeId seed;
        int guard = 0;
        do {
            seed = partitionInfo.globalId[chunkId][dist_cust(rng)];
        } while (current.routeOf[seed] == -1 && ++guard < chunkSize * 2);
        if (current.routeOf[seed] == -1) continue; // chunk fully unserved (shouldn't happen, but don't hang)

        std::vector<int> selected_routes;
        selected_routes.push_back(current.routeOf[seed]);
        int sk = std::min((int)local_granular_lists.nbr[seed].size(), local_granular_lists.k);
        for (int idx = 0; idx < sk; ++idx) {
            NodeId v = local_granular_lists.nbr[seed][idx];
            if (current.routeOf[v] == -1) continue;
            int r = current.routeOf[v];
            if (r != selected_routes[0]) { selected_routes.push_back(r); break; }
        }

        removed.clear();
        removed.insert(removed.end(), still_removed.begin(), still_removed.end());
        still_removed.clear();

        // Destroy the selected routes entirely.
        for (int r : selected_routes) {
            NodeId curr = current.routeHead[r];
            while (curr != 0) {
                NodeId next = current.succ[curr];
                remove_customer(current, curr, arena, inst);
                removed.push_back(curr);
                curr = next;
            }
        }

        if (std::uniform_int_distribution<int>(0, 1)(rng) == 0) {
            std::sort(removed.begin(), removed.end(), [&inst](NodeId a, NodeId b) { return inst.demand[a] > inst.demand[b]; });
        } else {
            std::shuffle(removed.begin(), removed.end(), rng);
        }

        for (NodeId c : removed) {
            candidate_routes.clear();
            int ck = std::min((int)local_granular_lists.nbr[c].size(), local_granular_lists.k);
            for (int idx = 0; idx < ck; ++idx) {
                NodeId v = local_granular_lists.nbr[c][idx];
                if (current.routeOf[v] == -1) continue;
                candidate_routes.push_back(current.routeOf[v]);
            }
            std::sort(candidate_routes.begin(), candidate_routes.end());
            candidate_routes.erase(std::unique(candidate_routes.begin(), candidate_routes.end()), candidate_routes.end());

            Cost bestDelta = 999999999;
            NodeId bestPred = 0, bestSucc = 0;
            int bestRoute = -1;
            Top3Insertions top3;
            for (int r : candidate_routes) {
                if (current.routeLoad[r] + inst.demand[c] > inst.Q) continue;
                get_top3_insertions(current, inst, c, r, top3);
                if (top3.count > 0 && top3.delta[0] < bestDelta) {
                    bestDelta = top3.delta[0];
                    bestPred = top3.pos_pred[0];
                    bestSucc = top3.pos_succ[0];
                    bestRoute = r;
                }
            }

            if (bestRoute != -1) {
                insert_customer(current, c, bestPred, bestSucc, bestRoute, arena, inst);
                cache.insert(c);
            } else {
                double roll = uniform01(rng);
                if (roll > t || count_live_routes(current) < kmin) {
                    int r = open_route(current);
                    current.routeLoad[r] = 0;
                    insert_customer(current, c, 0, 0, r, arena, inst);
                    update_route_info(current, r, inst);
                    cache.insert(c);
                } else {
                    still_removed.push_back(c);
                }
            }
        }

        // Must run BEFORE local_search, exactly like stage2_ils's ruin+recreate ->
        // rescan_touched_routes -> local_search ordering: local_search's eval_2opt_star
        // depends on cumLoad being current (see its own comment on why), and the
        // destroy-routes + reinsertion loops above only kept routeLoad/costToPred
        // incrementally correct, not cumLoad/routePosition (those are update_route_info's
        // job). Calling this after local_search instead (as an earlier version of this
        // function did) let local_search read stale cumLoad for routes just reinserted into,
        // producing a real capacity-check false-pass -- caught via a genuine [FATAL]
        // insert_customer overflow in stage2_ils several iterations later on X-n1001-k43.
        rescan_touched_routes(current, arena, inst);

        bool improved = true;
        while (improved) {
            // NARROW list here on purpose -- see the two-list comment above.
            improved = local_search(current, arena, cache, inst, local_narrow_lists, chunkSize);
        }
        // remove_customer/insert_customer/local_search only accumulate into
        // arena.pendingDelta (see local_search's own doc comment above) -- the caller must
        // fold it into totalCost itself, same as stage2_ils does for its own ruin/recreate/
        // local_search cascade.
        current.totalCost += arena.pendingDelta;

        if (still_removed.empty()) {
            int currentLive = count_live_routes(current);
            if (current.totalCost < bestSol.totalCost ||
                (current.totalCost == bestSol.totalCost && currentLive < bestLive)) {
                bestSol = current;
                bestLive = currentLive;
                if (bestLive <= kmin) break; // hit the target, stop early
            }
        }

        if (current.totalCost > bestSol.totalCost) {
            // Roll back to the best-known solution wholesale rather than replaying this
            // iteration's do/undo log -- ROUTEMIN's iteration budget (default ~1000, vs SA's
            // tens of thousands) makes an O(n) struct copy here cheap, and it sidesteps
            // having to reason about undo correctness across a variable-length sequence of
            // whole-route destructions + probabilistic reinsertions + a full local_search
            // pass, unlike stage2_ils's fixed single-ruin-then-recreate shape.
            current = bestSol;
            still_removed.clear();
        }

        t *= cool;
        arena.doCount = 0; arena.undoCount = 0; arena.pendingDelta = 0;
    }

#ifdef ROUTEMIN_DEBUG_CHECK
    for (int r = 0; r < bestSol.numRoutes; ++r) {
        if (bestSol.routeHead[r] == 0) continue;
        Cost load = 0;
        NodeId curr = bestSol.routeHead[r];
        int pos = 1;
        while (curr != 0) {
            load += inst.demand[curr];
            if (bestSol.cumLoad[curr] != load) {
                printf("[ROUTEMIN_DEBUG] route %d node %d cumLoad desync: tracked=%lld true=%lld\n", r, curr, (long long)bestSol.cumLoad[curr], (long long)load);
            }
            if (bestSol.routePosition[curr] != pos) {
                printf("[ROUTEMIN_DEBUG] route %d node %d routePosition desync: tracked=%d true=%d\n", r, curr, bestSol.routePosition[curr], pos);
            }
            if (bestSol.costToPred[curr] != dist(inst, bestSol.pred[curr], curr)) {
                printf("[ROUTEMIN_DEBUG] route %d node %d costToPred desync: tracked=%lld true=%lld\n", r, curr, (long long)bestSol.costToPred[curr], (long long)dist(inst, bestSol.pred[curr], curr));
            }
            pos++;
            curr = bestSol.succ[curr];
        }
        if (load != bestSol.routeLoad[r]) {
            printf("[ROUTEMIN_DEBUG] route %d routeLoad desync: tracked=%lld true=%lld\n", r, (long long)bestSol.routeLoad[r], (long long)load);
        }
    }
#endif
    return bestSol;
}

Cost stage3_healing_ils_pass(Solution& globalSolution, ThreadArena& arena, SVCCache& cache,
                             const Instance& inst, const NeighborLists& neighborLists,
                             const Stage0Result& partitionInfo,
                             const std::vector<int>& boundaryList,
                             int t1, int t2, std::mt19937& rng,
                             std::vector<int>* routeToChunk = nullptr) {
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
        
        int num_strings = 6;
        for (int s_idx = 0; s_idx < num_strings; ++s_idx) {
            NodeId s = (s_idx == 0) ? seed_cust : boundaryList[dist_cust(rng)];
            ruin(globalSolution, s, arena, cache, rng, virtual_chunk_size, local_granular_lists, inst, &route_creation_mutex, t1, t2, routeToChunk, s_idx > 0);
        }
        
        auto t_ruin = std::chrono::high_resolution_clock::now();
        
        recreate(globalSolution, arena, cache, inst, local_granular_lists, rng, &route_creation_mutex, t1, t2, routeToChunk);

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
            arena.doCount = 0; arena.undoCount = 0; arena.pendingDelta = 0;
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
    // Boost T0 to allow the solver to accept intermediate 
    // cost spikes caused by perturbations.
    double T0 = 0.5 * avg_arc_cost_estimate;
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
    //
    // Reuses sweepStart (not a fresh std::chrono::steady_clock::now() here) -- this actually
    // implements the "one clock, one budget" comment above sweepStart's declaration. A fresh
    // capture here previously gave the pre-loop full_sweep_local_search its own full
    // g_stage5_time_budget_ms AND then gave this loop another full budget on top, up to
    // doubling Stage 5's real wall time (measured: 88.6 s against a 45 s budget at Lazio).
    auto stageStart = sweepStart;
    for (int iter = 0; useTimeBudget || iter < max_iterations; ++iter) {
        if (useTimeBudget) {
            double elapsed_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - stageStart).count();
            if (iter % 100 == 0) {
                printf("SA Iter: %d, elapsed: %.1f ms\n", iter, elapsed_ms); fflush(stdout);
            }
            if (elapsed_ms >= g_stage5_time_budget_ms) {
                printf("SA time budget reached! Exiting loop...\n"); fflush(stdout);
                break;
            }
            
            static int current_cycle = 0;
            if (iter == 0) current_cycle = 0;
            double cycle_ms = g_stage5_time_budget_ms / 3.0; // 3 Sawtooth cycles
            int next_cycle = (int)(elapsed_ms / cycle_ms);
            if (next_cycle > current_cycle) {
                globalSolution = bestSol;
                // bestSol is maintained via snapshot_essential, which deliberately excludes
                // routePosition/cumLoad (report 006 Phase 2.3 -- they're pure derived caches,
                // expensive to keep copying every improving iteration). That's fine for
                // bestSol itself, since finalize_solution_derived_fields() regenerates them
                // once at the very end of this function. But copying bestSol INTO the live
                // globalSolution here mid-loop means globalSolution.cumLoad/routePosition are
                // now whatever bestSol's happened to be -- i.e. stale/unrelated to the
                // pred/succ/routeHead structure just copied in, not just "a bit behind". Any
                // route local_search doesn't happen to touch again after this reset keeps
                // that stale cumLoad forever, letting eval_2opt_star's capacity check pass
                // genuinely-infeasible moves against it (confirmed via a reproducible VDA
                // crash, further into Stage 5 than the Stage1_Construction bug). Full
                // solution is small (a few hundred to a few thousand routes) and this branch
                // only fires twice per run (3 sawtooth cycles), so an O(N) refresh here is negligible.
                finalize_solution_derived_fields(globalSolution, inst);
                stagnation = 0;
                current_cycle = next_cycle;
            }
            double ms_in_current_cycle = std::fmod(elapsed_ms, cycle_ms);
            temperature = T0 * std::pow(Tf / T0, ms_in_current_cycle / cycle_ms);
        }
        arena.doCount = 0;
        arena.undoCount = 0;
        arena.pendingDelta = 0;
        arena.removed_count = 0;
        
        int prevNumRoutes = globalSolution.numRoutes;
        
        if (stagnation > 50) {
            std::uniform_real_distribution<double> dist_shock(0.0, 1.0);
            double shock_val = dist_shock(rng);
            if (shock_val < 0.05) {
                int smallest_route = -1;
                int min_nodes = 999999;
                for (int r = 0; r < globalSolution.numRoutes; ++r) {
                    if (globalSolution.routeHead[r] == 0) continue;
                    int nodes = 0;
                    NodeId curr = globalSolution.routeHead[r];
                    int loops = 0;
                    while (curr != 0) { 
                        if (++loops > inst.n + 2) { printf("[FATAL] HANG in stage3 smallest route nodes\n"); fflush(stdout); exit(1); }
                        nodes++; curr = globalSolution.succ[curr]; 
                    }
                    if (nodes < min_nodes) {
                        min_nodes = nodes;
                        smallest_route = r;
                    }
                }
                if (smallest_route != -1) {
                    NodeId curr = globalSolution.routeHead[smallest_route];
                    int loops = 0;
                    while (curr != 0) {
                        if (++loops > inst.n + 2) { printf("[FATAL] HANG in stage3 smallest route remove\n"); fflush(stdout); exit(1); }
                        NodeId nxt = globalSolution.succ[curr];
                        remove_customer(globalSolution, curr, arena, inst);
                        cache.insert(curr);
                        arena.removed_customers[arena.removed_count++] = curr;
                        curr = nxt;
                    }
                }
            } else if (shock_val < 0.15) {
                std::vector<int> active_routes;
                for (int r = 0; r < globalSolution.numRoutes; ++r) {
                    if (globalSolution.routeLoad[r] > 0) active_routes.push_back(r);
                }
                if (!active_routes.empty()) {
                    std::uniform_int_distribution<int> dist_cust(1, inst.n);
                    for (int s_idx = 0; s_idx < 20; ++s_idx) {
                        ruin(globalSolution, dist_cust(rng), arena, cache, rng, inst.n, neighborLists, inst, nullptr, -1, -1, nullptr, true);
                    }
                }
            } else {
                std::uniform_int_distribution<int> dist_cust(1, inst.n);
                NodeId seed_cust = dist_cust(rng);
                int num_strings = 40;
                ruin(globalSolution, seed_cust, arena, cache, rng, inst.n, neighborLists, inst, nullptr, -1, -1, nullptr, false);
                int strings_ruined = 1;
                int max_k = std::min((int)neighborLists.nbr[seed_cust].size(), neighborLists.k);
                for (int j_idx = 0; j_idx < max_k && strings_ruined < num_strings; ++j_idx) {
                    NodeId s = neighborLists.nbr[seed_cust][j_idx];
                    if (globalSolution.routeOf[s] != -1) {
                        ruin(globalSolution, s, arena, cache, rng, inst.n, neighborLists, inst, nullptr, -1, -1, nullptr, true);
                        strings_ruined++;
                    }
                }
            }
        } else {
            std::uniform_int_distribution<int> dist_cust(1, inst.n);
            NodeId seed_cust = dist_cust(rng);
            int num_strings = 40;
            ruin(globalSolution, seed_cust, arena, cache, rng, inst.n, neighborLists, inst, nullptr, -1, -1, nullptr, false);
            int strings_ruined = 1;
            int max_k = std::min((int)neighborLists.nbr[seed_cust].size(), neighborLists.k);
            for (int j_idx = 0; j_idx < max_k && strings_ruined < num_strings; ++j_idx) {
                NodeId s = neighborLists.nbr[seed_cust][j_idx];
                if (globalSolution.routeOf[s] != -1) {
                    ruin(globalSolution, s, arena, cache, rng, inst.n, neighborLists, inst, nullptr, -1, -1, nullptr, true);
                    strings_ruined++;
                }
            }
        }

        auto t_ruin = std::chrono::high_resolution_clock::now();

        recreate(globalSolution, arena, cache, inst, neighborLists, rng, nullptr);

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
        int local_search_calls = 0;
        while(local_search_improved) {
            local_search_calls++;
            local_search_improved = local_search(globalSolution, arena, cache, inst, neighborLists, inst.n, nullptr);
            if (useTimeBudget) {
                double elapsed_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - stageStart).count();
                if (local_search_calls % 10 == 0) {
                    printf("  local_search_calls: %d, elapsed: %.1f ms\n", local_search_calls, elapsed_ms); fflush(stdout);
                }
                if (elapsed_ms >= g_stage5_time_budget_ms) {
                    printf("  LS time budget reached! Exiting while loop...\n"); fflush(stdout);
                    break;
                }
            }
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
            arena.doCount = 0; arena.undoCount = 0; arena.pendingDelta = 0;
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
