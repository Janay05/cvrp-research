#include "Stage4_5_CleanupPolish.hpp"
#include "Stage2_ILS.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>

void stage4_route_cleanup(Solution& globalSolution, const Instance& inst, const NeighborLists& neighborLists) {
    for (size_t r = 0; r < globalSolution.routeHead.size(); ++r) {
        if (globalSolution.routeHead[r] == 0) continue; 
        
        int num_customers = 0;
        NodeId curr = globalSolution.routeHead[r];
        while (curr != 0) {
            num_customers++;
            curr = globalSolution.succ[curr];
        }
        
        if (globalSolution.routeLoad[r] < 0.2 * inst.Q || num_customers <= 2) {
            curr = globalSolution.routeHead[r];
            while (curr != 0) {
                NodeId c = curr;
                curr = globalSolution.succ[curr]; 
                
                Cost best_delta = 1e9;
                NodeId best_p = 0;
                NodeId best_s = 0;
                int best_route = -1;
                
                int k = std::min((int)neighborLists.nbr[c].size(), neighborLists.k);
                for (int j_idx = 0; j_idx < k; ++j_idx) {
                    NodeId nbr = neighborLists.nbr[c][j_idx];
                    int r_nbr = globalSolution.routeOf[nbr];
                    if (r_nbr == -1 || r_nbr == r) continue;
                    if (globalSolution.routeLoad[r_nbr] + inst.demand[c] > inst.Q) continue;
                    
                    NodeId p = nbr;
                    NodeId s = globalSolution.succ[nbr];
                    Cost delta_after = dist(inst, p, c) + dist(inst, c, s) - dist(inst, p, s);
                    if (delta_after < best_delta) {
                        best_delta = delta_after;
                        best_p = p;
                        best_s = s;
                        best_route = r_nbr;
                    }
                    
                    p = globalSolution.pred[nbr];
                    s = nbr;
                    Cost delta_before = dist(inst, p, c) + dist(inst, c, s) - dist(inst, p, s);
                    if (delta_before < best_delta) {
                        best_delta = delta_before;
                        best_p = p;
                        best_s = s;
                        best_route = r_nbr;
                    }
                }
                
                NodeId orig_p = globalSolution.pred[c];
                NodeId orig_s = globalSolution.succ[c];
                Cost rem_cost = dist(inst, orig_p, c) + dist(inst, c, orig_s) - dist(inst, orig_p, orig_s);
                
                if (best_route != -1 && best_delta - rem_cost < 1e-6) {
                    globalSolution.succ[orig_p] = orig_s;
                    globalSolution.pred[orig_s] = orig_p;
                    if (orig_p == 0) globalSolution.routeHead[r] = orig_s;
                    if (orig_s == 0) globalSolution.routeTail[r] = orig_p;
                    globalSolution.routeLoad[r] -= inst.demand[c];
                    globalSolution.totalCost -= rem_cost;
                    
                    globalSolution.pred[c] = best_p;
                    globalSolution.succ[c] = best_s;
                    globalSolution.succ[best_p] = c;
                    globalSolution.pred[best_s] = c;
                    if (best_p == 0) globalSolution.routeHead[best_route] = c;
                    if (best_s == 0) globalSolution.routeTail[best_route] = c;
                    globalSolution.routeLoad[best_route] += inst.demand[c];
                    
                    globalSolution.routeOf[c] = best_route;
                    globalSolution.totalCost += best_delta;
                }
            }
        }
    }
    
    std::vector<NodeId> new_routeHead;
    std::vector<NodeId> new_routeTail;
    std::vector<Cost> new_routeLoad;
    
    for (size_t r = 0; r < globalSolution.routeHead.size(); ++r) {
        if (globalSolution.routeHead[r] != 0) {
            int new_r = new_routeHead.size();
            new_routeHead.push_back(globalSolution.routeHead[r]);
            new_routeTail.push_back(globalSolution.routeTail[r]);
            new_routeLoad.push_back(globalSolution.routeLoad[r]);
            
            NodeId curr = globalSolution.routeHead[r];
            while (curr != 0) {
                globalSolution.routeOf[curr] = new_r;
                curr = globalSolution.succ[curr];
            }
        }
    }
    
    globalSolution.routeHead = new_routeHead;
    globalSolution.routeTail = new_routeTail;
    globalSolution.routeLoad = new_routeLoad;
    globalSolution.numRoutes = new_routeHead.size();
    
    // Stage 4 modifies routes (relocates customers) but leaves cumLoad and routePosition
    // stale. Stage 5 Polish immediately follows and relies on cumLoad for capacity checks 
    // (eval_2opt_star). If we don't rescan the routes here, Stage 5 will evaluate moves against 
    // stale load data and silently construct over-capacity routes, which will then trigger 
    // fatal assertions in subsequent insert_customer calls.
    for (int r = 0; r < globalSolution.numRoutes; ++r) {
        update_route_info(globalSolution, r, inst);
    }
}
