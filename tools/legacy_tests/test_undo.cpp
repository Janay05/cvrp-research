#include <iostream>
#include <vector>
#include <cassert>

typedef int NodeId;
typedef long long Cost;

struct Instance {
    int n;
    int Q;
    std::vector<int> demand;
};

struct DoUndoEntry {
    enum Type { INSERT, REMOVE } type;
    NodeId customer;
    NodeId prevPred, prevSucc;
    NodeId newPred, newSucc;
    int prevRoute, newRoute;
    Cost costDelta;
};

struct ThreadArena {
    std::vector<DoUndoEntry> undoList;
    int undoCount = 0;
    std::vector<int> route_modified_gen;
    std::vector<int> modified_routes_list;
    int modified_routes_gen = 0;
    int doCount = 0;
    int pendingDelta = 0;
};

struct Solution {
    int numRoutes;
    std::vector<NodeId> routeHead, routeTail;
    std::vector<Cost> routeLoad;
    std::vector<NodeId> succ, pred;
    std::vector<int> routeOf;
};

void insert_customer(Solution& sol, NodeId c, NodeId p, NodeId s, int route, ThreadArena& arena, const Instance& inst) {
    DoUndoEntry undo_entry;
    undo_entry.type = DoUndoEntry::REMOVE;
    undo_entry.customer = c;
    undo_entry.prevPred = p; undo_entry.prevSucc = s;
    undo_entry.newPred = p; undo_entry.newSucc = s;
    undo_entry.prevRoute = route; undo_entry.newRoute = -1;
    
    arena.undoList[arena.undoCount++] = undo_entry;

    sol.succ[p] = c; sol.pred[c] = p;
    sol.succ[c] = s; sol.pred[s] = c;
    
    sol.routeOf[c] = route;
    sol.routeLoad[route] += inst.demand[c];
    
    if (p == 0) sol.routeHead[route] = c;
    if (s == 0) sol.routeTail[route] = c;
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
    
    arena.undoList[arena.undoCount++] = undo_entry;

    if (p != 0) sol.succ[p] = s;
    if (s != 0) sol.pred[s] = p;
    sol.routeLoad[sol.routeOf[c]] -= inst.demand[c];
    
    if (p == 0) sol.routeHead[sol.routeOf[c]] = s;
    if (s == 0) sol.routeTail[sol.routeOf[c]] = p;
    
    sol.routeOf[c] = -1;
    sol.pred[c] = 0; sol.succ[c] = 0;
}

void apply_undo_list(Solution& sol, ThreadArena& arena, const Instance& inst) {
    for (int i = arena.undoCount - 1; i >= 0; --i) {
        const auto& entry = arena.undoList[i];
        if (entry.type == DoUndoEntry::INSERT) {
            NodeId c = entry.customer; NodeId p = entry.newPred; NodeId s = entry.newSucc; int route = entry.newRoute;
            sol.succ[p] = c; sol.pred[c] = p; sol.succ[c] = s; sol.pred[s] = c;
            sol.routeOf[c] = route;
            sol.routeLoad[route] += inst.demand[c];
            if (p == 0) sol.routeHead[route] = c;
            if (s == 0) sol.routeTail[route] = c;
        } else {
            NodeId c = entry.customer; int route = entry.prevRoute;
            NodeId p = sol.pred[c]; NodeId s = sol.succ[c];
            
            if (p != 0) sol.succ[p] = s;
            if (s != 0) sol.pred[s] = p;
            sol.pred[c] = 0; sol.succ[c] = 0;
            
            if (route != -1) {
                sol.routeOf[c] = -1;
                sol.routeLoad[route] -= inst.demand[c];
                if (p == 0) sol.routeHead[route] = s;
                if (s == 0) sol.routeTail[route] = p;
            }
        }
    }
    arena.doCount = 0; arena.undoCount = 0; arena.pendingDelta = 0;
}

void verify(const Solution& sol, const Instance& inst) {
    for (int r = 0; r < sol.numRoutes; ++r) {
        Cost load = 0;
        NodeId curr = sol.routeHead[r];
        while (curr != 0) {
            load += inst.demand[curr];
            curr = sol.succ[curr];
        }
        if (load != sol.routeLoad[r]) {
            std::cout << "DESYNC! route=" << r << " trueLoad=" << load << " trackedLoad=" << sol.routeLoad[r] << std::endl;
            exit(1);
        }
    }
}

int main() {
    Instance inst;
    inst.n = 5;
    inst.Q = 100;
    inst.demand = {0, 10, 20, 30, 40, 50}; // Depot has 0
    
    Solution sol;
    sol.numRoutes = 2;
    sol.routeHead = {1, 4};
    sol.routeTail = {3, 5};
    sol.routeLoad = {60, 90};
    sol.succ = {0, 2, 3, 0, 5, 0};
    sol.pred = {0, 0, 1, 2, 0, 4};
    sol.routeOf = {-1, 0, 0, 0, 1, 1};
    
    ThreadArena arena;
    arena.undoList.resize(100);
    
    verify(sol, inst);
    std::cout << "Initial OK" << std::endl;
    
    // Test apply_swap
    NodeId i = 2; // in route 0
    NodeId j = 5; // in route 1
    NodeId p_i = sol.pred[i], s_i = sol.succ[i];
    NodeId p_j = sol.pred[j], s_j = sol.succ[j];
    int r_i = sol.routeOf[i], r_j = sol.routeOf[j];
    
    remove_customer(sol, i, arena, inst);
    remove_customer(sol, j, arena, inst);
    insert_customer(sol, j, p_i, s_i, r_i, arena, inst);
    insert_customer(sol, i, p_j, s_j, r_j, arena, inst);
    
    verify(sol, inst);
    std::cout << "After apply_swap OK" << std::endl;
    
    apply_undo_list(sol, arena, inst);
    verify(sol, inst);
    std::cout << "After apply_undo_list OK" << std::endl;
    
    return 0;
}
