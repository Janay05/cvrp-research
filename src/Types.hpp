#pragma once
#include <cstdint>
#include <vector>
#include <cmath>

typedef int64_t Coord; // scaled integer coordinate
typedef int64_t Cost;  // scaled integer edge cost
typedef int32_t NodeId; // 0 = depot, 1..n = customers

struct Instance {
    int n; // number of customers (excludes depot)
    Coord Q; // vehicle capacity
    std::vector<Coord> x, y; // size n+1, index 0 = depot
    std::vector<int64_t> demand; // size n+1, demand[0] = 0
};

extern thread_local long long thread_dist_calls;
#include <atomic>
extern std::atomic<long long> global_dist_calls;

inline Cost dist(const Instance& inst, NodeId a, NodeId b) {
#ifdef PROFILE_DIST
    thread_dist_calls++;
#endif
    double dx = (double)(inst.x[a] - inst.x[b]);
    double dy = (double)(inst.y[a] - inst.y[b]);
    return (Cost) std::llround(std::sqrt(dx*dx + dy*dy));
}
