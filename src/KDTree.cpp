#include "KDTree.hpp"
#include <algorithm>
#include <queue>
#include <cmath>
#include <thread>
#include <vector>

namespace {
    int build_recursive(std::vector<KDNode>& nodes, std::vector<NodeId>& ids, int start, int end, int depth, const Instance& inst) {
        if (start >= end) return -1;

        int axis = depth % 2;
        int mid = start + (end - start) / 2;

        auto comp = [&](NodeId a, NodeId b) {
            if (axis == 0) return inst.x[a] < inst.x[b];
            return inst.y[a] < inst.y[b];
        };

        std::nth_element(ids.begin() + start, ids.begin() + mid, ids.begin() + end, comp);

        KDNode node;
        node.id = ids[mid];
        node.axis = axis;
        
        int node_idx = nodes.size();
        nodes.push_back(node);
        
        int left_idx = build_recursive(nodes, ids, start, mid, depth + 1, inst);
        int right_idx = build_recursive(nodes, ids, mid + 1, end, depth + 1, inst);
        
        nodes[node_idx].left = left_idx;
        nodes[node_idx].right = right_idx;
        
        return node_idx;
    }
}

void KDTree::build(const Instance& inst) {
    std::vector<NodeId> ids(inst.n + 1);
    for (int i = 0; i <= inst.n; ++i) {
        ids[i] = i;
    }
    nodes.clear();
    nodes.reserve(inst.n + 1);
    root = build_recursive(nodes, ids, 0, ids.size(), 0, inst);
}

namespace {
    // Queries nodes [lo, hi) against the (already-built, read-only) tree. Independent per
    // node: own priority_queue, writes only nbr[i] for i in this thread's own range -- no
    // shared mutable state, so this is safe to call from multiple threads concurrently over
    // disjoint ranges with no locking.
    void knn_query_range(const Instance& inst, const KDTree& tree, int k, int lo, int hi,
                          std::vector<std::vector<NodeId>>& nbr) {
        for (int i = lo; i < hi; ++i) {
            std::priority_queue<std::pair<Cost, NodeId>> pq;

            auto search = [&](auto& self, int node_idx, int depth) -> void {
                if (node_idx == -1) return;
                const auto& node = tree.nodes[node_idx];

                if (node.id != i) {
                    Cost d = dist(inst, i, node.id);
                    if (pq.size() < k || d < pq.top().first) {
                        pq.push({d, node.id});
                        if (pq.size() > k) pq.pop();
                    }
                }

                double dist_to_plane = 0;
                if (node.axis == 0) {
                    dist_to_plane = (double)inst.x[i] - (double)inst.x[node.id];
                } else {
                    dist_to_plane = (double)inst.y[i] - (double)inst.y[node.id];
                }

                int near_child = dist_to_plane < 0 ? node.left : node.right;
                int far_child = dist_to_plane < 0 ? node.right : node.left;

                self(self, near_child, depth + 1);

                if (pq.size() < k || std::abs(dist_to_plane) < pq.top().first) {
                    self(self, far_child, depth + 1);
                }
            };

            search(search, tree.root, 0);

            std::vector<NodeId>& curr_nbr = nbr[i];
            curr_nbr.reserve(pq.size());
            while (!pq.empty()) {
                curr_nbr.push_back(pq.top().second);
                pq.pop();
            }
            std::reverse(curr_nbr.begin(), curr_nbr.end()); // sorted by distance ascending
        }
    }
}

void NeighborLists::build(const Instance& inst, int k_neighbors, int num_threads) {
    k = k_neighbors;
    nbr.assign(inst.n + 1, std::vector<NodeId>());

    KDTree tree;
    tree.build(inst);

    if (num_threads <= 1 || inst.n < 1000) {
        // Small instances: thread setup overhead isn't worth it.
        knn_query_range(inst, tree, k, 0, inst.n + 1, nbr);
        symmetrize(inst, k);
        return;
    }

    std::vector<std::thread> threads;
    int chunk = (inst.n + 1 + num_threads - 1) / num_threads;
    for (int t = 0; t < num_threads; ++t) {
        int lo = t * chunk;
        int hi = std::min(inst.n + 1, lo + chunk);
        if (lo >= hi) break;
        threads.emplace_back(knn_query_range, std::cref(inst), std::cref(tree), k, lo, hi, std::ref(nbr));
    }
    for (auto& th : threads) th.join();

    symmetrize(inst, k);
}

// kNN as computed above is asymmetric: j in nbr[i] does not imply i in nbr[j].
// For each i, offer i as a candidate to every j in nbr[i] that doesn't already have it,
// then re-sort each affected list by distance and truncate back to k. This keeps list
// sizes and the existing min(size(), k) call-site convention unchanged, while letting
// genuinely close symmetric pairs unseat a farther one-directional neighbor.
void NeighborLists::symmetrize(const Instance& inst, int k_neighbors) {
    int n1 = inst.n + 1;
    std::vector<std::vector<NodeId>> extra(n1);
    for (int i = 0; i < n1; ++i) {
        for (NodeId j : nbr[i]) {
            bool present = false;
            for (NodeId x : nbr[j]) {
                if (x == i) { present = true; break; }
            }
            if (!present) extra[j].push_back(i);
        }
    }
    for (int j = 0; j < n1; ++j) {
        if (extra[j].empty()) continue;
        nbr[j].insert(nbr[j].end(), extra[j].begin(), extra[j].end());
        std::sort(nbr[j].begin(), nbr[j].end(), [&](NodeId a, NodeId b) {
            return dist(inst, j, a) < dist(inst, j, b);
        });
        if ((int)nbr[j].size() > k_neighbors) nbr[j].resize(k_neighbors);
    }
}

void NeighborLists::build_reverse_index() {
    reverseIdx.assign(nbr.size(), std::vector<std::pair<NodeId,int>>());
    for (int i = 0; i < (int)nbr.size(); ++i) {
        for (int j_idx = 0; j_idx < (int)nbr[i].size(); ++j_idx) {
            NodeId v = nbr[i][j_idx];
            reverseIdx[v].emplace_back((NodeId)i, j_idx);
        }
    }
}
