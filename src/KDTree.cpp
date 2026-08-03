#include "KDTree.hpp"
#include <algorithm>
#include <queue>
#include <cmath>

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
    std::vector<NodeId> ids(inst.n);
    for (int i = 1; i <= inst.n; ++i) {
        ids[i - 1] = i;
    }
    nodes.clear();
    nodes.reserve(inst.n);
    root = build_recursive(nodes, ids, 0, ids.size(), 0, inst);
}

void NeighborLists::build(const Instance& inst, int k_neighbors) {
    k = k_neighbors;
    nbr.assign(inst.n + 1, std::vector<NodeId>());

    KDTree tree;
    tree.build(inst);

    for (int i = 1; i <= inst.n; ++i) {
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
