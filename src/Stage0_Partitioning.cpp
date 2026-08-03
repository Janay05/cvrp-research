#include "Stage0_Partitioning.hpp"
#include "MathUtils.hpp"
#include <algorithm>
#include <cmath>

Stage0Result run_stage0(const Instance& inst, const NeighborLists& neighborLists, int num_chunks) {
    Stage0Result res;
    res.numChunks = num_chunks;
    int n = inst.n;
    
    // Compute Hilbert index for every customer
    Coord min_x = inst.x[1], max_x = inst.x[1];
    Coord min_y = inst.y[1], max_y = inst.y[1];
    for (int i = 2; i <= n; ++i) {
        min_x = std::min(min_x, inst.x[i]);
        max_x = std::max(max_x, inst.x[i]);
        min_y = std::min(min_y, inst.y[i]);
        max_y = std::max(max_y, inst.y[i]);
    }
    
    uint32_t order = 16;
    Coord range_x = max_x - min_x;
    Coord range_y = max_y - min_y;
    Coord max_range = std::max(range_x, range_y);
    if (max_range == 0) max_range = 1;
    
    std::vector<std::pair<uint64_t, int>> hilbertVals(n);
    for (int i = 1; i <= n; ++i) {
        uint32_t norm_x = (uint32_t)(((inst.x[i] - min_x) * ((1ull << order) - 1)) / max_range);
        uint32_t norm_y = (uint32_t)(((inst.y[i] - min_y) * ((1ull << order) - 1)) / max_range);
        hilbertVals[i - 1] = {xy2d(order, norm_x, norm_y), i};
    }
    
    // Sort by Hilbert index
    std::sort(hilbertVals.begin(), hilbertVals.end());
    
    std::vector<int> sortedIds(n);
    for (int i = 0; i < n; ++i) sortedIds[i] = hilbertVals[i].second;
    
    // Split into chunks
    res.chunkOf.assign(n + 1, -1);
    int target_chunk_size = (n + num_chunks - 1) / num_chunks;
    
    res.chunkSize.assign(num_chunks, 0);
    res.globalId.assign(num_chunks, std::vector<int>());
    res.localId.assign(n + 1, -1);
    
    int current_chunk = 0;
    for (int i = 0; i < n; ++i) {
        int id = sortedIds[i];
        if (res.chunkSize[current_chunk] == target_chunk_size && current_chunk < num_chunks - 1) {
            current_chunk++;
        }
        res.chunkOf[id] = current_chunk;
        res.localId[id] = res.chunkSize[current_chunk] + 1; // 1-indexed local ids
        if (res.globalId[current_chunk].empty()) {
            res.globalId[current_chunk].push_back(0); // Depot map
        }
        res.globalId[current_chunk].push_back(id);
        res.chunkSize[current_chunk]++;
    }
    
    // Boundary tagging
    res.isBoundary.assign(n + 1, false);
    res.boundaryChunkPair.assign(n + 1, std::vector<int>());
    
    std::vector<double> avgKnnEdgeLen(n + 1, 0.0);
    for (int i = 1; i <= n; ++i) {
        double sum = 0;
        int k = std::min(neighborLists.k, (int)neighborLists.nbr[i].size());
        for (int j = 0; j < k; ++j) {
            sum += dist(inst, i, neighborLists.nbr[i][j]);
        }
        if (k > 0) avgKnnEdgeLen[i] = sum / k;
    }
    
    std::vector<double> sortedLens = avgKnnEdgeLen;
    std::sort(sortedLens.begin() + 1, sortedLens.end());
    double median_len = sortedLens[1 + n / 2];
    double multiplier = 2.0;
    double R = median_len * multiplier;
    
    for (int i = 1; i <= n; ++i) {
        int c_i = res.chunkOf[i];
        std::set<int> adj_chunks;
        
        for (int j : neighborLists.nbr[i]) {
            if (dist(inst, i, j) > R) break; // Sorted ascending
            int c_j = res.chunkOf[j];
            if (c_i != c_j) {
                res.isBoundary[i] = true;
                adj_chunks.insert(c_j);
            }
        }
        
        if (res.isBoundary[i]) {
            res.boundaryList.push_back(i);
            res.boundaryChunkPair[i].assign(adj_chunks.begin(), adj_chunks.end());
        }
    }
    
    return res;
}
