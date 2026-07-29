#include "DataStructures.hpp"
#include "MacroPartitioner.hpp"
#include "MicroClusterer.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <random>

using namespace cvrp::partitioning;

// Generate dummy CVRP data for visualization
std::vector<Node> generate_dummy_nodes(int count) {
    std::vector<Node> nodes;
    std::mt19937 gen(42); // fixed seed
    std::uniform_real_distribution<double> dist(-1000.0, 1000.0);
    std::uniform_real_distribution<double> demand_dist(10.0, 50.0);
    
    for (int i = 1; i <= count; ++i) {
        Node n;
        n.original_id = i;
        n.x = dist(gen);
        n.y = dist(gen);
        n.demand = demand_dist(gen);
        nodes.push_back(n);
    }
    return nodes;
}

int main() {
    int total_nodes = 50000;
    std::cout << "Generating " << total_nodes << " nodes for visual test..." << std::endl;
    
    Node depot;
    depot.original_id = 0;
    depot.x = 0.0;
    depot.y = 0.0;
    depot.demand = 0.0;

    auto global_nodes = generate_dummy_nodes(total_nodes);

    // Stage 1: Macro-Partitioning
    MacroPartitioner macro_partitioner(45.0); // 8 wedges
    std::cout << "Running Stage 1 (Macro Partitioning)..." << std::endl;
    auto wedges = macro_partitioner.partition(depot, global_nodes);
    
    std::cout << "Generated " << wedges.size() << " macro-wedges." << std::endl;

    // Stage 2: Micro-Clustering
    ConcentricSweepClusterer micro_clusterer(5); // 5 bands per wedge
    
    std::ofstream out("clusters_output.csv");
    out << "original_id,x,y,macro_id,micro_id\n";
    
    int target_subproblem_size = 250; // Aiming for ~250 nodes per final chunk
    double vehicle_capacity = 1000.0; // Arbitrary for this test, we rely on target size

    std::cout << "Running Stage 2 (Micro Clustering) and writing output..." << std::endl;
    for (size_t i = 0; i < wedges.size(); ++i) {
        auto micro_partitions = micro_clusterer.cluster(depot, wedges[i], vehicle_capacity, target_subproblem_size);
        
        for (const auto& subproblem : micro_partitions) {
            for (const auto& node : subproblem.nodes) {
                out << node.original_id << ","
                    << node.x << ","
                    << node.y << ","
                    << i << ","
                    << subproblem.id << "\n";
            }
        }
    }
    
    out.close();
    std::cout << "Output saved to clusters_output.csv!" << std::endl;
    std::cout << "You can now run a Python script to scatter plot this CSV, coloring by 'micro_id'." << std::endl;

    return 0;
}
