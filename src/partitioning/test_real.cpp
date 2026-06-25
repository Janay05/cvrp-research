#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include "MacroPartitioner.hpp"
#include "MicroClusterer.hpp"

using namespace cvrp::partitioning;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_vrp>" << std::endl;
        return 1;
    }
    std::string vrp_path = argv[1];
    std::ifstream in(vrp_path);
    if (!in) {
        std::cerr << "Cannot open " << vrp_path << std::endl;
        return 1;
    }
    
    std::vector<Node> global_nodes;
    double vehicle_capacity = 0;
    std::string line;
    
    enum Section { HEADER, COORD, DEMAND, DEPOT };
    Section current_section = HEADER;
    
    std::cout << "Parsing VRP instance..." << std::endl;
    
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        if (line.find("CAPACITY") != std::string::npos) {
            auto colon = line.find(':');
            if (colon != std::string::npos) {
                vehicle_capacity = std::stod(line.substr(colon + 1));
            }
        } else if (line.find("NODE_COORD_SECTION") != std::string::npos) {
            current_section = COORD;
        } else if (line.find("DEMAND_SECTION") != std::string::npos) {
            current_section = DEMAND;
        } else if (line.find("DEPOT_SECTION") != std::string::npos) {
            current_section = DEPOT;
        } else if (line.find("EOF") != std::string::npos) {
            break;
        } else {
            std::stringstream ss(line);
            if (current_section == COORD) {
                Node n;
                ss >> n.original_id >> n.x >> n.y;
                n.demand = 0;
                global_nodes.push_back(n);
            } else if (current_section == DEMAND) {
                int id;
                double demand;
                ss >> id >> demand;
                if (id >= 1 && id <= global_nodes.size()) {
                    global_nodes[id - 1].demand = demand;
                }
            } else if (current_section == DEPOT) {
                int id;
                ss >> id;
                if (id == -1) break;
            }
        }
    }
    
    if (global_nodes.empty()) {
        std::cerr << "Error: No nodes parsed." << std::endl;
        return 1;
    }

    Node depot = global_nodes[0];
    global_nodes.erase(global_nodes.begin()); // Remove depot from partitioning pool
    
    std::cout << "Loaded " << global_nodes.size() << " customer nodes." << std::endl;
    std::cout << "Vehicle Capacity: " << vehicle_capacity << std::endl;
    
    std::cout << "Running Stage 1 (Macro Partitioning)..." << std::endl;
    MacroPartitioner macro_partitioner(45.0); // 8 wedges
    auto wedges = macro_partitioner.partition(depot, global_nodes);
    std::cout << "Generated " << wedges.size() << " macro-wedges." << std::endl;
    
    // For 1M scale, we want HGS to run on subproblems of roughly ~1000 to ~2000 nodes.
    int target_subproblem_size = 1500; 
    
    std::cout << "Running Stage 2 (Micro Clustering)..." << std::endl;
    std::ofstream out("clusters_output.csv");
    out << "original_id,x,y,macro_id,micro_id\n";
    
    // Output the depot
    out << depot.original_id << "," << depot.x << "," << depot.y << ",-1,-1\n";
    
    int global_micro_id = 0;
    for (size_t i = 0; i < wedges.size(); ++i) {
        int wedge_size = wedges[i].size();
        
        // Dynamic banding based on density:
        // We want petals to be roughly square. If we aim for 1500 nodes per cluster,
        // and we want ~4-5 clusters per band to create an angular sweep, 
        // each band should contain roughly 6000 - 7500 nodes.
        int nodes_per_band = 6000;
        int bands = std::max(1, wedge_size / nodes_per_band);
        
        ConcentricSweepClusterer micro_clusterer(bands);
        auto micro_partitions = micro_clusterer.cluster(depot, wedges[i], vehicle_capacity, target_subproblem_size);
        
        for (auto& sp : micro_partitions) {
            for (auto& n : sp.nodes) {
                out << n.original_id << "," << n.x << "," << n.y << "," << i << "," << global_micro_id << "\n";
            }
            global_micro_id++;
        }
    }
    
    std::cout << "Generated total " << global_micro_id << " micro-clusters." << std::endl;
    std::cout << "Output saved to clusters_output.csv!" << std::endl;
    
    return 0;
}
