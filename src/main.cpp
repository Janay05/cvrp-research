#include "Types.hpp"
#include "KDTree.hpp"
#include "Stage0_Partitioning.hpp"
#include "Stage3_MergeHealing.hpp"
#include "Stage4_5_CleanupPolish.hpp"
#include "Worker.hpp"
#include "Barrier.hpp"
#include <iostream>
#include <fstream>
#include <thread>
#include <vector>
#include <functional>
#include <chrono>
#include <string>

#include <crtdbg.h>

std::atomic<long long> global_dist_calls(0);
thread_local long long thread_dist_calls = 0;

int main(int argc, char** argv) {
#if defined(_MSC_VER) && defined(_DEBUG)
    _CrtSetReportMode(_CRT_ASSERT, _CRTDBG_MODE_FILE | _CRTDBG_MODE_DEBUG);
    _CrtSetReportFile(_CRT_ASSERT, _CRTDBG_FILE_STDERR);
    _CrtSetReportMode(_CRT_ERROR, _CRTDBG_MODE_FILE | _CRTDBG_MODE_DEBUG);
    _CrtSetReportFile(_CRT_ERROR, _CRTDBG_FILE_STDERR);
#endif
    std::ofstream logFile("results/run_log.txt");
    if (!logFile) {
        std::cerr << "Failed to open results/run_log.txt" << std::endl;
        return 1;
    }
    
    std::cout << "Starting CVRP Parallel..." << std::endl;
    logFile << "Starting CVRP Parallel..." << std::endl;
    Instance inst;
    inst.n = 2000;
    inst.Q = 100;
    inst.x.assign(inst.n + 1, 0);
    inst.y.assign(inst.n + 1, 0);
    inst.demand.assign(inst.n + 1, 1);
    inst.demand[0] = 0;
    
    // Assign random coordinates so kd-tree doesn't choke on zero-distance
    for(int i=1; i<=inst.n; ++i) {
        inst.x[i] = rand() % 1000;
        inst.y[i] = rand() % 1000;
    }
    
    // Dump instance to test_2000.vrp for FILO2 comparison
    std::ofstream vrpFile("test_2000.vrp");
    vrpFile << "NAME : test_2000\nTYPE : CVRP\nDIMENSION : " << (inst.n + 1) << "\nEDGE_WEIGHT_TYPE : EUC_2D\nCAPACITY : " << inst.Q << "\nNODE_COORD_SECTION\n";
    vrpFile << "1 0 0\n";
    for(int i=1; i<=inst.n; ++i) vrpFile << (i + 1) << " " << inst.x[i] << " " << inst.y[i] << "\n";
    vrpFile << "DEMAND_SECTION\n1 0\n";
    for(int i=1; i<=inst.n; ++i) vrpFile << (i + 1) << " " << inst.demand[i] << "\n";
    vrpFile << "DEPOT_SECTION\n1\n-1\n";
    vrpFile.close();

    auto t_start = std::chrono::high_resolution_clock::now();

    std::cout << "Building neighbor lists..." << std::endl;
    NeighborLists neighborLists;
    neighborLists.build(inst, 30); // limit k to 30 for SWAP* pruning constraint

    int P = 4; // number of chunks
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "-p" && i + 1 < argc) {
            P = std::stoi(argv[i+1]);
            i++;
        }
    }
    
    std::cout << "Running Stage 0 with P=" << P << " chunks..." << std::endl;
    Stage0Result partitionInfo = run_stage0(inst, neighborLists, P);
    
    auto t_stage0 = std::chrono::high_resolution_clock::now();

    std::cout << "Setting up workers..." << std::endl;
    int W = P; 
    Barrier worker_barrier(W);

    std::vector<WorkerContext> contexts(W);
    for (int i = 0; i < W; ++i) {
        contexts[i].workerId = i;
        contexts[i].assignedChunks.push_back(i); 
        contexts[i].stage_barrier = &worker_barrier;
        contexts[i].instance = &inst;
        contexts[i].neighborLists = &neighborLists;
        contexts[i].partitionInfo = &partitionInfo;
        contexts[i].rng.seed(1337 + i); // Isolated deterministic seed per thread
    }

    std::cout << "Launching threads..." << std::endl;
    std::vector<std::thread> threads;
    for (int i = 0; i < W; ++i) {
        threads.emplace_back(worker_main, std::ref(contexts[i]));
    }

    for (auto& t : threads) {
        t.join();
    }
    
    std::cout << "Threads joined. Merging..." << std::endl;

    Solution globalSolution;
    std::vector<Solution> allSolutions(partitionInfo.numChunks);
    for (int i = 0; i < W; ++i) {
        for (size_t j = 0; j < contexts[i].assignedChunks.size(); ++j) {
            int chunkId = contexts[i].assignedChunks[j];
            allSolutions[chunkId] = std::move(contexts[i].chunkSolutions[j]);
        }
    }
    
    merge_all_chunks_into_global(globalSolution, allSolutions, partitionInfo, inst);
    
    auto t_stage12 = std::chrono::high_resolution_clock::now();
    
    std::cout << "Cost BEFORE Stage 3 Healing: " << globalSolution.totalCost << std::endl;
    logFile << "Cost BEFORE Stage 3 Healing: " << globalSolution.totalCost << std::endl;

    std::cout << "Running Stage 3 Healing..." << std::endl;
    run_stage3_healing(globalSolution, inst, partitionInfo, neighborLists);
    
    Cost new_cost = 0;
    for (int r = 0; r < globalSolution.numRoutes; ++r) {
        NodeId curr = globalSolution.routeHead[r];
        if (curr == 0) continue;
        NodeId p = 0;
        while (curr != 0) {
            new_cost += dist(inst, p, curr);
            p = curr;
            curr = globalSolution.succ[curr];
        }
        new_cost += dist(inst, p, 0);
    }
    std::cout << "Incremental Cost Bookkeeping: " << globalSolution.totalCost << std::endl;
    std::cout << "Scratch Computed Final cost: " << new_cost << std::endl;
    
    if (globalSolution.totalCost != new_cost) {
        std::cout << "WARNING: Cost Bookkeeping mismatch! Incremental=" << globalSolution.totalCost << " vs Scratch=" << new_cost << std::endl;
        globalSolution.totalCost = new_cost;
    }

    std::cout << "Cost AFTER Stage 3 Healing: " << globalSolution.totalCost << std::endl;
    logFile << "Cost AFTER Stage 3 Healing: " << globalSolution.totalCost << std::endl;
    
    auto t_stage3 = std::chrono::high_resolution_clock::now();
    
    std::cout << "Running Stage 4 Cleanup..." << std::endl;
    logFile << "Running Stage 4 Cleanup..." << std::endl;
    stage4_route_cleanup(globalSolution, inst, neighborLists);
    
    // Recalculate true global cost after all ILS deltas and modifications
    Cost true_cost = 0;
    for (int r = 0; r < globalSolution.numRoutes; ++r) {
        NodeId curr = globalSolution.routeHead[r];
        if (curr == 0) continue;
        NodeId p = 0;
        while (curr != 0) {
            true_cost += dist(inst, p, curr);
            p = curr;
            curr = globalSolution.succ[curr];
        }
        true_cost += dist(inst, p, 0);
    }
    globalSolution.totalCost = true_cost;
    
    std::cout << "Running Stage 4 Cleanup..." << std::endl;
    // Stage 4 Cleanup
    
    std::cout << "Running Stage 5 Polish..." << std::endl;
    logFile << "Running Stage 5 Polish..." << std::endl;
    ThreadArena globalArena;
    globalArena.reserve_fixed_capacity(inst.n);
    stage5_serial_polish(globalSolution, globalArena, inst, neighborLists);
    
    auto t_end = std::chrono::high_resolution_clock::now();
    double ms_setup = std::chrono::duration<double, std::milli>(t_stage0 - t_start).count();
    double ms_stage12 = std::chrono::duration<double, std::milli>(t_stage12 - t_stage0).count();
    double ms_stage3 = std::chrono::duration<double, std::milli>(t_stage3 - t_stage12).count();
    double ms_stage45 = std::chrono::duration<double, std::milli>(t_end - t_stage3).count();
    double ms_total = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    
    std::cout << "Final cost: " << globalSolution.totalCost << std::endl;
    logFile << "Final cost: " << globalSolution.totalCost << std::endl;
    
    std::cout << "--- Stage by Stage Profiling ---" << std::endl;
    std::cout << "Setup (Stage 0): " << ms_setup << " ms" << std::endl;
    std::cout << "Stage 1 & 2 (Parallel Construction/ILS): " << ms_stage12 << " ms" << std::endl;
    std::cout << "Stage 3 (Parallel Healing): " << ms_stage3 << " ms" << std::endl;
    std::cout << "Stage 4 & 5 (Cleanup/Polish): " << ms_stage45 << " ms" << std::endl;
    std::cout << "Total time: " << ms_total << " ms" << std::endl;
    
    std::cout << "Total distance evaluations: " << global_dist_calls.load() << std::endl;
    
    std::ofstream solFile("results/final_solution.txt");
    solFile << "Final Cost: " << globalSolution.totalCost << "\n";
    solFile << "Num Routes: " << globalSolution.numRoutes << "\n";
    for (int r = 0; r < globalSolution.numRoutes; ++r) {
        if (globalSolution.routeHead[r] == 0) continue;
        solFile << "Route " << r << " (Load: " << globalSolution.routeLoad[r] << "): 0 -> ";
        NodeId curr = globalSolution.routeHead[r];
        while (curr != 0) {
            solFile << curr << " -> ";
            curr = globalSolution.succ[curr];
        }
        solFile << "0\n";
    }
    
    return 0;
}
