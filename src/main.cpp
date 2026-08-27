#include "Types.hpp"
#include "KDTree.hpp"
#include "Stage0_Partitioning.hpp"
#include "Stage3_MergeHealing.hpp"
#include "Stage4_5_CleanupPolish.hpp"
#include "Worker.hpp"
#include "Barrier.hpp"
#include "VrpParser.hpp"
#include <iostream>
#include <fstream>
#include <thread>
#include <vector>
#include <functional>
#include <chrono>
#include <string>

#if defined(_MSC_VER)
#include <crtdbg.h>
#endif

std::atomic<long long> global_dist_calls(0);
thread_local long long thread_dist_calls = 0;
int g_iters_per_node = 50; // overridable via --iters-per-node, see Stage2_ILS.cpp
int g_max_iterations_override = -1; // overridable via --max-iterations; -1 = use inst.n*g_iters_per_node
// Per-stage wall-clock time budgets (ms), overridable via --stage2/3/5-ms. -1 = use the
// legacy iteration-count path above instead. See docs/reports/004_time_budget_scheduling.md.
int g_stage2_time_budget_ms = -1;
int g_stage3_time_budget_ms = -1;
int g_stage5_time_budget_ms = -1;
// RNG seed base, overridable via --seed. Default 1337 matches the previously-hardcoded
// per-worker base (1337+i) and the Stage 3/5 offsets are chosen so the default reproduces
// every prior report's numbers byte-for-byte (see uses in Stage3_MergeHealing.cpp and
// Stage2_ILS.cpp's stage5_serial_polish). This is the only entropy source in the whole
// pipeline -- the solver is otherwise fully deterministic, so repeat runs at a fixed seed
// carry zero variance information (see docs/reports/005_cost_optimization.md, Phase 0).
int g_seed = 1337;
// T3 ROUTEMIN iterations per chunk, overridable via --routemin-iters. Default 0 (disabled):
// measured net-negative on VDA (mean cost +0.16% worse, route count UP not down toward kmin)
// -- our 9-operator local_search can't compensate for whole-route destruction the way FILO2's
// much richer 22-operator move set does, so the cost-primary accept rule tends to keep
// cost-cheaper-but-more-numerous route configurations. Implementation is verified correct
// (deterministic, feasible, matches FILO2's algorithm) and left available via this flag --
// likely worth revisiting once T2 (SMD rewrite, which is also what would bring the fuller
// move-generator set) exists. See docs/reports/009_plan_beating_filo2.md T3.
// Stays 0 (off) deliberately -- see docs/reports/010 section 0.3. With both port defects
// fixed, ROUTEMIN is a clear 0.48% win at Valle-D-Aosta scale (n~20k) using
// --routemin-iters 2000 --routemin-k 1000, but it needs a WIDE neighbour list to work at
// all, and that width does not scale: at Lazio (n~1M) k=300 costs +50% wall clock
// (315s -> 473s) and 7.11GB peak (94% of this machine's WSL ceiling) for +0.048% cost,
// while any k cheap enough to build there (k<=100) makes route count WORSE. So enabling it
// globally would trade a small-instance win for a large-instance regression.
// Recommended: n <= ~50k use --routemin-iters 2000 --routemin-k 1000; n >= ~500k leave off.
int g_routemin_iterations = 0;
// Candidate-list width used by ROUTEMIN specifically, overridable via --routemin-k.
// FILO2 runs ROUTEMIN at gamma=1.0 over a neighbour list of up to 1500 ("We are going to
// use all the available move generators for during this procedure", opt/routemin.hpp),
// because that phase's job is finding residual capacity ANYWHERE in the solution to absorb
// customers from the routes it destroys. Our original port reused the k=30 granular list,
// a ~50x narrower candidate set -- with only 30 candidates most of their routes are full,
// reinsertion fails, and we open a new route instead, which is the suspected reason route
// count rose instead of falling. The list is only built when ROUTEMIN is actually enabled
// (it costs n*k*4 bytes, which is ~1.2GB at Lazio scale for k=300).
int g_routemin_k = 100;
// T6: use Clarke & Wright savings construction instead of MST+randomized-DFS.
// --construction cw|mst. See clarke_wright_routes in Stage1_Construction.cpp for the
// measured motivation (our MST construction starts ~1% worse than CW on both VDA and
// Lazio, and at Lazio FILO2's CW output alone beats our final answer).
int g_use_clarke_wright = 0;
// Savings candidates per customer. FILO2's DEFAULT_CW_NEIGHBORS is 100; ours is capped by
// whatever neighbour list Stage 1 is handed (currently k=30), so this mainly guards memory:
// the savings array is O(chunkSize * cw_neighbors) per chunk, built concurrently by P
// workers.
int g_cw_neighbors = 100;

int main(int argc, char** argv) {
#if defined(_MSC_VER) && defined(_DEBUG)
    _CrtSetReportMode(_CRT_ASSERT, _CRTDBG_MODE_FILE | _CRTDBG_MODE_DEBUG);
    _CrtSetReportFile(_CRT_ASSERT, _CRTDBG_FILE_STDERR);
    _CrtSetReportMode(_CRT_ERROR, _CRTDBG_MODE_FILE | _CRTDBG_MODE_DEBUG);
    _CrtSetReportFile(_CRT_ERROR, _CRTDBG_FILE_STDERR);
#endif
    int P = 4; // number of chunks
    std::string inputFile;
    // Stage 6-C (docs/reports/009_plan_beating_filo2.md): multi-start portfolio runs several
    // instances of this binary concurrently as separate OS processes (see
    // tools/multistart.sh) -- --out/--log let each instance write to its own file instead of
    // all of them racing to overwrite the same results/final_solution.txt and
    // results/run_log.txt. Defaults preserve single-instance behavior exactly.
    std::string outPath = "results/final_solution.txt";
    std::string logPath = "results/run_log.txt";
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-p" && i + 1 < argc) {
            P = std::stoi(argv[++i]);
        } else if (arg == "-f" && i + 1 < argc) {
            inputFile = argv[++i];
        } else if (arg == "--iters-per-node" && i + 1 < argc) {
            g_iters_per_node = std::stoi(argv[++i]);
        } else if (arg == "--max-iterations" && i + 1 < argc) {
            g_max_iterations_override = std::stoi(argv[++i]);
        } else if (arg == "--stage2-ms" && i + 1 < argc) {
            g_stage2_time_budget_ms = std::stoi(argv[++i]);
        } else if (arg == "--stage3-ms" && i + 1 < argc) {
            g_stage3_time_budget_ms = std::stoi(argv[++i]);
        } else if (arg == "--stage5-ms" && i + 1 < argc) {
            g_stage5_time_budget_ms = std::stoi(argv[++i]);
        } else if (arg == "--seed" && i + 1 < argc) {
            g_seed = std::stoi(argv[++i]);
        } else if (arg == "--routemin-iters" && i + 1 < argc) {
            g_routemin_iterations = std::stoi(argv[++i]);
        } else if (arg == "--routemin-k" && i + 1 < argc) {
            g_routemin_k = std::stoi(argv[++i]);
        } else if (arg == "--construction" && i + 1 < argc) {
            g_use_clarke_wright = (std::string(argv[++i]) == "cw") ? 1 : 0;
        } else if (arg == "--cw-neighbors" && i + 1 < argc) {
            g_cw_neighbors = std::stoi(argv[++i]);
        } else if (arg == "--out" && i + 1 < argc) {
            outPath = argv[++i];
        } else if (arg == "--log" && i + 1 < argc) {
            logPath = argv[++i];
        } else if (arg[0] != '-') {
            inputFile = arg;
        }
    }

    std::ofstream logFile(logPath);
    if (!logFile) {
        std::cerr << "Failed to open " << logPath << std::endl;
        return 1;
    }

    std::cout << "Starting CVRP Parallel..." << std::endl;
    logFile << "Starting CVRP Parallel..." << std::endl;
    Instance inst;

    if (!inputFile.empty()) {
        if (!load_vrp_file(inputFile, inst)) {
            std::cerr << "Failed to load instance from " << inputFile << std::endl;
            return 1;
        }
    } else {
        std::cerr << "No input file provided! Usage: cvrp_solver.exe <file.vrp>" << std::endl;
        return 1;
    }

    auto t_start = std::chrono::high_resolution_clock::now();

    std::cout << "Building neighbor lists..." << std::endl;
    NeighborLists neighborLists;
    neighborLists.build(inst, 30, P); // limit k to 30 for SWAP* pruning constraint; P threads (same as Stage 1/2)

    NeighborLists stage5_neighborLists;
    stage5_neighborLists.build(inst, 100, P);

    // One wide candidate list, shared by ROUTEMIN and Clarke & Wright -- both need width for
    // the same underlying reason (enough distinct candidate routes to merge into / reinsert
    // into), and building it twice would double an already-expensive allocation. Only built
    // when at least one of them is enabled; see g_routemin_k's comment above for the cost.
    NeighborLists routemin_neighborLists;
    if (g_routemin_iterations > 0 || g_use_clarke_wright) {
        // Built with ALL available cores, not P. NeighborLists::build partitions node ranges
        // across threads with no shared mutable state, and its own contract says output is
        // "identical, deterministic ... regardless of thread count" (KDTree.hpp), so this is
        // free parallelism. It matters a lot at scale: the k=300 list for Lazio's ~1M
        // customers took 194.6 s of a 315 s budget at P=4 -- 62% of the entire run spent
        // building a neighbour list. The narrower neighborLists/stage5_neighborLists above
        // are deliberately left at P so every previously published baseline timing stays
        // comparable; only this new list changes.
        unsigned hw = std::thread::hardware_concurrency();
        int build_threads = std::max(P, (int)(hw ? hw : (unsigned)P));
        std::cout << "Building wide neighbor list (k=" << g_routemin_k
                  << ", " << build_threads << " threads)..." << std::endl;
        routemin_neighborLists.build(inst, g_routemin_k, build_threads);
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
        contexts[i].stage5_neighborLists = &stage5_neighborLists;
        contexts[i].routemin_neighborLists = &routemin_neighborLists;
        contexts[i].partitionInfo = &partitionInfo;
        contexts[i].rng.seed(g_seed + i); // Isolated deterministic seed per thread
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

    for (int i = 0; i < W; ++i) {
        logFile << contexts[i].log.str();
    }
    logFile.flush();

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
    // NOT globalSolution.routeHead.size() here: stage4_route_cleanup (just above) compacts
    // routeHead down to only the live routes, but stage5_serial_polish's own recreate() can
    // grow numRoutes further as it runs (routeHead.resize(r+100) on demand, uncapped) -- a
    // snapshot taken before Stage 5 starts has zero headroom for that growth. This was a
    // real regression: it replaced the previous single-arg call's inst.n+100 default (which
    // happened to have plenty of slack) with a much SMALLER bound, causing out-of-bounds
    // arena access once Stage 5 created even a few new routes -- confirmed via Tier-1
    // capacity-violation and heap-corruption crashes (docs/reports/006_throughput_and_parallelism.md
    // Phase 4.1). Use the same generous, forward-looking bound Stage 3 commits to instead.
    globalArena.reserve_fixed_capacity(inst.n, 2 * inst.n + 10000);
    try {
        // std::cout << "Running Stage 5 Fleet Minimization..." << std::endl;
        // stage5_fleet_minimization(globalSolution, globalArena, inst, stage5_neighborLists, partitionInfo.medianKnnEdgeLen);
        std::cout << "Running Stage 5 Serial Polish..." << std::endl;
        stage5_serial_polish(globalSolution, globalArena, inst, stage5_neighborLists, partitionInfo.medianKnnEdgeLen);
    } catch (const std::exception& e) {
        std::cerr << "STAGE 5 CRASHED: " << e.what() << std::endl;
        std::exit(1);
    } catch (...) {
        std::cerr << "STAGE 5 CRASHED: Unknown exception" << std::endl;
        std::exit(1);
    }

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
    
    // numRoutes can include "dead" slots left behind by ILS moves that emptied a route
    // without compacting the routeHead array (only Stage 4's cleanup compacts; Stage 5
    // polish can re-empty a route afterwards). Count actual non-empty routes so the
    // "Num Routes:" header always matches the route lines actually written below.
    int liveRouteCount = 0;
    for (int r = 0; r < globalSolution.numRoutes; ++r) {
        if (globalSolution.routeHead[r] != 0) liveRouteCount++;
    }

    std::ofstream solFile(outPath);
    solFile << "Final Cost: " << globalSolution.totalCost << "\n";
    solFile << "Num Routes: " << liveRouteCount << "\n";
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
