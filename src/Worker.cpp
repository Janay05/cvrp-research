#include "Worker.hpp"
#include "Stage1_Construction.hpp"
#include "Stage2_ILS.hpp"
#include "Stage3_MergeHealing.hpp"
#include "Stage4_5_CleanupPolish.hpp"
#include "ThreadArena.hpp"
#include <iostream>
#include <chrono>

void worker_main(WorkerContext& ctx) {
    try {
        ThreadArena arena;
        arena.reserve_fixed_capacity(ctx.instance->n);
        SVCCache cache;
        cache.init(ctx.instance->n);

        thread_dist_calls = 0;
        auto t1_start = std::chrono::high_resolution_clock::now();
        std::cout << "Worker " << ctx.workerId << " Stage 1 starting" << std::endl;
        // Stage 1
        for (int chunkId : ctx.assignedChunks) {
            std::cout << "Worker " << ctx.workerId << " constructing chunk " << chunkId << std::endl;
            int chunkSize = (int)ctx.partitionInfo->globalId[chunkId].size() - 1;
            extern int g_iters_per_node;
            // max_iterations is inst.n*g_iters_per_node (see Stage2_ILS.cpp), not
            // chunkSize*50 -- every thread gets the same absolute budget regardless
            // of its chunk size.
            ctx.log << "Worker " << ctx.workerId << " chunk " << chunkId
                     << " chunkSize=" << chunkSize << " max_iterations=" << (ctx.instance->n * g_iters_per_node) << "\n";
            Solution sol = stage1_construct(chunkId, *ctx.instance, *ctx.partitionInfo, *ctx.neighborLists, ctx.rng);
            ctx.chunkSolutions.push_back(std::move(sol));
        }
        std::cout << "Worker " << ctx.workerId << " Stage 1 finished" << std::endl;

        auto t1_end = std::chrono::high_resolution_clock::now();
        double ms_s1 = std::chrono::duration<double, std::milli>(t1_end - t1_start).count();

        ctx.stage_barrier->arrive_and_wait(); // SYNC POINT 1

        auto t2_start = std::chrono::high_resolution_clock::now();
        std::cout << "Worker " << ctx.workerId << " Stage 2 starting" << std::endl;
        // Stage 2
        for (size_t i = 0; i < ctx.assignedChunks.size(); ++i) {
            std::cout << "Worker " << ctx.workerId << " ILS chunk " << ctx.assignedChunks[i] << std::endl;
            cache.clear();
            ctx.chunkSolutions[i] = stage2_ils(ctx.chunkSolutions[i], arena, cache, *ctx.instance,
                                               *ctx.partitionInfo, *ctx.neighborLists, ctx.assignedChunks[i], ctx.rng);
        }
        std::cout << "Worker " << ctx.workerId << " Stage 2 finished" << std::endl;
        auto t2_end = std::chrono::high_resolution_clock::now();
        double ms_s2 = std::chrono::duration<double, std::milli>(t2_end - t2_start).count();

        std::cout << "Worker " << ctx.workerId << " Stage 1: " << ms_s1 << " ms | Stage 2: " << ms_s2 << " ms" << std::endl;
        ctx.log << "Worker " << ctx.workerId << " Stage 1: " << ms_s1 << " ms | Stage 2: " << ms_s2 << " ms"
                 << " | dist_calls=" << thread_dist_calls << "\n";
        global_dist_calls.fetch_add(thread_dist_calls, std::memory_order_relaxed);
        
        ctx.stage_barrier->arrive_and_wait(); // SYNC POINT 2
    } catch (const std::exception& e) {
        std::cerr << "Thread " << ctx.workerId << " caught exception: " << e.what() << std::endl;
        exit(1);
    } catch (...) {
        std::cerr << "Thread " << ctx.workerId << " caught unknown exception!" << std::endl;
        exit(1);
    }
}
