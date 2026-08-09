cmake_minimum_required(VERSION 3.10)
project(CVRP_Parallel)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

file(GLOB SOURCES "*.cpp")

# Release Build
add_executable(cvrp_parallel ${SOURCES})
if(MSVC)
    # -O3 -march=native above is GCC/Clang syntax that MSVC silently ignores (CMake's own
    # Release config still supplies /O2, so this was "no AVX2, no whole-program LTCG", not
    # "unoptimized"). /fp:fast is deliberately NOT used: dist() (Types.hpp) is
    # llround(sqrt(...)) and src/verifier.py independently recomputes distances with
    # Python's math.sqrt -- a lower-precision reciprocal-sqrt could shift a value across a
    # rounding boundary and produce genuine verifier mismatches. /arch:AVX2 does not change
    # scalar sqrt semantics and is safe. See docs/reports/006_throughput_and_parallelism.md.
    target_compile_options(cvrp_parallel PRIVATE /O2 /Ob3 /GL /arch:AVX2)
    target_link_options(cvrp_parallel PRIVATE /LTCG)
else()
    target_compile_options(cvrp_parallel PRIVATE -O3 -march=native)
endif()
if(NOT WIN32)
    target_link_libraries(cvrp_parallel pthread)
endif()

# Debug Build (ThreadSanitizer)
add_executable(cvrp_parallel_debug ${SOURCES})
target_compile_options(cvrp_parallel_debug PRIVATE -g -fsanitize=thread)
target_link_options(cvrp_parallel_debug PRIVATE -fsanitize=thread)
if(NOT WIN32)
    target_link_libraries(cvrp_parallel_debug pthread)
endif()
