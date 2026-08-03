cmake_minimum_required(VERSION 3.10)
project(CVRP_Parallel)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

file(GLOB SOURCES "*.cpp")

# Release Build
add_executable(cvrp_parallel ${SOURCES})
target_compile_options(cvrp_parallel PRIVATE -O3 -march=native)
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
