cmake_minimum_required(VERSION 3.17)

project(EvolutionaryAlgorithms VERSION 1.0.0
                               DESCRIPTION "Evolutionary Algorithms library"
                               LANGUAGES CXX)

add_subdirectory(src)

add_library(evo_alg INTERFACE)
target_include_directories(evo_alg INTERFACE include)
target_link_libraries(evo_alg INTERFACE operators commons core)
target_compile_features(evo_alg INTERFACE cxx_std_17)

if (OMP)
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(evo_alg INTERFACE OpenMP::OpenMP_CXX)
    endif()
endif()

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    add_subdirectory(tests)
    add_subdirectory(examples)
endif()