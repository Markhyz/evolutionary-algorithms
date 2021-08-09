add_library(fitness STATIC fitness.cpp)
target_compile_features(fitness PUBLIC cxx_std_17)
target_compile_options(fitness PRIVATE ${CXX_FLAGS})

add_library(core INTERFACE)
target_link_libraries(core INTERFACE fitness)
target_compile_features(core INTERFACE cxx_std_17)
