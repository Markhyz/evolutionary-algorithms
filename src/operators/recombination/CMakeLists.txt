add_library(sbx STATIC sbx.cpp)
target_compile_features(sbx PUBLIC cxx_std_17)
target_compile_options(sbx PRIVATE ${CXX_FLAGS})

add_library(arithmetic STATIC arithmetic.cpp)
target_compile_features(arithmetic PUBLIC cxx_std_17)
target_compile_options(arithmetic PRIVATE ${CXX_FLAGS})

add_library(recombination INTERFACE)
target_link_libraries(recombination INTERFACE sbx arithmetic)
target_compile_features(recombination INTERFACE cxx_std_17)
