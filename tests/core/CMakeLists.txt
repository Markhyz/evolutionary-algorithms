add_executable(test_individual testIndividual.cpp)
target_link_libraries(test_individual PRIVATE evo_alg gtest_main)
target_compile_features(test_individual PRIVATE cxx_std_17)
target_compile_options(test_individual PRIVATE ${CXX_FLAGS})

add_executable(test_population testPopulation.cpp)
target_link_libraries(test_population PRIVATE evo_alg gtest_main)
target_compile_features(test_population PRIVATE cxx_std_17)
target_compile_options(test_population PRIVATE ${CXX_FLAGS})