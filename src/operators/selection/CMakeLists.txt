add_library(tournament STATIC tournament.cpp)
target_compile_features(tournament PUBLIC cxx_std_17)
target_compile_options(tournament PRIVATE ${CXX_FLAGS})

add_library(roulette STATIC roulette.cpp)
target_compile_features(roulette PUBLIC cxx_std_17)
target_compile_options(roulette PRIVATE ${CXX_FLAGS})

add_library(selection INTERFACE)
target_link_libraries(selection INTERFACE tournament roulette)
target_compile_features(selection INTERFACE cxx_std_17)
