add_subdirectory(mutation)
add_subdirectory(recombination)
add_subdirectory(selection)

add_library(operators INTERFACE)
target_link_libraries(operators INTERFACE mutation recombination selection)
target_compile_features(operators INTERFACE cxx_std_17)

