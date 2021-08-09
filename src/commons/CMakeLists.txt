
add_library(utils STATIC utils.cpp)
target_compile_features(utils PUBLIC cxx_std_17)
target_compile_options(utils PRIVATE ${CXX_FLAGS})

add_library(commons INTERFACE)
target_link_libraries(commons INTERFACE utils)
target_compile_features(commons INTERFACE cxx_std_17)

