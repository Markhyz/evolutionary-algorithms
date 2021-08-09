set(CMAKE_CXX_EXTENSIONS OFF)
set(CXX_FLAGS -Wall -Wextra -Wshadow -Wnon-virtual-dtor -pedantic -Wunused -Wconversion -Wsign-conversion)

add_subdirectory(core)
add_subdirectory(commons)
add_subdirectory(operators)