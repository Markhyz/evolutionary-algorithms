set(CMAKE_CXX_EXTENSIONS OFF)
set(CXX_FLAGS -Wall -Wextra -Wshadow -Wnon-virtual-dtor -pedantic -Wunused -Wconversion -Wsign-conversion -g)

add_subdirectory(ga)
add_subdirectory(brkga)