set(CMAKE_CXX_EXTENSIONS OFF)
set(CXX_FLAGS -Wall -Wextra -Wshadow -Wnon-virtual-dtor -pedantic -Wunused -Wconversion -Wsign-conversion)

include(FetchContent)

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        release-1.10.0
)
FetchContent_MakeAvailable(googletest)

add_subdirectory(core)