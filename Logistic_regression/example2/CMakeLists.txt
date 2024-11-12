cmake_minimum_required(VERSION 3.20)
project(demo02)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O3 -march=native")
set(CMAKE_PREFIX_PATH "/usr/local/lib/libtorch")

find_package(Torch REQUIRED)
include_directories("/usr/local/lib/libtorch/include")

add_executable(demo02 demo02.cc)

target_link_libraries(demo02 "${TORCH_LIBRARIES}")
