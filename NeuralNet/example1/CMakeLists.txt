cmake_minimum_required(VERSION 3.20)
project(example1)

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O3 -march=native")
set(CMAKE_PREFIX_PATH "/usr/local/lib/libtorch")

find_package(Torch REQUIRED)
include_directories(
    "/usr/local/lib/libtorch/include"
    ./include)

add_executable(example1 example1.cpp src/NeuralNetwork.cpp)

target_link_libraries(example1 ${TORCH_LIBRARIES})

set_property(TARGET example1 PROPERTY CXX_STANDARD 17)
