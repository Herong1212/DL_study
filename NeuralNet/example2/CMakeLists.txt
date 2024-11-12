cmake_minimum_required(VERSION 3.20)
project(example2)

set(CMAKE_PREFIX_PATH "/usr/local/lib/libtorch")

find_package(Torch REQUIRED)
include_directories("/usr/local/lib/libtorch/include")

add_executable(example2 example2.cpp )
target_link_libraries(example1 ${Torch_LIBRARIES})
