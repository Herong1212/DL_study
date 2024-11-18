cmake_minimum_required(VERSION 3.20)
project(simple)

set(CMAKE_PREFIX_PATH "/usr/local/lib/libtorch")

find_package(Torch REQUIRED)

include_directories(
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include
    ${TORCH_INCLUDE_DIRS}
)

message("")
message(STATUS "------------------------- Torch Version: " ${Torch_VERSION})
message(STATUS "------------------------- Torch include dirs: " ${TORCH_INCLUDE_DIRS})
message("")
message(STATUS "------------------------- PROJECT_SOURCE_DIR: " ${PROJECT_SOURCE_DIR})
message(STATUS "------------------------- Torch library dirs: " ${TORCH_LIBRARIES})
message("")

add_executable(simple simple.cpp src/NeuralNetwork.cpp)

target_link_libraries(simple ${TORCH_LIBRARIES})

set_property(TARGET simple PROPERTY CXX_STANDARD 17)
