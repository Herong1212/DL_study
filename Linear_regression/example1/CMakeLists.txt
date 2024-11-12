cmake_minimum_required(VERSION 3.20)
project(singleX)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O3 -march=native")

# 这行代码告诉 CMake 在 /usr/local/lib/libtorch 路径下查找库和相关的 CMake 配置文件，这样 find_package(Torch REQUIRED) 可以成功找到 libtorch。
set(CMAKE_PREFIX_PATH "/usr/local/lib/libtorch")
find_package(Torch REQUIRED)

include_directories("/usr/local/lib/libtorch/include")

add_executable(singleX singleX.cpp)

target_link_libraries(singleX "${TORCH_LIBRARIES}") # 不能用 ${TORCH_LIBS}
