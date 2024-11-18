cmake_minimum_required(VERSION 3.20)
project(purchase_product)

set(CMAKE_PREFIX_PATH "/usr/local/lib/libtorch")

find_package(Torch REQUIRED)

include_directories(
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include
    ${TORCH_INCLUDE_DIRS}
)
message("")
message("TORCH_VERSION: " ${TORCH_VERSION}) # 打印不出来
message("Torch_VERSION: " ${Torch_VERSION})
message(STATUS "TORCH_INCLUDE_DIRS: " ${TORCH_INCLUDE_DIRS})
message(STATUS "Torch_INCLUDE_DIRS: " ${Torch_INCLUDE_DIRS}) # 小写打印不出来
message(STATUS "TORCH_LIBRARIES: " ${TORCH_LIBRARIES})
message(STATUS "Torch_LIBRARIES: " ${Torch_LIBRARIES}) # 小写打印不出来
message("")

add_executable(purchase purchase.cpp src/Purchase_product.cpp)

target_link_libraries(purchase ${TORCH_LIBRARIES})

set_property(TARGET purchase PROPERTY CXX_STANDARD 17)
