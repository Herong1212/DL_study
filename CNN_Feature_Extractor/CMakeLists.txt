cmake_minimum_required(VERSION 3.20)
project(CNN_FeatureExtractor)


# 配置 libtorch 路径（确保 Libtorch 已安装）
set(CMAKE_PREFIX_PATH "/usr/local/lib/libtorch")  # 修改为 libtorch 的实际安装路径

# 查找 libtorch
find_package(Torch REQUIRED)

# 查找 OpenCV
find_package(OpenCV REQUIRED)

# 包含头文件目录
include_directories(
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include
    ${TORCH_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS})

# 添加可执行文件和源码
add_executable(main main.cpp src/FeatureExtractor.cpp)

# 链接依赖库
target_link_libraries(main 
${TORCH_LIBRARIES}
${OpenCV_LIBS})

# 防止链接时出错
set_property(TARGET main PROPERTY  CXX_STANDARD 17)
