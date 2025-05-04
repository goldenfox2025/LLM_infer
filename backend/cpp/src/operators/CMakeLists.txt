# 设置最低CMake版本
cmake_minimum_required(VERSION 3.10)

# 设置CUDA架构
set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 89 90)

# 添加CUDA编译选项
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++17")

# 添加CPU算子源文件
file(GLOB CPU_OPERATOR_SOURCES "cpu/*.cpp")

# 添加CUDA算子源文件
file(GLOB CUDA_OPERATOR_SOURCES "cuda/*.cu")

# 创建算子库 - 使用不同的名称避免冲突
add_library(unified_operators STATIC
    ${CPU_OPERATOR_SOURCES}
    ${CUDA_OPERATOR_SOURCES}
)

# 设置包含目录
target_include_directories(unified_operators PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${CMAKE_SOURCE_DIR}
)

# 链接CUDA库
target_link_libraries(unified_operators PUBLIC
    CUDA::cudart
    CUDA::cublas
)

# 设置C++标准和位置无关代码
set_target_properties(unified_operators PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
)
