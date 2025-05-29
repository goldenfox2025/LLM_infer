cmake_minimum_required(VERSION 3.18)

project(cute_study LANGUAGES CXX CUDA)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置CUDA标准
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 查找CUDA
find_package(CUDA REQUIRED)

# 设置CUDA架构
set(CMAKE_CUDA_ARCHITECTURES 80 86 89 90)

# 包含CUTLASS头文件
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../cutlass/include)

# 设置编译选项
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -lineinfo")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# 添加可执行文件
add_executable(cute_add cute_add.cu)

# 设置CUDA属性
set_target_properties(cute_add PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# 链接CUDA库
target_link_libraries(cute_add ${CUDA_LIBRARIES})
