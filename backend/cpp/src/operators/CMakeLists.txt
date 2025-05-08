# 设置最低CMake版本
cmake_minimum_required(VERSION 3.10)

# 设置CUDA架构 - 只保留需要的架构以减少编译时间
set(CMAKE_CUDA_ARCHITECTURES 89)

# 添加CUDA编译选项
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++17")

# 为Debug模式添加优化选项，减少NCU请求数量
set(CMAKE_CUDA_FLAGS_DEBUG "-lineinfo -O2 --generate-line-info --resource-usage --display-error-number --threads 0 --maxrregcount=64")
set(CMAKE_CXX_FLAGS_DEBUG "-g1 -O2 -mavx2 -mfma")

# 添加CPU算子源文件
file(GLOB CPU_OPERATOR_SOURCES "cpu/*.cpp")

# 添加CUDA算子源文件
file(GLOB CUDA_OPERATOR_SOURCES "cuda/*.cu")

# 添加MatMul算子源文件
file(GLOB MATMUL_OPERATOR_SOURCES
    "cuda/matmul/*.cu"
    "cpu/matmul*.cpp"
    "cuda/matmul_selector.cpp"
)

# 创建算子库 - 使用不同的名称避免冲突
add_library(unified_operators STATIC
    ${CPU_OPERATOR_SOURCES}
    ${CUDA_OPERATOR_SOURCES}
    ${MATMUL_OPERATOR_SOURCES}
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
    CUDA_SEPARABLE_COMPILATION ON
)

# 添加编译选项
target_compile_options(unified_operators PRIVATE
    # C++ 编译器选项
    $<$<CONFIG:Debug>:-g1;-O2;-mavx2;-mfma>
    $<$<CONFIG:Release>:-O3;-march=native;-funroll-loops;-mavx2;-mfma;-DNDEBUG>

    # CUDA 编译器选项
    $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-lineinfo;-O2;--generate-line-info;--resource-usage;--display-error-number>
    $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-O3;--use_fast_math;-DNDEBUG>

    # 通用 CUDA 配置
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;-gencode;arch=compute_89,code=sm_89;--threads;0;--maxrregcount=64>
)
