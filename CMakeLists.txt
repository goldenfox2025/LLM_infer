cmake_minimum_required(VERSION 3.15)
# 在project()之前设置编译器
set(CMAKE_CXX_COMPILER "/usr/bin/g++-12")
set(CMAKE_CUDA_HOST_COMPILER "/usr/bin/g++-12")


set(PYBIND11_FINDPYTHON ON)
project(llm_infer LANGUAGES CXX CUDA)

# --- 基本配置 ---
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 89)

# --- 优化Debug模式以兼容NCU分析 ---
# 修改默认的Debug模式配置，使其更适合NCU分析
set(CMAKE_CXX_FLAGS_DEBUG "-g1 -O2 -mavx2 -mfma" CACHE STRING "C++ flags for debug" FORCE)
set(CMAKE_CUDA_FLAGS_DEBUG "-lineinfo -O2 --generate-line-info --resource-usage" CACHE STRING "CUDA flags for debug" FORCE)
# --- 查找依赖 ---
find_package(CUDAToolkit REQUIRED)
set(PYBIND11_PYTHON_VERSION 3.10)
# find_package(Python ${PYBIND11_PYTHON_VERSION} COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)
# find_package(Torch REQUIRED)

# --- 添加子目录和源文件 ---
add_subdirectory(backend/cpp/src/CUDAoperators)
add_subdirectory(backend/cpp/src/operators)
# file(GLOB_RECURSE OPERATOR_SOURCES ...) # 如果需要

# --- 创建 Pybind11 模块 ---
pybind11_add_module(model_bridge SHARED
    interface/model_bridge.cpp
    interface/model_initializer.cpp
    # ${OPERATOR_SOURCES}
    backend/cpp/src/llama_decode.cpp
    backend/cpp/src/llama_prefill.cpp
    backend/cpp/src/inference.cpp
    backend/cpp/src/thread_pool.cpp
    backend/cpp/src/qwen.cpp
    backend/cpp/src/qwen3.cpp
    backend/cpp/src/qwen3_forward.cpp
    backend/cpp/src/qwen3_prefill.cpp
    backend/cpp/src/CudaMemoryPool.cpp
    backend/cpp/src/speculative_decoder.cpp)

# --- 设置链接库 ---
target_link_libraries(model_bridge PRIVATE
    # 使用PRIVATE关键字，防止链接标志被传递
    # 调整链接顺序：确保flash_attention所需的所有符号都能正确解析
    unified_operators  # 先链接统一的操作符接口
    operators         # 再链接具体的实现
    CUDA::cublas
    CUDA::cudart
    CUDA::cuda_driver
)


# --- 设置头文件包含路径 ---
target_include_directories(model_bridge PUBLIC
    backend/cpp/include
    interface
)
# target_include_directories(model_bridge PRIVATE ${CUDAToolkit_INCLUDE_DIRS})

# --- 设置编译选项 (区分 Debug 和 Release, 优化 CUDA 选项) ---
target_compile_options(model_bridge PRIVATE
    # --- C++ 编译器选项 ---
    $<$<CONFIG:Debug>:-g1;-O2;-mavx2;-mfma> # Debug: 减少调试信息，使用O2优化以减少请求数量
    $<$<CONFIG:Release>:-O3;-march=native;-funroll-loops;-flto;-mavx2;-mfma;-DNDEBUG> # Release: 完全优化

    # --- CUDA 编译器选项 (nvcc) ---
    # 使用 $<COMPILE_LANGUAGE:CUDA> 来确保这些选项只给 nvcc
    # Debug: 优化NCU兼容性，减少请求数量
    $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-lineinfo;-O2;--generate-line-info;--resource-usage;--display-error-number;--ptxas-options=-v>
    # Release: 完全优化
    $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-O3;--use_fast_math;-DNDEBUG>
    # 通用 CUDA 配置 (只给 CUDA)
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;-gencode;arch=compute_89,code=sm_89;--threads;0;--maxrregcount=64>
)

# --- 设置 CUDA 特定属性 ---
set_target_properties(model_bridge PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# --- (可选) 安装规则 ---
# install(TARGETS model_bridge DESTINATION your_install_path)