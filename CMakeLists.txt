cmake_minimum_required(VERSION 3.15)
project(llm_infer LANGUAGES CXX CUDA)

# --- 基本配置 ---
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 89)
set(CMAKE_CXX_COMPILER "/usr/bin/g++-12")
# --- 查找依赖 ---
find_package(CUDAToolkit REQUIRED)
set(PYBIND11_PYTHON_VERSION 3.10)
# find_package(Python ${PYBIND11_PYTHON_VERSION} COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)
# find_package(Torch REQUIRED)

# --- 添加子目录和源文件 ---
add_subdirectory(backend/cpp/src/CUDAoperators)
# file(GLOB_RECURSE OPERATOR_SOURCES ...) # 如果需要

# --- 创建 Pybind11 模块 ---
pybind11_add_module(model_bridge SHARED
    interface/model_bridge.cpp
    # ${OPERATOR_SOURCES}
    backend/cpp/src/llama_decode.cpp
    backend/cpp/src/llama_prefill.cpp
    backend/cpp/src/inference.cpp
    backend/cpp/src/thread_pool.cpp
    backend/cpp/src/qwen.cpp
)

# --- 设置链接库 ---
target_link_libraries(model_bridge PRIVATE operators)
target_link_libraries(model_bridge PRIVATE CUDA::cublas CUDA::cudart)
# target_link_libraries(model_bridge PRIVATE ${TORCH_LIBRARIES})

# --- 设置头文件包含路径 ---
target_include_directories(model_bridge PUBLIC backend/cpp/include)
# target_include_directories(model_bridge PRIVATE ${CUDAToolkit_INCLUDE_DIRS})

# --- 设置编译选项 (区分 Debug 和 Release, 修正 CUDA 选项传递) ---
target_compile_options(model_bridge PRIVATE
    # --- C++ 编译器选项 ---
# 在 target_compile_options 的 C++ 部分
    $<$<CONFIG:Debug>:-g;-O0;-mavx2;-mfma> # <-- 添加 -mavx2 和 -mfma
    $<$<CONFIG:Release>:-O3;-march=native;-funroll-loops;-flto;-mavx2;-mfma;-DNDEBUG>
    # --- CUDA 编译器选项 (nvcc) ---
    # 使用 $<COMPILE_LANGUAGE:CUDA> 来确保这些选项只给 nvcc
    $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G;-O0>   # Debug: -G 和 -O0 只给 CUDA
    $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-O3;--use_fast_math;-DNDEBUG> # Release: 优化选项只给 CUDA
    # 通用 CUDA 配置 (只给 CUDA)
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;-gencode;arch=compute_89,code=sm_89>
)

# --- 设置 CUDA 特定属性 ---
set_target_properties(model_bridge PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# --- (可选) 安装规则 ---
# install(TARGETS model_bridge DESTINATION your_install_path)