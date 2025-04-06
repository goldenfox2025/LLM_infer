cmake_minimum_required(VERSION 3.15)
project(llm_infer LANGUAGES CXX CUDA)
set(CMAKE_CUDA_ARCHITECTURES 89)

set(CMAKE_CXX_STANDARD 17)
set(PYBIND11_PYTHON_VERSION 3.12)
set(CMAKE_CXX_FLAGS_DEBUG "-g") # 为 Debug 构建类型添加 -g
set(CMAKE_BUILD_TYPE Debug) # 默认或显式设置为 Debug 类型以启用 -g
# 设置 pybind11 的 cmake 配置路径
set(pybind11_DIR "/root/miniconda3/lib/python3.12/site-packages/pybind11/share/cmake/pybind11")
find_package(pybind11 REQUIRED)

# 设置 Torch 的 cmake 配置路径
set(Torch_DIR "/home/libtorch/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)


# include_directories(backend/cpp/include)



add_subdirectory(backend/cpp/src/CUDAoperators)
# 自动收集 operators 文件夹下的所有源文件
file(GLOB_RECURSE OPERATOR_SOURCES
     "backend/cpp/src/CUDAoperators/*.cu"
     "backend/cpp/src/CUDAoperators/*.cpp")

pybind11_add_module(model_bridge 
    interface/model_bridge.cpp
    ${OPERATOR_SOURCES} 
    backend/cpp/src/llama_decode.cpp
    backend/cpp/src/llama_prefill.cpp
    backend/cpp/src/inference.cpp
    backend/cpp/src/thread_pool.cpp
    backend/cpp/src/qwen.cpp

    
)







target_link_libraries(model_bridge PRIVATE ${TORCH_LIBRARIES} operators)


# 将 backend/cpp/include 添加到 model_bridge 的包含路径
# PUBLIC: 如果其他目标链接 model_bridge，它们也能看到这个目录（虽然此例中没有）
# INTERFACE: 只让链接 model_bridge 的目标看到
# PRIVATE: 只有 model_bridge 自身编译时能看到
# 对于头文件目录，通常用 PUBLIC 或 INTERFACE
target_include_directories(model_bridge PUBLIC backend/cpp/include)

# 如果 operators 库也需要这个目录
# (假设 operators 库在子目录 CMakeLists.txt 中定义)
# 在 backend/cpp/src/CUDAoperators/CMakeLists.txt 中添加:
# target_include_directories(operators PUBLIC ../../include) # 相对于子目录的路径

# 为 g++ (CXX) 添加 Release 优化参数
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native -funroll-loops -flto")

# 为 nvcc 添加 Release 优化参数
# 注意：CUDA_NVCC_FLAGS_RELEASE 变量可能需要根据你的 CUDA 版本确认
set(CUDA_NVCC_FLAGS_RELEASE "${CUDA_NVCC_FLAGS_RELEASE} -O3 --use_fast_math  -arch=sm_89")

# 修改 target 的编译选项，使用生成器表达式针对 CXX 与 CUDA 分别设置
target_compile_options(model_bridge PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -funroll-loops -flto -mavx2 -mfma>
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -arch=sm_89>
)

# 启用 CUDA 分离式编译
set_target_properties(model_bridge PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
