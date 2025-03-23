cmake_minimum_required(VERSION 3.15)
project(llm_infer LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(PYBIND11_PYTHON_VERSION 3.12)

# 设置 pybind11 的 cmake 配置路径
set(pybind11_DIR "/root/miniconda3/lib/python3.12/site-packages/pybind11/share/cmake/pybind11")
find_package(pybind11 REQUIRED)

# 设置 Torch 的 cmake 配置路径
set(Torch_DIR "/home/libtorch/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)


include_directories(backend/cpp/include)
add_subdirectory(backend/cpp/src/CUDAoperators)
# 自动收集 operators 文件夹下的所有源文件
file(GLOB_RECURSE OPERATOR_SOURCES
     "backend/cpp/src/CUDAoperators/*.cu"
     "backend/cpp/src/CUDAoperators/*.cpp")

pybind11_add_module(model_bridge 
    interface/model_bridge.cpp
    ${OPERATOR_SOURCES}  # 自动添加算子实现文件
    backend/cpp/src/llama_decode.cpp
    backend/cpp/src/llama_prefill.cpp
    backend/cpp/src/inference.cpp
    backend/cpp/src/thread_pool.cpp
)

target_link_libraries(model_bridge PRIVATE ${TORCH_LIBRARIES} operators)

target_compile_options(model_bridge PRIVATE -mavx2 -mfma)

# 启用 CUDA 分离式编译
set_target_properties(model_bridge PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
