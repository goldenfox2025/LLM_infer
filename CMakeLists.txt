cmake_minimum_required(VERSION 3.15)
project(llm_infer)

set(CMAKE_CXX_STANDARD 17)
set(PYBIND11_PYTHON_VERSION 3.12)

# 设置 pybind11 的 cmake 配置路径
set(pybind11_DIR "/root/miniconda3/lib/python3.12/site-packages/pybind11/share/cmake/pybind11")
find_package(pybind11 REQUIRED)

# 设置 Torch 的 cmake 配置路径
set(Torch_DIR "/home/libtorch/libtorch/share/cmake/Torch")
# 或者：
# list(APPEND CMAKE_PREFIX_PATH "/home/libtorch/libtorch")
find_package(Torch REQUIRED)

include_directories(backend/cpp/include)

pybind11_add_module(model_bridge 
    interface/model_bridge.cpp
    backend/cpp/src/llama_decode.cpp
    backend/cpp/src/llama_prefill.cpp
    backend/cpp/src/inference.cpp
    backend/cpp/src/thread_pool.cpp)

target_link_libraries(model_bridge PRIVATE ${TORCH_LIBRARIES})
target_compile_options(model_bridge PRIVATE -mavx2 -mfma)
