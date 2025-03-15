# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.28

# compile CUDA with /usr/bin/nvcc
# compile CXX with /usr/bin/c++
CUDA_DEFINES = -DUSE_C10D_GLOO -DUSE_C10D_NCCL -DUSE_DISTRIBUTED -DUSE_RPC -DUSE_TENSORPIPE -Dmodel_bridge_EXPORTS

CUDA_INCLUDES = --options-file CMakeFiles/model_bridge.dir/includes_CUDA.rsp

CUDA_FLAGS =  -DONNX_NAMESPACE=onnx_c2 -gencode arch=compute_89,code=sm_89 -Xcudafe --diag_suppress=cc_clobber_ignored,--diag_suppress=field_without_dll_interface,--diag_suppress=base_class_has_different_dll_interface,--diag_suppress=dll_interface_conflict_none_assumed,--diag_suppress=dll_interface_conflict_dllexport_assumed,--diag_suppress=bad_friend_decl --expt-relaxed-constexpr --expt-extended-lambda -std=c++17 -Xcompiler=-fPIC -Xcompiler=-fvisibility=hidden -mavx2 -mfma -D_GLIBCXX_USE_CXX11_ABI=0

CXX_DEFINES = -DUSE_C10D_GLOO -DUSE_C10D_NCCL -DUSE_DISTRIBUTED -DUSE_RPC -DUSE_TENSORPIPE -Dmodel_bridge_EXPORTS

CXX_INCLUDES = -I/home/LLM_infer/backend/cpp/include -isystem /root/miniconda3/lib/python3.12/site-packages/pybind11/include -isystem /root/miniconda3/include/python3.12 -isystem /home/libtorch/libtorch/include -isystem /home/libtorch/libtorch/include/torch/csrc/api/include

CXX_FLAGS = -std=gnu++17 -fPIC -fvisibility=hidden -mavx2 -mfma -flto -fno-fat-lto-objects -D_GLIBCXX_USE_CXX11_ABI=0

