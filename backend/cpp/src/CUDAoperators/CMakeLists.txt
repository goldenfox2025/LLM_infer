# backend/cpp/src/CUDAoperators/CMakeLists.txt

# 自动查找当前目录及其子目录下的所有 .cu 和 .cpp 文件
# 注意: GLOB_RECURSE 可能包含不期望的文件，如果目录结构复杂，
#       显式列出文件或使用更精确的 GLOB 可能更好。
file(GLOB_RECURSE OPERATOR_SOURCES "*.cu" "*.cpp")

# --- 创建静态库目标 ---
add_library(operators STATIC ${OPERATOR_SOURCES})

# --- *** 添加: 启用位置无关代码 (fPIC) *** ---
# 这对于将此静态库链接到共享库 (如 Python 模块)至关重要
set_target_properties(operators PROPERTIES POSITION_INDEPENDENT_CODE ON)
# --- ************************************** ---

# --- 设置头文件包含路径 ---
target_include_directories(operators PUBLIC
    # 将当前目录添加到包含路径 (如果头文件和源文件混合存放)
    ${CMAKE_CURRENT_SOURCE_DIR}
    # 添加项目顶层的公共包含目录
    ${CMAKE_SOURCE_DIR}/backend/cpp/include
)

# --- 添加 CUTLASS 包含路径 ---
# 设置 CUTLASS 根目录 (相对于主 CMakeLists.txt)
set(CUTLASS_DIR "${CMAKE_SOURCE_DIR}/cutlass")
# 将 CUTLASS 的 include 和 tools/util/include 添加到包含路径
target_include_directories(operators PUBLIC
    ${CUTLASS_DIR}/include
    ${CUTLASS_DIR}/tools/util/include
)

# --- 设置编译选项 (继承或独立设置) ---
# 选项 1: 尝试从父级继承 (如果父级 CMakeLists.txt 设置正确且可见)
# (通常不需要额外操作，但如果不行，用选项 2)

# 选项 2: (如果需要独立设置或覆盖) 在这里为 operators 库单独设置编译选项
# 注意：这会覆盖从父级继承的同名选项。
#      确保这里的选项与主 CMakeLists.txt 中 model_bridge 的设置一致或兼容。
# target_compile_options(operators PRIVATE
#     # --- C++ 编译器选项 ---
#     $<$<CONFIG:Debug>:-g;-O0> # Debug C++
#     # $<$<CONFIG:Debug>:-mavx2;-mfma> # 如果 operators 库的 C++ 代码也用 AVX，Debug 时加上
#     $<$<CONFIG:Release>:-O3;-march=native;-funroll-loops;-flto;-mavx2;-mfma;-DNDEBUG> # Release C++

#     # --- CUDA 编译器选项 (nvcc) ---
#     $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G;-O0>   # Debug CUDA (-G)
#     $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-O3;--use_fast_math;--expt-relaxed-constexpr;-DNDEBUG> # Release CUDA
#     # 通用 CUDA 配置 (包括 Cutlass 警告需要的标志)
#     $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;-gencode;arch=compute_89,code=sm_89>
#     # 如果父级已经设置了 --expt-relaxed-constexpr, 这里理论上可以不重复，但写上更保险
# )

# --- (可选) 设置 CUDA 特定属性 ---
# 如果 operators 库内部也需要跨 .cu 文件调用设备函数
# set_target_properties(operators PROPERTIES CUDA_SEPARABLE_COMPILATION ON)