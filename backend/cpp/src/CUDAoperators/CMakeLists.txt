# backend/cpp/src/CUDAoperators/CMakeLists.txt
set(CMAKE_CXX_STANDARD 17)

# 设置CUDA编译器选项以处理GCC兼容性问题和抑制特定警告
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --allow-unsupported-compiler --expt-relaxed-constexpr --extended-lambda")

# 确保使用兼容的GCC版本
set(CMAKE_CUDA_HOST_COMPILER "/usr/bin/g++-12")

# 自动查找当前目录及其子目录下的所有 .cu 和 .cpp 文件
# 注意: GLOB_RECURSE 可能会意外包含不需要的文件。如果项目结构复杂，
#       建议明确列出源文件或使用更精确的 GLOB 模式。
file(GLOB_RECURSE OPERATOR_SOURCES "*.cu" "*.cpp")

# --- 创建静态库目标 'operators' ---
# 使用上面找到的源文件列表创建名为 "operators" 的静态库。
add_library(operators STATIC ${OPERATOR_SOURCES})

# --- 启用位置无关代码 (Position Independent Code - fPIC) ---
# 这对于将静态库链接到共享库 (例如 Python 的 C++ 扩展模块) 是必需的。
# 如果不设置，链接最终的共享库时可能会出现链接错误。
set_target_properties(operators PROPERTIES POSITION_INDEPENDENT_CODE ON)
# set_target_properties(operators PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# --- 设置头文件包含路径 ---
# target_include_directories 用于告诉编译器在哪里查找 #include 的文件。
# PUBLIC: 表示不仅 operators 库本身编译时需要这些路径，
#         链接到 operators 库的其他目标 (例如 model_bridge) 也会自动继承这些包含路径。
target_include_directories(operators PUBLIC
    # 将当前目录 (CUDAoperators 目录) 添加到包含路径。
    # 这适用于头文件 (.h/.hpp) 与源文件 (.cu/.cpp) 放在同一目录的情况。
    ${CMAKE_CURRENT_SOURCE_DIR}

    # 添加项目顶层的公共包含目录 (backend/cpp/include)。
    # ${CMAKE_SOURCE_DIR} 指向项目根目录 (包含主 CMakeLists.txt 的目录)。
    ${CMAKE_SOURCE_DIR}/backend/cpp/include
)

# --- 添加 CUTLASS 包含路径 ---
# 设置 CUTLASS 库的根目录。
# 这里假设 cutlass 目录位于项目根目录下。
set(CUTLASS_DIR "${CMAKE_SOURCE_DIR}/cutlass")

# 将 CUTLASS 的主要 include 目录和工具类 include 目录添加到包含路径。
# 这些是编译使用 CUTLASS 功能的代码所必需的。
target_include_directories(operators PUBLIC
    ${CUTLASS_DIR}/include
    ${CUTLASS_DIR}/tools/util/include
)

# --- 设置编译选项 ---
# target_compile_options 用于向编译器传递特定的标志。
# PRIVATE: 表示这些选项仅用于编译 operators 库本身，不会传递给链接它的目标。
target_compile_options(operators PRIVATE
    # --- C++ 编译器选项 (例如 g++, clang++) ---
    # $<CONFIG:Debug>: 仅在 Debug 模式下应用的选项。
    # -g1: 添加最小调试信息。
    # -O2: 适度优化，减少请求数量。
    # -mavx2, -mfma: 启用 AVX2 和 FMA 指令集 (如果 CPU 支持，可以提升性能)。
    $<$<CONFIG:Debug>:-g1;-O2;-mavx2;-mfma;-DDEBUG_AWQ>

    # $<CONFIG:Release>: 仅在 Release 模式下应用的选项。
    # -O3: 开启高级别优化。
    # -march=native: 针对当前编译机器的 CPU 架构进行优化。
    # -funroll-loops: 展开循环 (可能提升性能)。
    # -flto: 启用链接时优化 (Link Time Optimization)。
    # -mavx2, -mfma: 同样启用 AVX2/FMA。
    # -DNDEBUG: 定义 NDEBUG 宏，通常用于禁用 assert() 等调试代码。
    $<$<CONFIG:Release>:-O3;-march=native;-funroll-loops;-mavx2;-mfma;-DNDEBUG;-lineinfo>

    # --- CUDA 编译器选项 (nvcc) ---
    # 使用 $<COMPILE_LANGUAGE:CUDA> 来确保这些选项只传递给 nvcc。
    # $<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>: 仅在 Debug 模式下编译 CUDA 代码时应用。
    # -lineinfo: 添加行信息而不是完整调试信息，减少请求数量。
    # -O2: 适度优化，减少请求数量。
    # -DDEBUG_AWQ: 启用AWQ调试输出
    $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-lineinfo;-O2;--generate-line-info;--resource-usage;--display-error-number;-DDEBUG_AWQ>

    # $<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>: 仅在 Release 模式下编译 CUDA 代码时应用。
    # -O3: 开启优化。
    # --use_fast_math: 使用快速数学库 (可能牺牲一些精度以换取性能)。
    # -DNDEBUG: 定义 NDEBUG 宏。
    $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-O3;--use_fast_math;-DNDEBUG>

    # --- 通用的 CUDA 配置 (对 Debug 和 Release 的 CUDA 编译都生效) ---
    # $<COMPILE_LANGUAGE:CUDA>: 应用于所有 CUDA 编译。
    # --expt-relaxed-constexpr:  【关键】允许在 __host__ __device__ 函数中调用 __host__ constexpr 函数，
    #                           解决你遇到的 Cutlass 警告 #20015-D。
    # -gencode arch=compute_XX,code=sm_XX: 指定目标 GPU 架构。
    #                                    这里的 89 对应于 Hopper 架构 (H100)。
    #                                    确保这个值与你的目标硬件以及主 CMakeLists.txt 中的设置一致。
    # --threads 0: 使用所有可用的CPU线程进行编译，加快编译速度。
    # --maxrregcount=64: 限制每个线程使用的寄存器数量，可能有助于减少请求数量。
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;-gencode;arch=compute_89,code=sm_89;--threads;0;--maxrregcount=64>
)

# --- (可选) 设置 CUDA 特定属性 ---
# 如果 operators 库内部包含多个 .cu 文件，并且这些文件之间需要相互调用 __device__ 函数
# (或者一个 .cu 文件调用另一个 .cu 文件中定义的 __global__ 内核)，
# 则需要启用可分离编译 (Separable Compilation)。
# 如果你的 .cu 文件都是独立的，或者只通过 C++ 函数相互调用，则不需要开启。
# 开启会增加编译时间。
# set_target_properties(operators PROPERTIES CUDA_SEPARABLE_COMPILATION ON)