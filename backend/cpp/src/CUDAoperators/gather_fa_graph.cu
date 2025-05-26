#include <float.h>

#include <cmath>
#include <limits>
#include <vector>

#include "cudaOP.cuh"

#define MAX_BRANCHES 3

namespace cuda_OP {

// CUDA图优化版本的gather_fa kernel
// 仿照gather_fa_variable的模式，支持最多3个分支
template <typename T>
__global__ void gather_fa_kernel_graph_fixed(T **input_ptrs,
                                             T *output_ptr,
                                             int *segment_info,
                                             int q_h, int dqkv) {
  // 从设备内存读取分支信息
  int active_branches = segment_info[1];

  // 限制最大分支数
  if (active_branches > MAX_BRANCHES) active_branches = MAX_BRANCHES;

  // 每个 block 对应一个 head，每个线程处理该 head 中一个维度
  int head_id = blockIdx.x;
  int tid = threadIdx.x;
  if (head_id >= q_h || tid >= dqkv) {
    return;
  }

  // 每个 head 中 T 张量的 stride（最后 2 个元素用于存储 m 与 l）
  int input_stride = dqkv + 2;
  int output_stride = dqkv;
  int base_in = head_id * input_stride;
  int base_out = head_id * output_stride;

  // 初始化归约变量
  float global_m = -FLT_MAX;
  float global_l = 0.0f;
  float global_o = 0.0f;

  // 处理第一个分支
  if (active_branches > 0 && input_ptrs[0] != nullptr) {
    T *T1_ptr = input_ptrs[0];
    float m1 = static_cast<float>(T1_ptr[base_in + dqkv]);
    float l1 = static_cast<float>(T1_ptr[base_in + dqkv + 1]);
    float o1 = static_cast<float>(T1_ptr[base_in + tid]);

    global_m = m1;
    global_l = l1;
    global_o = o1;
  }

  // 处理第二个分支
  if (active_branches > 1 && input_ptrs[1] != nullptr) {
    T *T2_ptr = input_ptrs[1];
    float m2 = static_cast<float>(T2_ptr[base_in + dqkv]);
    float l2 = static_cast<float>(T2_ptr[base_in + dqkv + 1]);
    float o2 = static_cast<float>(T2_ptr[base_in + tid]);

    float old_global_m = global_m;
    float old_global_l = global_l;
    float new_global_m = fmaxf(old_global_m, m2);
    float exp_old = __expf(old_global_m - new_global_m);
    float exp_cur = __expf(m2 - new_global_m);
    global_l = old_global_l * exp_old + l2 * exp_cur;
    global_o = global_o * exp_old + o2 * exp_cur;
    global_m = new_global_m;
  }

  // 处理第三个分支
  if (active_branches > 2 && input_ptrs[2] != nullptr) {
    T *T3_ptr = input_ptrs[2];
    float m3 = static_cast<float>(T3_ptr[base_in + dqkv]);
    float l3 = static_cast<float>(T3_ptr[base_in + dqkv + 1]);
    float o3 = static_cast<float>(T3_ptr[base_in + tid]);

    float old_global_m = global_m;
    float old_global_l = global_l;
    float new_global_m = fmaxf(old_global_m, m3);
    float exp_old = __expf(old_global_m - new_global_m);
    float exp_cur = __expf(m3 - new_global_m);
    global_l = old_global_l * exp_old + l3 * exp_cur;
    global_o = global_o * exp_old + o3 * exp_cur;
    global_m = new_global_m;
  }

  // 最终归一化
  float final_out = (global_l > 0.0f) ? global_o / global_l : 0.0f;
  output_ptr[base_out + tid] = static_cast<T>(final_out);
}

// CUDA图优化版本：使用固定内存地址的gather_fa
template <typename T>
void gather_fa_graph_fixed(T **d_input_ptrs,
                           Tensor<T> &output,
                           int *d_segment_info,
                           cudaStream_t stream) {
  int dqkv = output.sizes()[1];
  int q_h = output.sizes()[0];

  // 设置kernel参数
  dim3 grid(q_h);
  dim3 block(dqkv);

  // 启动kernel
  gather_fa_kernel_graph_fixed<T><<<grid, block, 0, stream>>>(
      d_input_ptrs, output.data_ptr(), d_segment_info, q_h, dqkv);

  // 检查错误
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error("CUDA error in gather_fa_graph_fixed: " +
                            std::string(cudaGetErrorString(err)));
  }
}

// 显式模板实例化
template void gather_fa_graph_fixed<float>(
    float **d_input_ptrs,
    Tensor<float> &output,
    int *d_segment_info,
    cudaStream_t stream);

template void gather_fa_graph_fixed<__nv_bfloat16>(
    __nv_bfloat16 **d_input_ptrs,
    Tensor<__nv_bfloat16> &output,
    int *d_segment_info,
    cudaStream_t stream);

}  // namespace cuda_OP
