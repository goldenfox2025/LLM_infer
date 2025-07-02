#include "cudaOP.cuh"

namespace cuda_OP {

template <typename T, int intermediate_size = 8960>
__global__ void silu_multiply_kernel(T *output, const T *input, const T *input2, int in1_stride0, int in2_stride0,
                                     int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = idx; i < total; i += blockDim.x * gridDim.x) {
        int cur_row = i / intermediate_size;
        int cur_col = i % intermediate_size;

        float x = static_cast<float>(input[cur_row * in1_stride0 + cur_col]);
        x = x / (1.0f + expf(-x));
        output[i] = static_cast<T>(x * static_cast<float>(input2[cur_row * in2_stride0 + cur_col]));
    }
}

// 方便起见，我们默认这个仅适用于mlp融合和非mlp融合状态
// 也就是两个输入的第一维可能连续可能不连续，别的都连续
// 解耦total和线程块参数试试看
template <typename T>
void silu_multiply(Tensor<T> *output, const Tensor<T> *input, const Tensor<T> *input2, cudaStream_t stream) {
    int in1_stride0 = input->strides()[0];
    int in2_stride0 = input2->strides()[0];

    dim3 grid(64, 1, 1);
    dim3 block(256, 1, 1);
    int intermediate_size = input->sizes()[1];
    if (intermediate_size != 8960) {
        throw std::runtime_error("silu_multiply: intermediate_size != 8960");
    }
    int n0 = input->sizes()[0];
    int total = n0 * intermediate_size;
    silu_multiply_kernel<T><<<grid, block, 0, stream>>>(output->data_ptr(), input->data_ptr(), input2->data_ptr(),
                                                        in1_stride0, in2_stride0, total);
    checkCudaError(cudaGetLastError());
}

template void silu_multiply<float>(Tensor<float> *, const Tensor<float> *, const Tensor<float> *, cudaStream_t);
template void silu_multiply<nvbf16>(Tensor<nvbf16> *, const Tensor<nvbf16> *, const Tensor<nvbf16> *, cudaStream_t);
}  // namespace cuda_OP