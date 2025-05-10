
#include <pthread.h>
#include <sched.h>
#include <unistd.h>

#include <iostream>

#include "tensor.hpp"
template <typename T>
void debugPrintTensor(const Tensor<T>& tensor, const std::string& tensor_name, size_t num_to_print = 10) {
    std::cout << "[Debug] " << tensor_name << ":\n";

    // 1) Print shape
    std::cout << "  shape: [";
    for (auto s : tensor.sizes()) {
        std::cout << s << " ";
    }
    std::cout << "]\n";

    // 2) Print strides
    std::cout << "  strides: [";
    for (auto st : tensor.strides()) {
        std::cout << st << " ";
    }
    std::cout << "]\n";

    // 3) Print device
    std::cout << "  device: ";
    if (tensor.device() == Device::CPU) {
        std::cout << "CPU";
    } else if (tensor.device() == Device::CUDA) {
        std::cout << "CUDA";
    } else {
        std::cout << "UNKNOWN";
    }
    std::cout << "\n";

    // 4) Print elements starting from offset 0
    size_t offset = 0;  // 从开始处打印
    size_t total_elements = tensor.numel();
    size_t n_print = std::min(num_to_print, total_elements - offset);

    std::cout << "  elements from offset " << offset << " (" << n_print << " element(s)): ";
    if (tensor.device() == Device::CPU) {
        const T* ptr = tensor.data_ptr();
        for (size_t i = 0; i < n_print; i++) {
            std::cout << ptr[offset + i] << " ";
        }
        std::cout << "\n";
    } else {
        // Copy from GPU to CPU, then print
        std::vector<T> host_buffer(n_print);
        cudaError_t err =
            cudaMemcpy(host_buffer.data(), tensor.data_ptr() + offset, n_print * sizeof(T), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            std::cout << "  [Error] cudaMemcpy failed\n";
            return;
        }
        for (size_t i = 0; i < n_print; i++) {
            std::cout << host_buffer[i] << " ";
        }
        std::cout << "\n";
    }
}

void bind_this_thread_to_core(int core_id) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    pthread_t current_thread = pthread_self();

    int rc = pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset);
    if (rc != 0) {
        std::cerr << "Error calling pthread_setaffinity_np: " << strerror(rc) << "\n";
    }
}

// auto print_shape =
//     [](const std::string &name,
//        const auto &sizes) { /* ... (printing code) ... */
//                             std::cout << name << " shape: [";
//                             if (sizes.empty()) {
//                               std::cout << "<empty>";
//                             } else {
//                               for (size_t i = 0; i < sizes.size(); ++i)
//                               {
//                                 std::cout
//                                     << sizes[i]
//                                     << (i == sizes.size() - 1 ? ""
//                                                               : ", ");
//                               }
//                             }
//                             std::cout << "]";
//     };
