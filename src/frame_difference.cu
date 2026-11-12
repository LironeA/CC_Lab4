#include "frame_difference.cuh"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

int available_gpu_count() {
    int devices = 0;
    check_cuda(cudaGetDeviceCount(&devices), "cudaGetDeviceCount");
    return devices;
}

__global__ void difference_kernel(const unsigned char* previous, const unsigned char* current,
                                  unsigned char* difference, std::size_t bytes) {
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < bytes) {
        const int value = static_cast<int>(current[index]) - static_cast<int>(previous[index]);
        difference[index] = static_cast<unsigned char>(value < 0 ? -value : value);
    }
}

void difference_on_gpu(int device, const unsigned char* previous, const unsigned char* current,
                       unsigned char* difference, std::size_t bytes) {
    check_cuda(cudaSetDevice(device), "cudaSetDevice");

    unsigned char *previous_gpu, *current_gpu, *difference_gpu;
    check_cuda(cudaMalloc(&previous_gpu, bytes), "cudaMalloc previous");
    check_cuda(cudaMalloc(&current_gpu, bytes), "cudaMalloc current");
    check_cuda(cudaMalloc(&difference_gpu, bytes), "cudaMalloc difference");

    check_cuda(cudaMemcpy(previous_gpu, previous, bytes, cudaMemcpyHostToDevice), "copy previous to GPU");
    check_cuda(cudaMemcpy(current_gpu, current, bytes, cudaMemcpyHostToDevice), "copy current to GPU");

    constexpr int threads = 256;
    const int blocks = static_cast<int>((bytes + threads - 1) / threads);
    difference_kernel<<<blocks, threads>>>(previous_gpu, current_gpu, difference_gpu, bytes);
    check_cuda(cudaGetLastError(), "difference kernel");
    check_cuda(cudaMemcpy(difference, difference_gpu, bytes, cudaMemcpyDeviceToHost), "copy result to CPU");

    cudaFree(previous_gpu);
    cudaFree(current_gpu);
    cudaFree(difference_gpu);
}
