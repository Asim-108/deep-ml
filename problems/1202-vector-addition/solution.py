#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <cmath>

__global__ void add_kernel(const float* a, const float* b, float* c, int n) {
    // c[i] = a[i] + b[i], guarded by i < n
    int myIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (myIdx >= n){
        return;
    }
    c[myIdx] = a[myIdx] + b[myIdx];
}

std::vector<float> vector_add(const std::vector<float>& a, const std::vector<float>& b) {
    // allocate a, b, c on the device; copy a and b over (Host -> Device);
    // launch the kernel; copy c back; free memory
    int n = a.size();
    // check if vectors empty
    if (n == 0){
        return {};
    }
    // create pointers
    float* a_d = nullptr;
    float* b_d = nullptr;
    float* c_d = nullptr;
    // allocate memory on device
    cudaMalloc((void**) &a_d, sizeof(float) * n);
    cudaMalloc((void**) &b_d, sizeof(float) * n);
    cudaMalloc((void**) &c_d, sizeof(float) * n);
    // copy to device memory
    cudaMemcpy(a_d, a.data(), sizeof(float) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b.data(), sizeof(float) * n, cudaMemcpyHostToDevice);
    // set threadsPerBlock, calculate number of blocks needed based on that
    int threadsPerBlock = 256;
    // equivalent of ceil() operation with int
    int numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    // launch kernel
    add_kernel<<<numBlocks, threadsPerBlock>>>(a_d, b_d, c_d, n);

    // create vector on host memory (on heap)
    std::vector<float> vec(n);
    // copy from device to host memory
    cudaMemcpy(vec.data(), c_d, sizeof(float) * n, cudaMemcpyDeviceToHost);
    // free allocated device memory
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);

    return vec;
}
