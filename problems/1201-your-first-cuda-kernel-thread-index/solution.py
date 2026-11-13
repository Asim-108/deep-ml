#include <cuda_runtime.h>
#include <vector>

__global__ void index_kernel(int* out, int n) {
    // Compute this thread's global index and, if it is < n, write it to out.
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n){
        out[i] = i;
    }
}

std::vector<int> global_thread_indices(int n) {
    // 1. allocate device memory for n ints
    // 2. launch the kernel with enough threads to cover n
    // 3. copy the result back to the host and return it
    if (n <= 0){
        return {};
    }
    // define pointer for memory allocated on device
    int* ptr = nullptr;
    cudaMalloc(&ptr, n * sizeof(int));
    // threads we execute per block (number of workers per group)
    int threadsPerBlock = 256;
    // number of blocks needed for all elements
    // ceiling function so that we have enough blocks with enough workers for integers given
    int numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;

    // launch kernel with number of blocks and threads per block, pass ptr with allocated memory and n as arguments
    index_kernel<<<numBlocks, threadsPerBlock>>>(ptr, n);

    // create vector on host memory (on heap)
    std::vector<int> vec(n);
    // copy from device to host memory
    cudaMemcpy(vec.data(), ptr, sizeof(int) * n, cudaMemcpyDeviceToHost);
    // free allocated device memory
    cudaFree(ptr);

    return vec;
}
