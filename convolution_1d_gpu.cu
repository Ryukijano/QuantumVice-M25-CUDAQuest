// convolution_1d_gpu.cu
#include <iostream>
#include <vector>
#include <cmath> // For std::abs

#define CUDA_CHECK(err) \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << " in " << __FILE__ << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    }

// Kernel for 1D convolution using shared memory
// Assumes filter_width is odd for a symmetric radius
__global__ void convolution1DSharedMem(const float* input, float* output, const float* filter, int N, int filter_width) {
    // Shared memory for caching a segment of the input array
    // Size needed: blockDim.x (for the core elements) + (filter_width - 1) (for halo/ghost cells)
    extern __shared__ float s_data[]; // Dynamically allocated shared memory

    int tid_local = threadIdx.x; // Thread ID within the block
    int block_start_global = blockIdx.x * blockDim.x; // Global start index for this block's core output
    int global_thread_id = block_start_global + tid_local; // Global thread ID, corresponds to output index

    int filter_radius = filter_width / 2;

    // Load input data into shared memory
    // Each thread loads one element, potentially more if blockDim.x is small relative to shared mem size
    // This example: each thread loads one element for its core computation, plus halo elements are loaded by relevant threads
    
    // Load elements for the current block's computation range + halo regions
    // Left halo
    if (tid_local < filter_radius) {
        int load_idx_global = block_start_global - filter_radius + tid_local;
        if (load_idx_global >= 0) {
            s_data[tid_local] = input[load_idx_global];
        } else {
            s_data[tid_local] = 0.0f; // Zero-padding for out-of-bounds
        }
    }

    // Main part
    int main_load_idx_global = block_start_global + tid_local;
    if (main_load_idx_global < N) {
         s_data[tid_local + filter_radius] = input[main_load_idx_global];
    } else {
         s_data[tid_local + filter_radius] = 0.0f; // Zero-padding
    }

    // Right halo
    if (tid_local >= blockDim.x - filter_radius && blockDim.x < (filter_radius * 2 + 1) /*Ensure not double loading from small blocks*/) {
        int load_idx_global = block_start_global + blockDim.x + (tid_local - (blockDim.x - filter_radius));
         if (load_idx_global < N) {
            s_data[tid_local + filter_radius + filter_radius] = input[load_idx_global];
         } else {
            s_data[tid_local + filter_radius + filter_radius] = 0.0f;
         }
    }
    // A simpler loading strategy: each thread loads input[global_thread_id] to s_data[tid_local + filter_radius]
    // and then threads at edges of block load halo. Or, each thread loads multiple points if necessary.
    // For a robust version, consider loading (blockDim.x + filter_width - 1) elements starting from
    // block_start_global - filter_radius. Each thread can load one or more elements.

    // Example: Simpler loading (each thread loads one element into its "main" spot in shared memory)
    // This needs to be adjusted for halo loading. The above is a more complex attempt.
    // A common pattern:
    // int shared_mem_idx = tid_local + filter_radius;
    // if (global_thread_id < N) s_data[shared_mem_idx] = input[global_thread_id]; else s_data[shared_mem_idx] = 0.0f;
    // if (tid_local < filter_radius) { // Load left halo
    //    int left_halo_global_idx = global_thread_id - filter_radius + tid_local; // Incorrect, this should be block_start_global - filter_radius + tid_local
    //    if (left_halo_global_idx >=0) s_data[tid_local] = input[left_halo_global_idx]; else s_data[tid_local] = 0.0f;
    //    int right_halo_global_idx = global_thread_id + blockDim.x + tid_local; // Incorrect, this should be block_start_global + blockDim.x + tid_local
    //    if (right_halo_global_idx < N) s_data[tid_local + filter_radius + blockDim.x] = input[right_halo_global_idx]; else s_data[tid_local + filter_radius + blockDim.x] = 0.0f;
    // }
    // The loading part is tricky and needs care. A simpler first pass might not use shared memory or use a naive shared memory load.
    // For a robust shared memory load:
    int start_load_idx_global = block_start_global - filter_radius;
    for (int i = tid_local; i < blockDim.x + filter_width -1; i += blockDim.x) {
        int current_load_global = start_load_idx_global + i;
        if (current_load_global >= 0 && current_load_global < N) {
            s_data[i] = input[current_load_global];
        } else {
            s_data[i] = 0.0f; // Zero padding
        }
    }


    __syncthreads(); // Ensure all data is loaded into shared memory

    // Perform convolution using data from shared memory
    if (global_thread_id < N) {
        float sum = 0.0f;
        for (int j = 0; j < filter_width; ++j) {
            // Index into shared memory: tid_local (base for current thread's perspective) + filter_radius (to center it) + j (filter index) - filter_radius (to align filter)
            // This simplifies to: s_data[tid_local + j]
            sum += s_data[tid_local + j] * filter[j];
        }
        output[global_thread_id] = sum;
    }
}


void cpuConvolution1D(const std::vector<float>& input, std::vector<float>& output_cpu, const std::vector<float>& filter) {
    int N = input.size();
    int filter_width = filter.size();
    int filter_radius = filter_width / 2;
    output_cpu.assign(N, 0.0f);

    for (int i = 0; i < N; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < filter_width; ++j) {
            int input_idx = i + j - filter_radius;
            if (input_idx >= 0 && input_idx < N) {
                sum += input[input_idx] * filter[j];
            }
            // Implicit zero-padding for out-of-bounds
        }
        output_cpu[i] = sum;
    }
}


int main() {
    const int N = 1 << 10; // Input signal size
    const int filter_width = 5; // Example filter width (must be odd for simple radius)
    
    std::vector<float> h_input(N);
    std::vector<float> h_filter(filter_width);
    std::vector<float> h_output_gpu(N);
    std::vector<float> h_output_cpu(N);

    // Initialize input and filter
    for(int i = 0; i < N; ++i) h_input[i] = static_cast<float>(i % 10); // Simple pattern
    for(int i = 0; i < filter_width; ++i) h_filter[i] = 1.0f / filter_width; // Averaging filter

    float *d_input = nullptr, *d_output = nullptr, *d_filter = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_filter, filter_width * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_filter, h_filter.data(), filter_width * sizeof(float), cudaMemcpyHostToDevice));

    int threadsPerBlock = 256;
    int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    // Shared memory size: (threadsPerBlock + filter_width - 1) elements
    size_t shared_mem_size = (threadsPerBlock + filter_width - 1) * sizeof(float);
    std::cout << "Requesting " << shared_mem_size << " bytes of dynamic shared memory per block." << std::endl;


    convolution1DSharedMem<<<numBlocks, threadsPerBlock, shared_mem_size>>>(d_input, d_output, d_filter, N, filter_width);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_output_gpu.data(), d_output, N * sizeof(float), cudaMemcpyDeviceToHost));

    // CPU verification
    cpuConvolution1D(h_input, h_output_cpu, h_filter);
    bool success = true;
    int errors = 0;
    for(int i=0; i<N; ++i) {
        if (std::abs(h_output_gpu[i] - h_output_cpu[i]) > 1e-4) { // Tolerance for float comparisons
            if(errors < 5) std::cerr << "Mismatch at " << i << ": GPU=" << h_output_gpu[i] << ", CPU=" << h_output_cpu[i] << std::endl;
            errors++;
            success = false;
        }
    }
    if(success) std::cout << "1D Convolution: Verification SUCCESSFUL!" << std::endl;
    else std::cout << "1D Convolution: Verification FAILED with " << errors << " errors." << std::endl;


    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_filter));

    std::cout << "CUDA C++ 1D Convolution kernel execution complete." << std::endl;
    return 0;
}