// matrix_transpose_gpu.cu
#include <iostream>
#include <vector>
#include <cmath> // For std::abs in verification

#define TILE_DIM 32 // Tile dimension for shared memory

// Helper macro for CUDA error checking
#define CUDA_CHECK(err) \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << " in " << __FILE__ << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    }

__global__ void matrixTransposeKernel(const float* input, float* output, int width, int height) {
    // Shared memory tile: TILE_DIM x TILE_DIM. Add padding to one dimension to reduce bank conflicts.
    __shared__ float tile[TILE_DIM][TILE_DIM + 1]; // Note: TILE_DIM rows, TILE_DIM+1 columns

    // Calculate global indices for reading from input matrix based on block and thread IDs
    // Each thread in the block reads one element of the input matrix tile
    int x_in_global = blockIdx.x * TILE_DIM + threadIdx.x; // Global column index for input
    int y_in_global = blockIdx.y * TILE_DIM + threadIdx.y; // Global row index for input

    // Load data into shared memory from global memory (input)
    // Ensure threads do not read out of bounds for matrices not perfectly divisible by TILE_DIM
    if (x_in_global < width && y_in_global < height) {
        tile[threadIdx.y][threadIdx.x] = input[y_in_global * width + x_in_global];
    }

    // Synchronize to ensure all threads in the block have loaded their data into shared memory
    __syncthreads();

    // Calculate global indices for writing to output matrix (transposed)
    // The block that was processing block (blockIdx.x, blockIdx.y) of the input
    // will write to block (blockIdx.y, blockIdx.x) of the output.
    // threadIdx.x becomes the row offset within the shared memory tile (for reading)
    // threadIdx.y becomes the col offset within the shared memory tile (for reading)
    int x_out_global = blockIdx.y * TILE_DIM + threadIdx.x; // Global column index for output
    int y_out_global = blockIdx.x * TILE_DIM + threadIdx.y; // Global row index for output

    // Write data from shared memory to global memory (output) in transposed order
    // Ensure threads do not write out of bounds
    // The output matrix dimensions will be height (as width) x width (as height)
    if (x_out_global < height && y_out_global < width) { // Output width is original height, output height is original width
        output[y_out_global * height + x_out_global] = tile[threadIdx.x][threadIdx.y]; // Read from shared mem transposed
    }
}

// CPU transpose for verification
void cpuTranspose(const std::vector<float>& input, std::vector<float>& output_cpu, int width, int height) {
    // Output matrix (output_cpu) will have dimensions: rows = width, cols = height
    for (int r = 0; r < height; ++r) { // Iterate through rows of input matrix
        for (int c = 0; c < width; ++c) { // Iterate through columns of input matrix
            output_cpu[c * height + r] = input[r * width + c];
        }
    }
}

int main() {
    const int input_height = 1 << 9;  // 512 rows for input
    const int input_width  = 1 << 10; // 1024 columns for input

    // Output matrix dimensions will be: output_height = input_width, output_width = input_height
    const int output_height = input_width;
    const int output_width  = input_height;


    std::cout << "Input Matrix dimensions: " << input_height << "x" << input_width << std::endl;
    std::cout << "Output Matrix dimensions: " << output_height << "x" << output_width << std::endl;


    // Allocate host memory
    std::vector<float> h_input(input_width * input_height);
    std::vector<float> h_output_gpu(output_width * output_height); // Same total number of elements
    std::vector<float> h_output_cpu(output_width * output_height);


    // Initialize host input matrix
    for (int i = 0; i < input_width * input_height; ++i) {
        h_input[i] = static_cast<float>(i % 100); // Keep values small for easier debugging
    }

    // Declare device pointers
    float *d_input = nullptr, *d_output = nullptr;

    // Allocate device memory
    CUDA_CHECK(cudaMalloc((void**)&d_input, input_width * input_height * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_output, output_width * output_height * sizeof(float)));

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), input_width * input_height * sizeof(float), cudaMemcpyHostToDevice));

    // Define execution configuration (2D grid and blocks)
    dim3 threadsPerBlock(TILE_DIM, TILE_DIM);
    // Number of blocks needed to cover the input matrix
    dim3 numBlocks((input_width + TILE_DIM - 1) / TILE_DIM, (input_height + TILE_DIM - 1) / TILE_DIM);

    std::cout << "Launching kernel with " << numBlocks.x << "x" << numBlocks.y << " blocks and "
              << threadsPerBlock.x << "x" << threadsPerBlock.y << " threads per block." << std::endl;

    // Launch the kernel: Pass input dimensions (width, height)
    matrixTransposeKernel<<<numBlocks, threadsPerBlock>>>(d_input, d_output, input_width, input_height);
    CUDA_CHECK(cudaGetLastError()); // Check for kernel launch errors

    // Synchronize to ensure kernel completion before copying back
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy data from device to host
    CUDA_CHECK(cudaMemcpy(h_output_gpu.data(), d_output, output_width * output_height * sizeof(float), cudaMemcpyDeviceToHost));

    // Perform CPU transpose for verification
    cpuTranspose(h_input, h_output_cpu, input_width, input_height);

    // Verify results
    bool success = true;
    int errors = 0;
    const int max_errors_to_print = 5;
    for (int r_out = 0; r_out < output_height; ++r_out) { // Iterate output rows (original input_width)
        for (int c_out = 0; c_out < output_width; ++c_out) { // Iterate output columns (original input_height)
            int gpu_idx = r_out * output_width + c_out;
            int cpu_idx = r_out * output_width + c_out; // Same linear indexing for output comparison

            if (std::abs(h_output_gpu[gpu_idx] - h_output_cpu[cpu_idx]) > 1e-5) {
                if (errors < max_errors_to_print) {
                     // Original element was at h_input[c_out * input_width + r_out]
                    std::cerr << "Verification failed at output_gpu(" << r_out << "," << c_out << ") linear_idx=" << gpu_idx
                              << ". GPU: " << h_output_gpu[gpu_idx] << ", CPU: " << h_output_cpu[cpu_idx]
                              << ", (Original input was at input[" << c_out << "][" << r_out << "]="
                              << h_input[c_out * input_width + r_out] << ")"
                              << std::endl;
                }
                errors++;
                success = false;
            }
        }
    }
     if (errors > max_errors_to_print) {
        std::cerr << "...and " << (errors - max_errors_to_print) << " more errors." << std::endl;
    }


    if (success) {
        std::cout << "CUDA Matrix Transpose: Verification successful!" << std::endl;
    } else {
        std::cout << "CUDA Matrix Transpose: Verification FAILED with " << errors << " mismatches." << std::endl;
    }

    // Free device memory
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    std::cout << "CUDA C++ Matrix Transpose kernel execution complete." << std::endl;
    return 0;
}