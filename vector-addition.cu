/**
 * CUDA Basic Example - Vector Addition
 * 
 * This program demonstrates the fundamental concepts of CUDA programming:
 * 1. Allocating memory on the GPU
 * 2. Copying data between CPU and GPU
 * 3. Executing a kernel function on the GPU
 * 4. Synchronization between CPU and GPU
 * 5. Cleaning up resources
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h> // for fabs
#include <cuda_runtime.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)


// CUDA Kernel function to add two vectors

__global__ void vectorAdd(const float* A, const float* B, float* C, int numElements){
    // number of threads per block * block_idx + thread_idx

    int i = blockDim.x * blockIdx.x + threadIdx.x;

    // since padded blocks look for oob error

    if(i < numElements){
        C[i] = A[i] * B[i] + C[i];
        // C[i] = A[i] * B[i];
    }
}

int main(){

    int numElements = 50000;
    size_t size = numElements * sizeof(float);

    printf("Vector Addition of %d elements \n", numElements);

    // Allocate Host Memory

    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    //Allocate Device Memory
    float *d_A = NULL;
    float *d_B = NULL;
    float *d_C = NULL;

    CUDA_CHECK(cudaMalloc((void **)&d_A, size));
    CUDA_CHECK(cudaMalloc((void **)&d_B, size));
    CUDA_CHECK(cudaMalloc((void **)&d_C, size));

    //copy from host to device(GPU)
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    //Launch the cuda Kernel
    int threadsPerBlock = 256;  // 256 is a reasonable starting point, not universally optimal.

    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

    printf("CUDA kernel launch with %d blocks of %d threads \n", blocksPerGrid, threadsPerBlock);

    //The triple angle bracket syntax <<<blocksPerGrid, threadsPerBlock>>> is CUDA's way of specifying the execution configuration.
    // We're telling the GPU to launch 196 blocks with 256 threads each, giving us 50,176 total threads (slightly more than our 50,000 elements).


    //  cuda event to create timer start and end 
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));


    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);


    // launched now record stop 
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("Kernel execution time: %.3f ms\n", milliseconds);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    //Since it is asynchronous process cpu tell gpu to do something and go for next task
    //so always check for suucessful gpu implementation

    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());


    // First: cudaGetLastError()
    // Checks whether the kernel launch itself failed — for example, because the launch configuration was invalid.

     // Second: cudaDeviceSynchronize()
    // Waits for the GPU to finish executing the kernel and detects execution errors, such as certain illegal memory accesses.

    cudaError_t err = cudaGetLastError();

    // if(err!= cudaSuccess)
    // {
    //     fprintf(stderr, "Failed to Launch kernel: %s\n", cudaGetErrorString(err));
    //     exit(EXIT_FAILURE);
    // }

    //copy rsult from gpu to cpu Most mportant

    CUDA_CHECK(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));


    //verify the result 

    for(int i = 0; i < numElements; i++){
        if(fabsf((h_A[i] + h_B[i] - h_C[i]) > 1e-5f)){
            fprintf(stderr, "Result Verification failed at element %d! \n", i);
            exit(EXIT_FAILURE);
        }
    }


    printf("Test Passed \n");

    
    // Free device memory
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    
    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);
    
    printf("Done\n");
    return 0;

}


// To check the kernel occupancy 
// nvcc --ptxas-options=-v vector-addition.cu -o vector-addition

// nvcc --ptxas-options=-v vector-addition.cu -o 01-vector-addition
// ptxas info    : 0 bytes gmem
// ptxas info    : Compiling entry function '_Z9vectorAddPKfS0_Pfi' for 'sm_52'
// ptxas info    : Function properties for _Z9vectorAddPKfS0_Pfi
//      0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
// ptxas info    : Used 8 registers, 348 bytes cmem[0]


// Debugging Your CUDA Code
// When things go wrong, CUDA provides several debugging tools:

// cuda-memcheck: Detects memory errors like out-of-bounds access

// cuda-memcheck ./01-vector-addition

// cuda-gdb: GPU debugger for step-through debugging

// cuda-gdb ./01-vector-addition



//Unified Memory access
// To rmeove cpy from host ot device and vice versa 

// Use CUDA_CHECK(cudaMallocManaged(&d_A, size));
// CUDA_CHECK(cudaMallocManaged(&d_B, size));
// CUDA_CHECK(cudaMallocManaged(&d_C, size));