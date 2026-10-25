//When you write CUDA code, it doesn't directly become machine instructions that run on your GPU. 
// Instead, it goes through several compilation stages. 
// Understanding PTX sits at the sweet spot between high-level CUDA C++ and low-level GPU machine code (SASS).

#include <stdio.h>
#include <string>
#include <fstream>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
using namespace std;

// CUDA C++ (.cu)
//     ↓ [nvcc frontend]
// PTX Assembly (.ptx)
//     ↓ [ptxas assembler]
// SASS Machine Code (.cubin)
//     ↓ [driver at runtime]
// GPU Execution


// Architecture Independence: PTX code can run on different GPU architectures. 
// The driver compiles PTX to actual machine code for your specific GPU at runtime. 
// This means code compiled years ago can still take advantage of newer GPUs.


// Performance Tuning: Sometimes the CUDA compiler makes conservative choices. 
// By writing inline PTX, 
// you can hand-optimize critical sections when profiling shows they're bottlenecks.


// Understanding Compiler Output: 
// Reading PTX helps you understand what the compiler does with your code. 
// You can spot inefficiencies and write better high-level code.


// Low-Level Control: 
// PTX gives you access to special instructions and hardware features not exposed in CUDA C++.


// Basic Inline PTX Syntax
// The syntax for inline PTX in CUDA is:

// asm("instruction" : output_operands : input_operands : clobbered_registers);



// __device__ int addTwoNumbers(int a, int b){
//     int result;
//     asm("add.s32 %0, %1, %2;" : "=r"(result) : "r"(a) : "r"(b));
//     return result;
// }



// Method 1: Device function pointer approach
// This requires the PTX function to be linked at compile time
// We'll implement this function using inline PTX


__device__ void vector_add_ptx_device(int* a, int* b, int* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Using inline PTX assembly for vector addition
        int temp_result;
        asm("add.s32 %0, %1, %2;" : "=r"(temp_result) : "r"(a[idx]), "r"(b[idx]));
        result[idx] = temp_result;
    }
}

// Device function using function pointer
__device__ void device_vector_add_func(int* a, int* b, int* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int temp_result;
        asm("add.s32 %0, %1, %2;" : "=r"(temp_result) : "r"(a[idx]), "r"(b[idx]));
        result[idx] = temp_result;
    }
}

// The fma.rn.f32 instruction: "fma.rn.f32 %0, %1, %2, %3;"
// fma: Fused multiply-add
// rn: Rounding mode (round to nearest even)
// f32: 32-bit floating-point
// Computes: result = a * b + c


// __device__ float fma_custom(float a, float b, float c) {
//     float result;
//     asm("fma.rn.f32 %0, %1, %2, %3;"
//         : "=f"(result)
//         : "f"(a), "f"(b), "f"(c));
//     return result;
// }



// Regular CUDA

__global__ void saxpy_simple (float a, float* x , float* y, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n){
        y[i] = a * x[i] + y[i];
    }
}



// In line PTX:
__global__ void saxpy_ptx(float a, float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        float xi = x[i];
        float yi = y[i];
        float result;

        asm("fma.rn.f32 %0, %1, %2, %3;"
            : "=f"(result)
            : "f"(a), "f"(xi), "f"(yi));

        y[i] = result;
    }
}

// nvcc -ptx -o saxpy_simple.ptx saxpy.cu

// make 02-ptx-assembly
// ./02-ptx-assembly



// Understanding PTX Instruction Types
// Arithmetic Instructions
// add.s32  %r1, %r2, %r3;         // Integer addition
// sub.f32  %f1, %f2, %f3;         // Float subtraction
// mul.lo.s32  %r1, %r2, %r3;      // Integer multiply (low 32 bits)
// div.rn.f32  %f1, %f2, %f3;      // Float division (round nearest)
// mad.lo.s32  %r1, %r2, %r3, %r4; // Multiply-add
// fma.rn.f32  %f1, %f2, %f3, %f4; // Fused multiply-add

// Comparison and Selection
// setp.eq.s32  %p1, %r1, %r2;     // p1 = (r1 == r2)
// setp.lt.f32  %p1, %f1, %f2;     // p1 = (f1 < f2)
// selp.s32  %r1, %r2, %r3, %p1;   // r1 = p1 ? r2 : r3

// Bit Manipulation
// and.b32  %r1, %r2, %r3;         // Bitwise AND
// or.b32  %r1, %r2, %r3;          // Bitwise OR
// xor.b32  %r1, %r2, %r3;         // Bitwise XOR
// shl.b32  %r1, %r2, 3;           // Shift left by 3
// shr.u32  %r1, %r2, 3;           // Unsigned shift right

// Special Functions
// sin.approx.f32  %f1, %f2;       // Fast sine approximation
// cos.approx.f32  %f1, %f2;       // Fast cosine approximation
// rsqrt.approx.f32  %f1, %f2;     // Fast reciprocal square root


// Method 2  : Inline PTX assembly that calls another PTX fucntion

__device__ int addTwoNumbers(int a, int b){
    int result;

    asm("add.s32 %0, %1, %2;" : "=r"(result) : "r"(a) , "r"(b));

    return result;
}


// Register Declaration
// .reg .pred  %p<2>;      // Predicate registers (for conditionals)
// .reg .b32   %r<5>;      // 32-bit integer registers
// .reg .b64   %rd<11>;    // 64-bit integer registers
// .reg .f32   %f<8>;      // 32-bit floating-point registers


//PTX uses virtual registers with unlimited supply. 
// The actual hardware has a limited number of registers per thread, 
// so ptxas maps these virtual registers to physical ones. 
// This is why high register usage can limit occupancy.


// The mad.lo.s32 instruction performs multiply-add in a single operation. This is a fused multiply-add (FMA), which is both faster and more accurate than separate multiply and add.

// The special registers %ctaid, %ntid, and %tid correspond to CUDA built-ins:

// %ctaid: Block index (ctaid = compute task ID)
// %ntid: Block dimension (ntid = number of threads)
// %tid: Thread index within block



// Memory Operations
// Loading from global memory:

// ld.global.f32  %f1, [%rd1];     // Load 32-bit float from address in rd1

// Storing to global memory:

// st.global.f32  [%rd2], %f1;     // Store f1 to address in rd2

// The .global qualifier specifies the memory space. Other options include .shared, .local, .const, and .param.

// Control Flow
// PTX uses predicates and conditional branches:

// setp.ge.s32  %p1, %r1, %r2;     // p1 = (r1 >= r2)
// @%p1 bra  BB0_2;                // Branch to BB0_2 if p1 is true

// The setp instruction sets a predicate register based on a comparison. The @%p1 prefix makes the branch conditional on that predicate. This is more efficient than traditional if-else branches because the GPU can predicate instructions rather than diverge execution.






// Memory Spaces and Qualifiers
// PTX distinguishes different memory spaces:

// Global Memory (.global):

// ld.global.f32  %f1, [%rd1];
// st.global.f32  [%rd1], %f1;

// Highest capacity, highest latency. Accessible by all threads.

// Shared Memory (.shared):

// ld.shared.f32  %f1, [%rd1];
// st.shared.f32  [%rd1], %f1;

// Low latency, limited capacity. Shared within a thread block.

// Constant Memory (.const):

// ld.const.f32  %f1, [%rd1];

// Read-only, cached. Good for broadcast patterns.

// Local Memory (.local):

// ld.local.f32  %f1, [%rd1];
// st.local.f32  [%rd1], %f1;

// Per-thread private memory. Actually stored in global memory but cached.

// Registers:

// mov.f32  %f1, %f2;

// Fastest memory, most limited. Each thread has its own registers.





// Cache Control and Memory Modifiers
// PTX allows control over caching behavior:

// ld.global.ca.f32  %f1, [%rd1];   // Cache at all levels
// ld.global.cg.f32  %f1, [%rd1];   // Cache globally
// ld.global.cs.f32  %f1, [%rd1];   // Cache streaming
// ld.global.cv.f32  %f1, [%rd1];   // Don't cache

// These modifiers help optimize memory access patterns for different use cases. 
// For example, .cs is good for data that won't be reused, 
// while .ca is good for data with temporal locality.



// Warp-Level Operations
// PTX exposes warp-level primitives for efficient thread cooperation:

// Warp Shuffle
// shfl.sync.bfly.b32  %r1|%p1, %r2, %r3, %r4, %r5;

// Shuffle allows threads in a warp to exchange data without using shared memory. This is incredibly fast and useful for reductions and other collective operations.

// In CUDA C++, you'd use:

// __shfl_sync(unsigned mask, int var, int srcLane);

// But in PTX, you have more fine-grained control over the shuffle patterns (butterfly, up, down, indexed).

// Warp Vote
// vote.sync.all.pred  %p1, %p2, %r1;

// Vote instructions test conditions across all threads in a warp. Useful for early exit conditions and divergence detection.



// Performance Considerations
// When to Use Inline PTX
// Only use inline PTX when:

// Profiling Shows a Hotspot: Don't optimize prematurely. Profile first, identify bottlenecks, then consider PTX.

// Compiler Misses Optimization: Sometimes the compiler doesn't recognize patterns you can hand-optimize.

// Need Special Instructions: Some hardware features are only accessible via PTX.

// Require Specific Instruction Ordering: Control exact execution order for numerical accuracy or memory ordering.

// When Not to Use Inline PTX
// Avoid inline PTX when:

// Compiler Does It Better: Modern CUDA compilers are excellent. Test first before hand-coding.

// Portability Matters: PTX can vary between architectures. High-level CUDA is more portable.

// Maintainability Concerns: Assembly is harder to read and maintain than high-level code.

// Rapid Development: Write correct code first, optimize later if needed.






// Architecture-Specific Optimizations
// PTX can target specific GPU architectures using .target directives:

// .target sm_75              // Turing architecture
// .target sm_80              // Ampere architecture
// .target sm_90              // Hopper architecture

// Different architectures support different instructions. For example, Tensor Core operations are only available on sm_70+, and FP8 support requires sm_89+.

// Check your GPU's compute capability:

// nvidia-smi --query-gpu=compute_cap --format=csv

// Then target that architecture in your Makefile:

// NVCC_FLAGS = -arch=sm_89


//METHOD 3 Using Function Pointers with Runtime linking

__device__ void (*d_vector_add_func)(int*, int*, int*, int) = device_vector_add_func;

//kernel that calls ptx funcitons directly 

__global__ void kernelCallingPTX(int* a, int* b, int* result, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        //MEthod 1 : Direct Call to inline PTX

        result[idx] = addTwoNumbers(a[idx], b[idx]);

        //Method 2 : call external ptx if avialble
        //  vector_add_ptx_device(a, b, result, n);
    }
}


// Kernel using dynamic function pointer (advanced technique)
__global__ void kernelWithFunctionPointer(int* a, int* b, int* result, int n, 
                                         void (*func)(int*, int*, int*, int)) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n && func != nullptr) {
        // Call the provided function pointer
        func(a, b, result, n);
    }
}



// METHOD 4 : Co-Operative Groups Approach for dynamic execution

#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__global__ void kernelWithCooperativeGroups(int* a, int* b, int* result, int n){
    auto grid = cg::this_grid();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < n){
        // Each thread can call pTX functions independently

        result[idx] = addTwoNumbers(a[idx], b[idx]);
    }

    grid.sync();
}

// Kernel that demonstrates function pointer usage

__global__ void kernelWithInternalFunctionPointer(int* a, int* b, int* result, int n){
    // set up function pointer on device
    void (*func)(int* , int* , int* , int) = device_vector_add_func;

    if(func != nullptr){
        func(a, b, result, n);
    }
}


bool verifyVectorAdd(const int* a, const int* b, const int* result, int n){

    for(int i = 0; i < n; i++){
        int expected = a[i] + b[i];

        if(result[i] != expected){
            printf("verification failed at element %d : expected %d : got %d \n",  i, expected, result[i]);
            return false;
        }
    }

    printf("Verification passed for all %d elements\n\n", n);
    return true;

}

#include <iterator>
#include <cstdlib>
// Traditional PTX loading approach (host-side)
std::string readPTXFile(const char* filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        printf("Error: Could not open PTX file: %s\n", filename);
        return "";
    }
    
    std::string content((std::istreambuf_iterator<char>(file)),
                        std::istreambuf_iterator<char>());
    file.close();
    return content;
}

// Host function to demonstrate different approaches

bool demonstrateDevicePTXCalls(bool cooperative_launch_supported)
{
    const int n = 1000;
    int *h_a, *h_b, *h_result;
    int *d_a, *d_b, *d_result;

    size_t size = n * sizeof(int);

    //Allocate Host memory
        // Allocate host memory
    h_a = (int*)malloc(size);
    h_b = (int*)malloc(size);
    h_result = (int*)malloc(size);
    
    // Initialize data
    for (int i = 0; i < n; i++) {
        h_a[i] = i;
        h_b[i] = i + 1;
    }
    
    // Allocate device memory
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_result, size);

    
    // Copy data to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
    
    printf("=== Demonstrating PTX calls from GPU kernels ===\n\n");

    //
    // Method 1: Direct PTX calls in kernel
    //

    
    printf("1. Kernel with inline PTX assembly:\n");
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    cudaMemset(d_result, 0, size);
    kernelCallingPTX<<<numBlocks, blockSize>>>(d_a, d_b, d_result, n);
    cudaDeviceSynchronize();
    
    // Copy result back and verify
    cudaMemcpy(h_result, d_result, size, cudaMemcpyDeviceToHost);
    printf("First 5 results: ");
    for (int i = 0; i < 5; i++) {
        printf("%d+%d=%d ", h_a[i], h_b[i], h_result[i]);
    }
    printf("\n");
    bool all_results_valid = verifyVectorAdd(h_a, h_b, h_result, n);


    // Method 2: Cooperative Groups
    printf("2. Kernel with Cooperative Groups:\n");
    void* kernelArgs[] = {(void*)&d_a, (void*)&d_b, (void*)&d_result, (void*)&n};

    if (!cooperative_launch_supported) {
        printf("Skipped: this device does not support cooperative launch\n\n");
    } 
    else {
        cudaMemset(d_result, 0, size);
        cudaError_t launch_status = cudaLaunchCooperativeKernel(
            (void*)kernelWithCooperativeGroups,
            numBlocks, blockSize, kernelArgs, 0, 0);
        if (launch_status != cudaSuccess) {
            fprintf(stderr, "Cooperative kernel launch failed: %s\n\n",
                    cudaGetErrorString(launch_status));
            all_results_valid = false;
        } else {
            cudaError_t sync_status = cudaDeviceSynchronize();
            if (sync_status != cudaSuccess) {
                fprintf(stderr, "Cooperative kernel execution failed: %s\n\n",
                        cudaGetErrorString(sync_status));
                all_results_valid = false;
            } else {
                cudaMemcpy(h_result, d_result, size, cudaMemcpyDeviceToHost);
                printf("First 5 results: ");
                for (int i = 0; i < 5; i++) {
                    printf("%d+%d=%d ", h_a[i], h_b[i], h_result[i]);
                }
                printf("\n");
                all_results_valid &= verifyVectorAdd(h_a, h_b, h_result, n);
            }
        }
    }
    
    // Method 3: Function Pointer Approach
    printf("3. Kernel with function pointer:\n");
    cudaMemset(d_result, 0, size);
    kernelWithInternalFunctionPointer<<<numBlocks, blockSize>>>(d_a, d_b, d_result, n);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_result, d_result, size, cudaMemcpyDeviceToHost);
    printf("First 5 results: ");
    for (int i = 0; i < 5; i++) {
        printf("%d+%d=%d ", h_a[i], h_b[i], h_result[i]);
    }
    printf("\n");
    all_results_valid &= verifyVectorAdd(h_a, h_b, h_result, n);
    
    // Cleanup
    free(h_a);
    free(h_b);
    free(h_result);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);

    return all_results_valid;
}

bool demonstrateHostPTXLoading() {
    printf("=== Traditional Host-side PTX Loading ===\n");
    
    // Initialize CUDA Driver API
    CUresult result = cuInit(0);
    if (result != CUDA_SUCCESS) {
        printf("Error: Failed to initialize CUDA Driver API\n");
        return false;
    }
    
    // Get CUDA device and create context
    CUdevice device;
    CUcontext context;
    result = cuDeviceGet(&device, 0);
    if (result != CUDA_SUCCESS) {
        printf("Error: Failed to get CUDA device\n");
        return false;
    }

    result = cuCtxCreate(&context, 0, device);
    if (result != CUDA_SUCCESS) {
        printf("Error: Failed to create CUDA context\n");
        return false;
    }
    
    // Load PTX module
    std::string ptxSource = readPTXFile("vector_add.ptx");
    if (ptxSource.empty()) {
        printf("Failed to read PTX file\n");
        cuCtxDestroy(context);
        return false;
    }
    
    CUmodule module;
    result = cuModuleLoadData(&module, ptxSource.c_str());
    if (result != CUDA_SUCCESS) {
        printf("Error: Failed to load PTX module\n");
        cuCtxDestroy(context);
        return false;
    }
    
    // Get function and execute
    CUfunction function;
    result = cuModuleGetFunction(&function, module, "vector_add_ptx");
    if (result != CUDA_SUCCESS) {
        printf("Error: Failed to find vector_add_ptx in PTX module\n");
        cuModuleUnload(module);
        cuCtxDestroy(context);
        return false;
    }
    
    const int n = 1000;
    size_t size = n * sizeof(int);
    
    // Allocate and setup data
    CUdeviceptr d_a, d_b, d_result;
    int *h_a = (int*)malloc(size);
    int *h_b = (int*)malloc(size);
    int *h_result = (int*)malloc(size);
    
    for (int i = 0; i < n; i++) {
        h_a[i] = i;
        h_b[i] = i + 1;
    }
    
    cuMemAlloc(&d_a, size);
    cuMemAlloc(&d_b, size);
    cuMemAlloc(&d_result, size);
    
    cuMemcpyHtoD(d_a, h_a, size);
    cuMemcpyHtoD(d_b, h_b, size);

    cuMemsetD8(d_result, 0, size);

    // Launch PTX kernel from host
    void* args[] = { (void*)&d_a, (void*)&d_b, (void*)&d_result, (void*)&n };
    cuLaunchKernel(function, (n + 255) / 256, 1, 1, 256, 1, 1, 0, nullptr, args, nullptr);
    cuCtxSynchronize();
    
    cuMemcpyDtoH(h_result, d_result, size);
    
    printf("Host-launched PTX kernel results: ");
    for (int i = 0; i < 5; i++) {
        printf("%d+%d=%d ", h_a[i], h_b[i], h_result[i]);
    }
    printf("\n");
    bool results_valid = verifyVectorAdd(h_a, h_b, h_result, n);
    
    // Cleanup
    free(h_a);
    free(h_b);
    free(h_result);
    cuMemFree(d_a);
    cuMemFree(d_b);
    cuMemFree(d_result);
    cuModuleUnload(module);
    cuCtxDestroy(context);

    return results_valid;
}


int main() {
    printf("=== PTX Function Calls from GPU Kernels Demo ===\n\n");
    
    // Check CUDA device capabilities
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    printf("Device: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Supports Cooperative Launch: %s\n", 
           prop.cooperativeLaunch ? "Yes" : "No");
    printf("Supports Multi-Device Cooperative Launch: %s\n", 
           prop.cooperativeMultiDeviceLaunch ? "Yes" : "No");
    printf("\n");
    
    // Demonstrate different approaches
    bool device_calls_valid = demonstrateDevicePTXCalls(prop.cooperativeLaunch);
    bool host_ptx_valid = demonstrateHostPTXLoading();

    if (!device_calls_valid || !host_ptx_valid) {
        fprintf(stderr, "One or more PTX demonstrations failed validation\n");
        return 1;
    }
    
    printf("=== Summary ===\n");
    printf("Methods for calling PTX from GPU kernels:\n");
    printf("1. Inline PTX assembly within kernels (most common)\n");
    printf("2. Linked PTX functions at compile time\n");
    printf("3. Cooperative Groups for synchronized execution\n");
    printf("4. Dynamic Parallelism for recursive kernel launches\n");
    printf("5. Host-side PTX loading with Driver API (traditional)\n");
    
    return 0;
} 






// Case Study 1: Custom Atomic Operations
// Sometimes you need atomic operations not provided by CUDA:

// __device__ float atomicMinFloat(float* addr, float value) {
//     int* addr_as_int = (int*)addr;
//     int old = *addr_as_int, assumed;
//     do {
//         assumed = old;
//         old = atomicCAS(addr_as_int, assumed,
//                        __float_as_int(fminf(value, __int_as_float(assumed))));
//     } while (assumed != old);
//     return __int_as_float(old);
// }

// This requires PTX-level understanding of atomic operations and type punning.

// Case Study 2: Warp-Level Reduction
// Using warp shuffle for fast reduction:

// __device__ float warpReduceSum(float val) {
//     for (int offset = 16; offset > 0; offset /= 2) {
//         val += __shfl_down_sync(0xffffffff, val, offset);
//     }
//     return val;
// }

// Understanding the PTX shuffle instructions helps you use these primitives effectively.

// Case Study 3: Memory Fence Control
// Precise control over memory ordering:

// __device__ void writeWithFence(int* ptr, int value) {
//     *ptr = value;
//     asm volatile("membar.gl;" ::: "memory");
// }

// The membar.gl instruction ensures memory operations complete before continuing, critical for concurrent access patterns.

