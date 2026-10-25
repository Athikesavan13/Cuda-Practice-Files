#include <cuda_runtime.h>

extern "C" __global__
void vector_add_ptx(
    const int* a,
    const int* b,
    int* result,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        int value;
        asm("add.s32 %0, %1, %2;"
            : "=r"(value)
            : "r"(a[idx]), "r"(b[idx]));

        result[idx] = value;
    }
}