#include "../libs/common/common.h"
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void mathKernel1(float* c) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    float a,b;
    a = b = 0.0f;

    if (tid %2 == 0) {
        a = 100.0f;
    }
    else {
        b = 200.0f;
    }
    c[tid] = a + b;

}


__global__ void mathKernel2(float* c) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float a,b;
    a = b = 0.0f;

    if ((tid/warpSize) % 2 == 0) {
        a = 100.0f;
    }
    else { 
        b = 200.0f;
    }
    c[tid] = a + b;
}


__global__ void warmingup(float *c)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float ia, ib;
    ia = ib = 0.0f;

    if ((tid / warpSize) % 2 == 0)
    {
        ia = 100.0f;
    }
    else
    {
        ib = 200.0f;
    }

    c[tid] = ia + ib;
}

int main(int argc, char **argv) {
    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("%s using %d: %s\n", argv[0], dev, deviceProp.name);

    int size = 100000;
    int blocksize = 128;

    if (argc > 1) blocksize = atoi(argv[1]);
    if (argc > 2) blocksize = atoi(argv[2]);

    dim3 block (blocksize, 1);
    dim3 grid  ((size + block.x - 1)/ block.x, 1);
    
    float *d_C;
    size_t nBytes = size * sizeof(float);
    cudaMalloc((float**)&d_C, nBytes);

    size_t istart, ielaps;
    cudaDeviceSynchronize();
    istart = seconds();
    warmingup<<<grid, block>>> (d_C);
    cudaDeviceSynchronize();
    ielaps = seconds() - istart;
    printf("warmup <<< %4d %4d >>> elapsed %d sec\n", grid.x, block.x, (int)ielaps);

    istart = seconds();
    mathKernel1<<<grid, block>>> (d_C);
    cudaDeviceSynchronize();
    ielaps = seconds() - istart;
    printf("mathKernel1 <<< %4d %4d >>> elapsed %d sec\n", grid.x, block.x, (int)ielaps);

    istart = seconds();
    mathKernel2<<<grid, block>>> (d_C);
    cudaDeviceSynchronize();
    ielaps = seconds() - istart;
    printf("mathKernel2 <<< %4d %4d >>> elapsed %d sec\n", grid.x, block.x, (int)ielaps);

    CHECK(cudaFree(d_C));
    CHECK(cudaDeviceReset());
    return EXIT_SUCCESS;
}