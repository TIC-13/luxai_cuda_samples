#include <cuda_runtime.h>
#include <stdio.h>

// CPU addition
void add(float* A_h, float* B_h, float* C_h, int n) {
    for (int i = 0; i < n; i++) C_h[i] = A_h[i] + B_h[i];
}

// GPU addition
__global__ void addKernel(float* A, float* B, float* C, int n) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < n) C[i] = A[i] + B[i];
}

void vecAdd(float* A, float* B, float* C, int n) {
    // Dados do programa
    int size = n * sizeof(float);
    float *d_A, *d_B, *d_C;

    //  01 - Requisita memoria ao device para executar
    // bloco de código com os valores A,B,C
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    // 02 - Chama bloco que executará operações no device
    addKernel <<<ceil(n/256.0), 256>>>(d_A, d_B, d_C, n);

    // 03 - Copia resultado do device para o host
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    // Libera a memoria alocada
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

}