#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

#define TILE_WIDTH 16
#define BLOCK_X 16
#define BLOCK_Y 16

void matrixMultiplicationCPU(int *a, int *b, int *c, int n) {
    for (int row = 0; row < n; ++row) {
        for (int col = 0; col < n; ++col) {
            int sum = 0;
            for (int i = 0; i < n; ++i) {
                sum += a[row * n + i] * b[i * n + col];
            }
            c[row * n + col] = sum;
        }
    }
}

__global__ void matrixMultiplication_x(int *a, int *b, int *c, int n) {
   
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n) {
        for (int col = 0; col < n; ++col) {
            int sum = 0;
            for (int i = 0; i < n; ++i) {
                sum += a[row * n + i] * b[i * n + col];
            }
            c[row * n + col] = sum;
        }
    }
}

__global__ void matrixMultiplication_yx(int* a, int* b, int*c, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if ((row < n) && (col < n)) {
        int result = 0;
        for(int k=0; k < n; k++){
            result += a[row*n+k]*b[k*n+col];
        }
        c[row*n+col] = result;
    }
}

__global__ void matrixMultiplication_tiled(int* a, int* b, int* c, int n) {
    __shared__ int a_s[TILE_WIDTH][TILE_WIDTH];
    __shared__ int b_s[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;  int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    int result = 0;
    for (int t=0; t < n/TILE_WIDTH; ++t){
        a_s[ty][tx] = a[row*n + t*TILE_WIDTH + tx];
        b_s[ty][tx] = b[(t*TILE_WIDTH + ty)*n + col];
        __syncthreads();

        for(int k=0; k < TILE_WIDTH; ++k){
            result += a_s[ty][k] + b_s[k][tx];
        }
        __syncthreads();
    }

    c[row*n+col] = result;
}

int main() {

    int N = 1024;
    
    int *h_a, *h_b, *h_c; // Matrizes na CPU (host)
    int *d_a, *d_b, *d_c; // Matrizes no dispositivo (device)

    h_a = new int[N * N]; 
    h_b = new int[N * N]; 
    h_c = new int[N * N];

    for (int i = 0; i < N * N; ++i) {
        h_a[i] = i;
        h_b[i] = (i * 2) + 3;
    }

    cudaMalloc(&d_a, N * N * sizeof(int));
    cudaMalloc(&d_b, N * N * sizeof(int));
    cudaMalloc(&d_c, N * N * sizeof(int));

    // Kernel 01
    dim3 blockSize(N); 
    dim3 numBlocks(N);

    auto exec_and_copy_start = std::chrono::high_resolution_clock::now();
    cudaMemcpy(d_a, h_a, N * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * N * sizeof(int), cudaMemcpyHostToDevice);

    auto start = std::chrono::high_resolution_clock::now();
    matrixMultiplication_x<<<numBlocks, blockSize>>>(d_a, d_b, d_c, N);
    auto end = std::chrono::high_resolution_clock::now();

    cudaMemcpy(h_c, d_c, N * N * sizeof(int), cudaMemcpyDeviceToHost);
    auto exec_and_copy_end = std::chrono::high_resolution_clock::now();
   

    std::chrono::duration<double> elapsed_total = exec_and_copy_end - exec_and_copy_start;
    std::chrono::duration<double> elapsed = end - start;

    std::cout << "Kernel matrixMultiplication_x" << std::endl;
    std::cout << "Tempo decorrido pelo processamento feito em cuda: " << elapsed.count() << " segundos" << std::endl;
    std::cout << "Tempo decorrido de processamento e transferencia: " << elapsed_total.count() << " segundos" << std::endl;
    // End kernel 01
    
    // Kernel 02
    dim3 blockSize2(BLOCK_X, BLOCK_Y); 
    dim3 numBlocks2((N-1)/BLOCK_X + 1, (N-1)/BLOCK_Y + 1);

    exec_and_copy_start = std::chrono::high_resolution_clock::now();
    cudaMemcpy(d_a, h_a, N * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * N * sizeof(int), cudaMemcpyHostToDevice);

    start = std::chrono::high_resolution_clock::now();
    matrixMultiplication_yx<<<numBlocks2, blockSize2>>>(d_a, d_b, d_c, N);
    end = std::chrono::high_resolution_clock::now();

    cudaMemcpy(h_c, d_c, N * N * sizeof(int), cudaMemcpyDeviceToHost);
    exec_and_copy_end = std::chrono::high_resolution_clock::now();
   

    elapsed_total = exec_and_copy_end - exec_and_copy_start;
    elapsed = end - start;

    std::cout << "Kernel matrixMultiplication_yx" << std::endl;
    std::cout << "Tempo decorrido pelo processamento feito em cuda: " << elapsed.count() << " segundos" << std::endl;
    std::cout << "Tempo decorrido de processamento e transferencia: " << elapsed_total.count() << " segundos" << std::endl;
    // End kernel 02

    // Kernel 03
    dim3 blockSize3(BLOCK_X, BLOCK_Y); 
    dim3 numBlocks3((N-1)/BLOCK_X + 1, (N-1)/BLOCK_Y + 1);

    exec_and_copy_start = std::chrono::high_resolution_clock::now();
    cudaMemcpy(d_a, h_a, N * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * N * sizeof(int), cudaMemcpyHostToDevice);

    start = std::chrono::high_resolution_clock::now();
    matrixMultiplication_tiled<<<numBlocks3, blockSize3>>>(d_a, d_b, d_c, N);
    end = std::chrono::high_resolution_clock::now();

    cudaMemcpy(h_c, d_c, N * N * sizeof(int), cudaMemcpyDeviceToHost);
    exec_and_copy_end = std::chrono::high_resolution_clock::now();
   

    elapsed_total = exec_and_copy_end - exec_and_copy_start;
    elapsed = end - start;

    std::cout << "Kernel matrixMultiplication_tiled" << std::endl;
    std::cout << "Tempo decorrido pelo processamento feito em cuda: " << elapsed.count() << " segundos" << std::endl;
    std::cout << "Tempo decorrido de processamento e transferencia: " << elapsed_total.count() << " segundos" << std::endl;
    // End kernel 03

    // CPU
    auto start_cpu = std::chrono::high_resolution_clock::now();
    matrixMultiplicationCPU(h_a, h_b, h_c, N);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_cpu = end_cpu - start_cpu;
    std::cout << "Host" << std::endl;
    std::cout << "Tempo decorrido pelo processamento feito em CPU: " << elapsed_cpu.count() << " segundos" << std::endl;
    // END CPU

    
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}