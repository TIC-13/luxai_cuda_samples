#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

#define IN_TILE_DIM_1  8
#define OUT_TILE_DIM_1 6
#define BLOCK_DIM_1 8

#define IN_TILE_DIM_2  32
#define OUT_TILE_DIM_2 30
#define BLOCK_DIM_2 32

#define c0 1.0f
#define c1 0.1f
#define c2 0.1f
#define c3 0.1f
#define c4 0.1f
#define c5 0.1f
#define c6 0.1f

__global__ void stencil3d_basic(float* in, float* out, unsigned int N) {
    
    int i = blockIdx.z*blockDim.z + threadIdx.z;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    int k = blockIdx.x*blockDim.x + threadIdx.x;

    if(i >= 1 && i < N - 1 && j >= 1 && j < N - 1 && k >=1 && k < N - 1){
        out[i*N*N + j * N + k] = c0*in[i*N*N + j * N + k] +
                                 c1*in[i*N*N + j * N + (k-1)] +
                                 c2*in[i*N*N + j * N + (k+1)] +
                                 c3*in[i*N*N + (j-1) * N + k] +
                                 c4*in[i*N*N + (j+1) * N + k] +
                                 c5*in[(i-1)*N*N + j * N + k] +
                                 c6*in[(i+1)*N*N + j * N + k];
    }
}

__global__ void stencil3d_tiled(float* in, float* out, unsigned int N) {
    int i = blockIdx.z*OUT_TILE_DIM_1 + threadIdx.z - 1;
    int j = blockIdx.y*OUT_TILE_DIM_1 + threadIdx.y - 1;
    int k = blockIdx.x*OUT_TILE_DIM_1 + threadIdx.x - 1;

    __shared__ float in_s[IN_TILE_DIM_1][IN_TILE_DIM_1][IN_TILE_DIM_1];
    if (i >= 0 && i < N && j >= 0 && j < N && k >= 0 && k < N){
        in_s[threadIdx.z][threadIdx.y][threadIdx.x] = in[i*N*N + j*N + k];
    }

    __syncthreads();

    if(i >= 1 && i < N - 1 && j >= 1 && j < N - 1 && k >=1 && k < N - 1){
        if (threadIdx.z >= 1 && threadIdx.z < IN_TILE_DIM_1-1 && threadIdx.y >=1
            && threadIdx.y < IN_TILE_DIM_1-1 && threadIdx.x >= 1 && threadIdx.x < IN_TILE_DIM_1-1){
            out[i*N*N + j * N + k] = c0*in_s[threadIdx.z][threadIdx.y][threadIdx.x] +
                                    c1*in_s[threadIdx.z][threadIdx.y][threadIdx.x-1] +
                                    c2*in_s[threadIdx.z][threadIdx.y][threadIdx.x+1] +
                                    c3*in_s[threadIdx.z][threadIdx.y-1][threadIdx.x] +
                                    c4*in_s[threadIdx.z][threadIdx.y+1][threadIdx.x] +
                                    c5*in_s[threadIdx.z-1][threadIdx.y][threadIdx.x] +
                                    c6*in_s[threadIdx.z+1][threadIdx.y][threadIdx.x];
        }
    }
}

__global__ void stencil3d_tiled_coarse(float* in, float* out, unsigned int N) {
    int iStart = blockIdx.z*OUT_TILE_DIM_2;

    int j = blockIdx.y*OUT_TILE_DIM_2 + threadIdx.y - 1;
    int k = blockIdx.x*OUT_TILE_DIM_2 + threadIdx.x - 1;

    __shared__ float inPrev_s[IN_TILE_DIM_2][IN_TILE_DIM_2];
    __shared__ float inCurr_s[IN_TILE_DIM_2][IN_TILE_DIM_2];
    __shared__ float inNext_s[IN_TILE_DIM_2][IN_TILE_DIM_2];

    if (iStart-1 >= 0 && iStart-1 < N && j >= 0 && j < N && k >= 0 && k < N){
        inPrev_s[threadIdx.y][threadIdx.x] = in[(iStart-1)*N*N + j*N + k];
    }

    if (iStart >= 0 && iStart < N && j >= 0 && j < N && k >= 0 && k < N){
        inCurr_s[threadIdx.y][threadIdx.x] = in[iStart*N*N + j*N + k];
    }

    for (int i = iStart; i < iStart + OUT_TILE_DIM_2; ++i){
        if (i + 1 >= 0 && i + 1 < N && j >= 0 && j < N && k >= 0 && k < N){
            inNext_s[threadIdx.y][threadIdx.x] = in[(i+1)*N*N + j*N + k];
        }
        __syncthreads();

        if(i >= 1 && i < N - 1 && j >= 1 && j < N - 1 && k >=1 && k < N - 1){
            if (threadIdx.y >=1 && threadIdx.y < IN_TILE_DIM_2-1 
                && threadIdx.x >= 1 && threadIdx.x < IN_TILE_DIM_2-1){
                out[i*N*N + j * N + k] = c0*inCurr_s[threadIdx.y][threadIdx.x] +
                                         c1*inCurr_s[threadIdx.y][threadIdx.x-1] +
                                         c2*inCurr_s[threadIdx.y][threadIdx.x+1] +
                                         c3*inCurr_s[threadIdx.y-1][threadIdx.x] +
                                         c4*inCurr_s[threadIdx.y+1][threadIdx.x] +
                                         c5*inPrev_s[threadIdx.y][threadIdx.x] +
                                         c6*inNext_s[threadIdx.y][threadIdx.x];
            }
        }
        __syncthreads();
        inPrev_s[threadIdx.y][threadIdx.x] = inCurr_s[threadIdx.y][threadIdx.x];
        inCurr_s[threadIdx.y][threadIdx.x] = inNext_s[threadIdx.y][threadIdx.x];
    }
}

__global__ void stencil3d_regtiled(float* in, float* out, unsigned int N) {
    int iStart = blockIdx.z*OUT_TILE_DIM_2;

    int j = blockIdx.y*OUT_TILE_DIM_2 + threadIdx.y - 1;
    int k = blockIdx.x*OUT_TILE_DIM_2 + threadIdx.x - 1;

    __shared__ float inCurr_s[IN_TILE_DIM_2][IN_TILE_DIM_2];
    float inPrev;
    float inCurr;
    float inNext;

    if (iStart-1 >= 0 && iStart-1 < N && j >= 0 && j < N && k >= 0 && k < N){
        inPrev = in[(iStart-1)*N*N + j*N + k];
    }

    if (iStart >= 0 && iStart < N && j >= 0 && j < N && k >= 0 && k < N){
        inCurr = in[iStart*N*N + j*N + k];
        inCurr_s[threadIdx.y][threadIdx.x] = inCurr;
    }

    for (int i = iStart; i < iStart + OUT_TILE_DIM_2; ++i){
        if (i + 1 >= 0 && i + 1 < N && j >= 0 && j < N && k >= 0 && k < N){
            inNext = in[(i+1)*N*N + j*N + k];
        }
        __syncthreads();

        if(i >= 1 && i < N - 1 && j >= 1 && j < N - 1 && k >=1 && k < N - 1){
            if (threadIdx.y >=1 && threadIdx.y < IN_TILE_DIM_2-1 
                && threadIdx.x >= 1 && threadIdx.x < IN_TILE_DIM_2-1){
                out[i*N*N + j * N + k] = c0*inCurr +
                                         c1*inCurr_s[threadIdx.y][threadIdx.x-1] +
                                         c2*inCurr_s[threadIdx.y][threadIdx.x+1] +
                                         c3*inCurr_s[threadIdx.y-1][threadIdx.x] +
                                         c4*inCurr_s[threadIdx.y+1][threadIdx.x] +
                                         c5*inPrev +
                                         c6*inNext;
            }
        }
        __syncthreads();
        inPrev = inCurr;
        inCurr = inNext;
        inCurr_s[threadIdx.y][threadIdx.x] = inNext;
    }
}

int main() {

    int N = 1024;
    
    float *h_a, *h_b; // Matrizes na CPU (host)
    float *d_a, *d_b; // Matrizes no dispositivo (device)

    h_a = new float[N * N * N]; 
    h_b = new float[N * N * N]; 

    for (int i = 0; i < N * N * N; ++i) h_a[i] = i;
    
    cudaMalloc(&d_a, N * N * N * sizeof(float));
    cudaMalloc(&d_b, N * N * N * sizeof(float));

    dim3 blockSize3d(BLOCK_DIM_1, BLOCK_DIM_1, BLOCK_DIM_1); 
    dim3 gridSize3d((N-1)/BLOCK_DIM_1 + 1, (N-1)/BLOCK_DIM_1 + 1, (N-1)/BLOCK_DIM_1+1);

    dim3 blockSize2d(BLOCK_DIM_2, BLOCK_DIM_2); 
    dim3 gridSize2d((N-1)/BLOCK_DIM_2 + 1, (N-1)/BLOCK_DIM_2 + 1);

    // Start Stencil Basic
    auto exec_and_copy_start = std::chrono::high_resolution_clock::now();
    cudaMemcpy(d_a, h_a, N * N * N * sizeof(float), cudaMemcpyHostToDevice);

    auto start = std::chrono::high_resolution_clock::now();
    stencil3d_basic<<<gridSize3d, blockSize3d>>>(d_a, d_b, N);
    auto end = std::chrono::high_resolution_clock::now();

    cudaMemcpy(h_b, d_b, N * N * N * sizeof(int), cudaMemcpyDeviceToHost);
    auto exec_and_copy_end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed_total = exec_and_copy_end - exec_and_copy_start;
    std::chrono::duration<double> elapsed = end - start;

    std::cout << "Stencil Basic" << std::endl;
    std::cout << "Tempo decorrido pelo processamento feito em cuda: " << elapsed.count() << " segundos" << std::endl;
    std::cout << "Tempo decorrido de processamento e transferencia: " << elapsed_total.count() << " segundos" << std::endl;
    // END Stencil Basic

    // Start Stencil tiled
    exec_and_copy_start = std::chrono::high_resolution_clock::now();
    cudaMemcpy(d_a, h_a, N * N * N * sizeof(float), cudaMemcpyHostToDevice);

    start = std::chrono::high_resolution_clock::now();
    stencil3d_tiled<<<gridSize3d, blockSize3d>>>(d_a, d_b, N);
    end = std::chrono::high_resolution_clock::now();

    cudaMemcpy(h_b, d_b, N * N * N * sizeof(int), cudaMemcpyDeviceToHost);
    exec_and_copy_end = std::chrono::high_resolution_clock::now();
   
    elapsed_total = exec_and_copy_end - exec_and_copy_start;
    elapsed = end - start;

    std::cout << "Stencil Tiled" << std::endl;
    std::cout << "Tempo decorrido pelo processamento feito em cuda: " << elapsed.count() << " segundos" << std::endl;
    std::cout << "Tempo decorrido de processamento e transferencia: " << elapsed_total.count() << " segundos" << std::endl;
    // End Stencil tiled

    // Start Stencil Coarsed
    exec_and_copy_start = std::chrono::high_resolution_clock::now();
    cudaMemcpy(d_a, h_a, N * N * N * sizeof(float), cudaMemcpyHostToDevice);

    start = std::chrono::high_resolution_clock::now();
    stencil3d_tiled_coarse<<<gridSize2d, blockSize2d>>>(d_a, d_b, N);
    end = std::chrono::high_resolution_clock::now();

    cudaMemcpy(h_b, d_b, N * N * N * sizeof(int), cudaMemcpyDeviceToHost);
    exec_and_copy_end = std::chrono::high_resolution_clock::now();

    elapsed_total = exec_and_copy_end - exec_and_copy_start;
    elapsed = end - start;

    std::cout << "Stencil coarsed" << std::endl;
    std::cout << "Tempo decorrido pelo processamento feito em cuda: " << elapsed.count() << " segundos" << std::endl;
    std::cout << "Tempo decorrido de processamento e transferencia: " << elapsed_total.count() << " segundos" << std::endl;
    // END Stencil Coarsed

    // Start Stencil Coarsed_reg
    exec_and_copy_start = std::chrono::high_resolution_clock::now();
    cudaMemcpy(d_a, h_a, N * N * N * sizeof(float), cudaMemcpyHostToDevice);

    start = std::chrono::high_resolution_clock::now();
    stencil3d_regtiled<<<gridSize2d, blockSize2d>>>(d_a, d_b, N);
    end = std::chrono::high_resolution_clock::now();

    cudaMemcpy(h_b, d_b, N * N * N * sizeof(int), cudaMemcpyDeviceToHost);
    exec_and_copy_end = std::chrono::high_resolution_clock::now();

    elapsed_total = exec_and_copy_end - exec_and_copy_start;
    elapsed = end - start;

    std::cout << "Stencil coarsed_register" << std::endl;
    std::cout << "Tempo decorrido pelo processamento feito em cuda: " << elapsed.count() << " segundos" << std::endl;
    std::cout << "Tempo decorrido de processamento e transferencia: " << elapsed_total.count() << " segundos" << std::endl;
    // END Stencil Coarsed_reg

    delete[] h_a;
    delete[] h_b;
    
    cudaFree(d_a);
    cudaFree(d_b);

    return 0;
}