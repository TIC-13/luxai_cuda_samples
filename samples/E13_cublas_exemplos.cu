#include <iostream>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <cublas_v2.h>

int main() {
    // Definir o tamanho do vetor
    const int N = 10;
    float h_x[N] = {322.0f, 6.0f, 23.0f, 576.0f, 2.0f, 5.0f, 90.0f, 1.0f, 34.0f, 17.0f}; // Definindo como floats

    // Inicializar o cublas
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Alocar memória para o vetor no dispositivo
    float *d_x;
    cudaMalloc((void **)&d_x, N * sizeof(float));

    // Copiar o vetor para o dispositivo
    cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice);

    // Calcular a média usando cublasSasum
    float mean;
    cublasSasum(handle, N, d_x, 1, &mean);
    mean /= N;

    // Calcular a soma dos elementos
    float sum;
    cublasSasum(handle, N, d_x, 1, &sum);

    // Calcular a soma dos quadrados usando cublasSdot
    float sum_of_squares;
    cublasSdot(handle, N, d_x, 1, d_x, 1, &sum_of_squares);

    // Calcular o produto escalar (dot product)
    float dot_product;
    cublasSdot(handle, N, d_x, 1, d_x, 1, &dot_product);

    // Imprimir os resultados
    std::cout << "Média: " << mean << std::endl;
    std::cout << "Soma dos elementos: " << sum << std::endl;
    std::cout << "Soma dos quadrados: " << sum_of_squares << std::endl;
    std::cout << "Produto escalar (dot product): " << dot_product << std::endl;

    // Liberar a memória alocada
    cudaFree(d_x);
    cublasDestroy(handle);

    return 0;
}
