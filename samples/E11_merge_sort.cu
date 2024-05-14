#include <iostream>
#include <cuda_runtime.h>

// Número de threads por bloco
#define BLOCK_SIZE 32

// Função para merge dos vetores ordenados
__device__ void merge(int *arr, int l, int m, int r) {
    int n1 = m - l + 1, n2 = r - m;

    // Criação de vetores temporários
    int *L = new int[n1], *R = new int[n2];

    // Copiando os dados para os vetores temporários L[] e R[]
    for (int i = 0; i < n1; i++)
        L[i] = arr[l + i];
    for (int j = 0; j < n2; j++)
        R[j] = arr[m + 1 + j];

    // Merge dos vetores temporários de volta para arr[l..r]
    int i = 0, j = 0, k = l;

    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            arr[k] = L[i];
            i++;
        }
        else {
            arr[k] = R[j];
            j++;
        }
        k++;
    }

    // Copia os elementos restantes de L[], se houver
    while (i < n1) {
        arr[k] = L[i];
        i++;
        k++;
}

    // Copia os elementos restantes de R[], se houver
    while (j < n2) {
        arr[k] = R[j];
        j++;
        k++;
    }

    delete[] L, delete[] R;
}

// Função kernel para merge paralelo
__global__ void mergeParallel(int *arr, int n, int chunkSize) {
    // Calcula o ID global do thread
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    // Calcula o início do intervalo atual
    int l = tid * chunkSize;
    // Calcula o final do intervalo atual, limitado a n - 1
    int r = min((tid + 1) * chunkSize - 1, n - 1);

    // Loop principal do merge sort
    for (int curr_size = 1; curr_size <= r - l; curr_size *= 2) {
        // Loop para percorrer os subvetores
        for (int left_start = l; left_start < r; left_start += 2 * curr_size) {
            // Calcula o meio do subvetor
            int mid = min(left_start + curr_size - 1, r);
            // Calcula o final do subvetor, limitado a r
            int right_end = min(left_start + 2 * curr_size - 1, r);

            // Realiza o merge dos subvetores
            merge(arr, left_start, mid, right_end);
        }
    }

    // Impressão quando a thread termina sua execução
    printf("Thread %d terminou sua execução.\n", tid);
}

int main() {
    const int N = 10;
    const int size = N * sizeof(int);
    int arr[N] = {322, 6, 23, 576, 2, 5, 90, 1, 34, 17};
    int *d_arr;

    // Imprime o vetor antes da ordenação
    std::cout << "Vetor antes da ordenação: [";
    for (int i = 0; i < N; i++) {
        if(i<N-1)
            std::cout << arr[i] << ", ";
        else
            std::cout << arr[i];
    }
    std::cout << "]" << std::endl;

    // Aloca memória para o vetor no device
    cudaMalloc(&d_arr, size);

    // Copia o vetor do host para o device
    cudaMemcpy(d_arr, arr, size, cudaMemcpyHostToDevice);

    // Define o número de blocos e threads por bloco
    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Define o tamanho do chunk para cada thread
    int chunkSize = (N + numBlocks - 1) / numBlocks;

    // Chama a função kernel para merge paralelo
    mergeParallel<<<numBlocks, BLOCK_SIZE>>>(d_arr, N, chunkSize);

    // Copia o vetor ordenado do device para o host
    cudaMemcpy(arr, d_arr, size, cudaMemcpyDeviceToHost);

    // Exibe o vetor ordenado
    std::cout << "Vetor depois da ordenação: [";
    for (int i = 0; i < N; i++) {
        if(i<N-1)
            std::cout << arr[i] << ", ";
        else
            std::cout << arr[i];
    }
    std::cout << "]" << std::endl;

    // Libera memória
    cudaFree(d_arr);

    return 0;
}