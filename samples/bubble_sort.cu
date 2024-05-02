#include <iostream>
#include <cstdlib>

// Definindo um mutex e suas funções de lock e unlock
__device__ int mutex = 0;

__device__ void lock(int *mutex) {
    while (atomicCAS(mutex, 0, 1) != 0);
}

__device__ void unlock(int *mutex) {
    atomicExch(mutex, 0);
}

// Função para trocar dois elementos em um vetor
__device__ void swap(int* array, int i, int j) {
    int temp = array[i];  // Armazena o valor do elemento na posição i
    array[i] = array[j];  // Substitui o valor na posição i pelo valor na posição j
    array[j] = temp;      // Substitui o valor na posição j pelo valor temporário (originalmente na posição i)
    printf("Swap realizado pela thread %d: %d <-> %d\n", threadIdx.x + blockIdx.x * blockDim.x, array[j], array[i]);
}

// Kernel para ordenação de bolhas
__global__ void bubbleSort(int* array, int n) {
    // Calcula o índice global da thread
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    // Verifica se o índice global está dentro dos limites do vetor
    if (index < n) {
        // Loop externo para percorrer todos os elementos do vetor, exceto o último
        for (int i = 0; i < n - 1; ++i) {
            // Loop interno para percorrer todos os elementos do vetor não ordenados
            for (int j = 0; j < n - i - 1; ++j) {
                // Verifica se o elemento atual é maior que o próximo elemento
                lock(&mutex);
                if (array[j] > array[j + 1])
                    swap(array, j, j + 1);
                unlock(&mutex);
            }
        }
        // Informa que a thread terminou sua execução
        printf("Thread %d terminou sua execução.\n", threadIdx.x + blockIdx.x * blockDim.x);
    }
}


// Função para imprimir um vetor
void printArray(const int* arr, int N) {
    std::cout << "[";
    for (int i = 0; i < N; i++) {
        if (i < N - 1)
            std::cout << arr[i] << ", ";
        else
            std::cout << arr[i];
    }
    std::cout << "]" << std::endl;
}

int main() {
    const int N = 10;
    const int arraySize = N * sizeof(int);
    int array[N] = {9, 7, 3, 5, 6, 1, 4, 8, 2, 0};

    // Imprime o vetor antes da ordenação
    std::cout << "Vetor antes da ordenação: ";
    printArray(array, N);

    // Declara e aloca memória na GPU
    int* d_array;
    cudaMalloc((void**)&d_array, arraySize);

    // Copia dados da CPU para a GPU
    cudaMemcpy(d_array, array, arraySize, cudaMemcpyHostToDevice);

    // Configura os parâmetros do kernel
    int blockSize = 1;
    int numBlocks = (N/2 + blockSize - 1) / blockSize;

    // Chama o kernel para ordenação
    bubbleSort<<<numBlocks, blockSize>>>(d_array, N);

    // Copia resultados da GPU para a CPU
    cudaMemcpy(array, d_array, arraySize, cudaMemcpyDeviceToHost);

    // Libera memória da GPU
    cudaFree(d_array);

    // Imprime o vetor depois da ordenação
    std::cout << "Vetor depois da ordenação: ";
    printArray(array, N);

    return 0;
}
