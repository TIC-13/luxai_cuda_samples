#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

__global__ void matrixMultiplication(int *a, int *b, int *c, int n) {
    // Calcula o índice da linha da matriz resultante (uma linha por thread)
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    // Verifica se a linha calculada está dentro dos limites da matriz
    if (row < n) {
        // Loop para calcular os elementos da linha da matriz resultante
        for (int col = 0; col < n; ++col) {
            int sum = 0;
            // Loop para calcular o elemento da matriz resultante na posição (row, col)
            for (int i = 0; i < n; ++i) {
                // Multiplica e acumula os elementos correspondentes das linhas de 'a' e colunas de 'b'
                sum += a[row * n + i] * b[i * n + col];
            }
            // Armazena o resultado no índice apropriado da matriz resultante 'c'
            c[row * n + col] = sum;
        }
        // Imprime quando a thread termina de calcular a linha
        printf("Thread[%d] calculou sua linha!\n", row);
    }
}

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


// Função para visualizar as matrizes
void printMatrix(int *matrix, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        std::cout << "[";
        for (int j = 0; j < cols; ++j) {
            std::cout << matrix[i * cols + j];
            if (j < cols - 1) {
                std::cout << ", ";
            }
        }
        std::cout << "]";
        if (i < rows - 1) {
            std::cout << std::endl;
        }
    }
    std::cout << std::endl;
}

int main() {

    int N;
    
    // OBS: teste entrada <10 vs >1000
    std::cout << "Digite o tamanho desejado para as dimensoes: " << std::endl;
    std::cin >> N;

    int *h_a, *h_b, *h_c; // Matrizes na CPU (host)
    int *d_a, *d_b, *d_c; // Matrizes no dispositivo (device)

    // Alocar memória para matrizes na CPU
    h_a = new int[N * N];
    h_b = new int[N * N];
    h_c = new int[N * N];

    // Preencher as matrizes com valores de exemplo
    for (int i = 0; i < N * N; ++i) {
        h_a[i] = i;
        h_b[i] = (i * 2) + 3;
    }

    if(N<=5){
        // Imprimir as matrizes originais
        std::cout << "Matriz A:" << std::endl;
        printMatrix(h_a, N, N);
        std::cout << std::endl;

        std::cout << "Matriz B:" << std::endl;
        printMatrix(h_b, N, N);
        std::cout << std::endl;
    }

    // Alocar memória para matrizes no dispositivo
    cudaMalloc(&d_a, N * N * sizeof(int));
    cudaMalloc(&d_b, N * N * sizeof(int));
    cudaMalloc(&d_c, N * N * sizeof(int));

    // Copiar dados das matrizes da CPU para o dispositivo
    cudaMemcpy(d_a, h_a, N * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * N * sizeof(int), cudaMemcpyHostToDevice);

    // Configurar o número de threads por bloco.
    // Neste caso, estamos configurando 1 thread por bloco para cada linha da matriz.
    dim3 blockSize(N); // N thread por bloco
    // O número de blocos é igual ao número de linhas da matriz.
    dim3 numBlocks(N);

    // Captura o tempo atual antes da execução
    auto start = std::chrono::high_resolution_clock::now();

    // Chamar o kernel para multiplicação de matrizes
    matrixMultiplication<<<numBlocks, blockSize>>>(d_a, d_b, d_c, N);

    // Copiar o resultado de volta para a CPU
    cudaMemcpy(h_c, d_c, N * N * sizeof(int), cudaMemcpyDeviceToHost);

    // Captura o tempo após a execução
    auto end = std::chrono::high_resolution_clock::now();

    // Calcula a duração, ou seja, o tempo decorrido
    std::chrono::duration<double> elapsed = end - start;

    // Converte a duração para segundos e imprime
    std::cout << "Tempo decorrido pelo processamento feito em cuda: " << elapsed.count() << " segundos" << std::endl;

    if(N<=5){
        // Imprimir a matriz resultante
        std::cout << std::endl;
        std::cout << "Matriz Resultante:" << std::endl;
        printMatrix(h_c, N, N);
    }

    // Captura o tempo atual antes da execução
    auto start_cpu = std::chrono::high_resolution_clock::now();

    // Chamar a função de soma em CPU
    matrixMultiplicationCPU(h_a, h_b, h_c, N);

    // Captura o tempo após a execução
    auto end_cpu = std::chrono::high_resolution_clock::now();

    // Calcula a duração, ou seja, o tempo decorrido
    std::chrono::duration<double> elapsed_cpu = end_cpu - start_cpu;

    // Converte a duração para segundos e imprime
    std::cout << "Tempo decorrido pelo processamento feito em CPU: " << elapsed_cpu.count() << " segundos" << std::endl;

    // Liberar memória alocada na gpu e cpu
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}