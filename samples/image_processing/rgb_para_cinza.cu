#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

// Kernel CUDA para converter uma imagem colorida em tons de cinza
__global__ void grayscaleKernel(const uchar* input, uchar* output, int rows, int cols) {
    // Calcula as coordenadas (x, y) do pixel a ser processado
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Verifica se as coordenadas estão dentro dos limites da imagem
    if (x < cols && y < rows) {
        int tid = y * cols + x; // Calcula o índice do pixel no vetor de pixels
        // Converte o pixel RGB para tons de cinza
        output[tid] = (input[tid * 3] + input[tid * 3 + 1] + input[tid * 3 + 2]) / 3;
    }
}

int main() {
    // Carrega a imagem de entrada
    cv::Mat inputImage = cv::imread("imagem.jpg", cv::IMREAD_COLOR);

    // Verifica se a imagem foi carregada corretamente
    if (inputImage.empty()) {
        std::cerr << "Erro ao carregar imagem." << std::endl;
        return 1;
    }

    // Obtém as dimensões da imagem
    int rows = inputImage.rows;
    int cols = inputImage.cols;

    // Aloca memória na GPU para os dados de entrada e saída
    uchar* d_input;
    uchar* d_output;
    cudaMalloc((void**)&d_input, sizeof(uchar) * rows * cols * 3);
    cudaMalloc((void**)&d_output, sizeof(uchar) * rows * cols);

    // Copia os dados da imagem de entrada da CPU para a GPU
    cudaMemcpy(d_input, inputImage.data, sizeof(uchar) * rows * cols * 3, cudaMemcpyHostToDevice);

    // Define as dimensões dos blocos (threads agrupados) e da grade (grid) CUDA
    // Aqui estamos definindo o tamanho dos blocos como 16x16. Isso significa que cada bloco terá 16 threads na dimensão x e 16 threads na dimensão y.
    dim3 blockDim(16, 16);

    // Agora vamos definir as dimensões da grade. A grade é composta por múltiplos blocos.
    // A fórmula usada aqui garante que teremos o número certo de blocos para cobrir todos os elementos da matriz.
    // A expressão (cols + blockDim.x - 1) / blockDim.x calcula o número de blocos na dimensão x, onde cols é o número de colunas na matriz.
    // A expressão (rows + blockDim.y - 1) / blockDim.y calcula o número de blocos na dimensão y, onde rows é o número de linhas na matriz.
    // Adicionamos blockDim.x - 1 e blockDim.y - 1 antes de dividir para garantir que a divisão arredonde para cima quando não for um múltiplo exato.
    dim3 gridDim((cols + blockDim.x - 1) / blockDim.x, (rows + blockDim.y - 1) / blockDim.y);

    // Chama o kernel CUDA para processar a imagem
    grayscaleKernel<<<gridDim, blockDim>>>(d_input, d_output, rows, cols);

    // Copia os dados da imagem de saída da GPU para a CPU
    uchar* outputData = new uchar[rows * cols];
    cudaMemcpy(outputData, d_output, sizeof(uchar) * rows * cols, cudaMemcpyDeviceToHost);

    // Cria a imagem de saída utilizando os dados copiados
    cv::Mat outputImage(rows, cols, CV_8UC1, outputData);

    // Exibe as imagens de entrada e saída
    cv::imshow("Input Image", inputImage);
    cv::imshow("Output Image", outputImage);
    cv::waitKey(0);

    // Libera a memória alocada na GPU e na CPU
    cudaFree(d_input);
    cudaFree(d_output);
    delete[] outputData;

    return 0;
}
