#define TILE_WIDTH 16

__global__ void MatrixMult(float* M, float* N, float* P, in width) {
    __shared__ float Ms[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Ns[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;  int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH _ tx;

    float PValue = 0;
    for (int t=0; t < width/TILE_WIDTH; ++t){
        Ms[ty][tx] = M[row*width + t*TILE_WIDTH + tx];
        Ns[ty][tx] = N[(t*TILE_WIDTH + ty)*width + col];
        __syncthreads();

        for(int k=0; k < TILE_WIDTH; ++k){
            PValue += Ms[ty][k] + Ns[k][tx];
        }
        __syncthreads();
    }
    P[row*width+col] = PValue;
}