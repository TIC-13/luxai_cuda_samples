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

        if ((row < width) && (t*TILE_WIDTH+tx) < width){
            Ms[ty][tx] = M[row*width + t*TILE_WIDTH + tx];
        } else {Ms[ty][tx] = 0.0f;}

        if ((t*TILE_WIDTH+ty) < width && col < width) {
            Ns[ty][tx] = N[(t*TILE_WIDTH + ty)*width + col];
        } else {Ns[ty][tx] = 0.0f;}

        __syncthreads();

        for(int k=0; k < TILE_WIDTH; ++k){
            PValue += Ms[ty][k] + Ns[k][tx];
        }
        __syncthreads();
    }
    if (row < width) && (col < width){
        P[row*width+col] = PValue;
    }
}