
#include <cuda_runtime.h>
#include <stdio.h>

int main(int argc, char **argv) {
    int devCount;
    cudaGetDeviceCount(&devCount);
    printf("Number of device: %d", devCount);

    cudaDeviceProp devprop;
    cudaGetDeviceProperties(&deviceProp, 0);
    printf("%s using %d: %s\n", argv[0], 0, deviceProp.name);

}