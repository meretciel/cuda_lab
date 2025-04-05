
#include <cstdio>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void sumArraysOnHost(float *A, float *B, float *C, int N) {
    for (int idx=0; idx<N; idx++) {
        C[idx] = A[idx] + B[idx];
    }
}

__global__ void checkIndex(void) {
    printf("threadIdx:(%d, %d, %d) blockIdx:(%d, %d, %d) blockDim:(%d, %d, %d) "
    "gridDim:(%d, %d, %d)\n", threadIdx.x, threadIdx.y, threadIdx.z,
    blockIdx.x, blockIdx.y, blockIdx.z, blockDim.x, blockDim.y, blockDim.z,

    gridDim.x,gridDim.y,gridDim.z);
}

void initialData(float *ip, int size) {
    time_t t;
    srand((unsigned int) time(&t));
    for (int i = 0; i < size; i++) {
        ip[i] = (float) rand() / (float) RAND_MAX;
    }
}

int main(int argc, char **argv) {
    int nElem = 6;
    size_t nBytes = nElem * sizeof(float);

    dim3 block(3);
    dim3 grid((nElem + block.x - 1) / block.x);

    printf("grid.x %d grid.y %d grid.z %d\n",grid.x, grid.y, grid.z);
    printf("block.x %d block.y %d block.z %d\n",block.x, block.y, block.z);

    checkIndex <<<grid, block>>> ();
    cudaDeviceReset();

//    float* h_A;
//    float* h_B;
//    float* h_C;
//
//    h_A = (float *) malloc(nBytes);
//    h_B = (float *) malloc(nBytes);
//    h_C = (float *) malloc(nBytes);
//
//    initialData(h_A, nElem);
//    initialData(h_B, nElem);
//
//    sumArraysOnHost(h_A, h_B, h_C, nElem);
//    free(h_A);
//    free(h_B);
//    free(h_C);
    return 0;
}