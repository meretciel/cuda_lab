
#include <iostream>
#include <cuda_runtime.h>
#include <cstdio>
#include <stdlib.h>
#include <time.h>

#define TILE_WIDTH 4

void print_matrix(int* mat, int nrows, int ncols) {
    printf("[");
    for(int i = 0; i < nrows; i++)  {// Displaying first 10 elements
        for (int j = 0; j < ncols; j++) {
            if (j == 0) {
                printf("[%d", mat[i * ncols + j]);
            } else {
                printf(",%d", mat[i * ncols + j]);
            }
        }
        printf("],\n");
    }
    printf("]");
}

__global__
void matrixMul(int* M, int* N, int* P, int width) {
    __shared__ int Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ int Nds[TILE_WIDTH][TILE_WIDTH];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int row = by * blockDim.y + ty;
    int col = bx * blockDim.x + tx;

    int v = 0;
    for (int phase = 0; phase < ceil(width / (float)TILE_WIDTH); ++phase) {
        // fetch data to shared memory
        if (col < width) {
            Mds[ty][tx] = M[(ty + by * blockDim.y) * width + (phase * TILE_WIDTH + tx)];
        } else {
            Mds[ty][tx] = 0;
        }

        if (row < width) {
            Nds[ty][tx] = N[ (phase * TILE_WIDTH + ty) * width  + (bx * blockDim.x + tx)];

        } else {
            Nds[ty][tx] = 0;
        }

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k) {
            v += Mds[ty][k] * Nds[k][tx];
        }
        __syncthreads();
    }

    P[row * width + col] = v;
}

int main() {
    time_t t;
    srand((unsigned) time(&t));

    int width= 40;
    int nrows = width;
    int ncols = width;
    int nitems = nrows * ncols;


    int *M, *N, *P; // Host arrays
    int *M_d, *N_d, *P_d; // Device arrays

    // Allocate memory on the host
    M = (int*)malloc(nitems * sizeof(int));
    N = (int*)malloc(nitems * sizeof(int));
    P = (int*)malloc(nitems * sizeof(int));

    // Initialize arrays with data
    for (int i = 0; i < nrows; ++i) {
        for (int j = 0; j < ncols; ++j) {
            M[i * ncols + j] = rand() % 20;
            N[i * ncols + j] = rand() % 20;
        }
    }

    printf("M:\n");
    print_matrix(M, nrows, ncols);

    printf("\nN:\n");
    print_matrix(N, nrows, ncols);

    // Allocate memory on the device
    cudaMalloc((void **)&M_d, nitems * sizeof(int));
    cudaMalloc((void **)&N_d, nitems * sizeof(int));
    cudaMalloc((void **)&P_d, nitems * sizeof(int));

    // Copy data from host to device
    cudaMemcpy(M_d, M, nitems * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(N_d, N, nitems * sizeof(int), cudaMemcpyHostToDevice);

    // Launch the kernel
    dim3 dimGrid(ceil(width / TILE_WIDTH), ceil(width / 4), 1);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    matrixMul<<<dimGrid, dimBlock>>>(M_d, N_d, P_d, width);

    // Copy result back to host
    cudaMemcpy(P, P_d, nitems * sizeof(int), cudaMemcpyDeviceToHost);

    printf("\nResult:\n");
    print_matrix(P, nrows, ncols);

    // Free device memory
    cudaFree(M_d);
    cudaFree(N_d);
    cudaFree(P_d);

    // Free host memory
    free(M);
    free(N);
    free(P);

    return 0;
}