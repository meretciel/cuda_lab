#include <cuda_runtime.h>
#include <stdio.h>


int main(int argc, char **argv) {
    printf("%s Starting...\n", argv[0]);
    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);


    if (error_id != cudaSuccess) {
        printf("cudaGetDeviceCount returned %d\n-> %s\n", (int) error_id, cudaGetErrorString(error_id));
        printf("Result = FAIL\n");
        exit(EXIT_FAILURE);
    }

    if (deviceCount == 0) {
        printf("There are no available device(s) that support CUDA\n");
    } else {
        printf("-----------------------------------------------\n");
        printf("Detected %d CUDA Capable device(s)\n", deviceCount);
        printf("-----------------------------------------------\n");
    }

    for (int dev = 0; dev < deviceCount; ++dev) {
        printf("\n");
        int driverVersion = 0;
        int runtimeVersion = 0;
        dev = 0;

        cudaSetDevice(dev);
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        printf("Device %d: \"%s\"\n", dev, deviceProp.name);
        cudaDriverGetVersion(&driverVersion);
        cudaRuntimeGetVersion(&runtimeVersion);
        printf("\tCUDA Driver Version / Runtime Version %d.%d / %d.%d\n",
               driverVersion / 1000, (driverVersion % 100) / 10,
               runtimeVersion / 1000, (runtimeVersion % 100) / 10);
        printf("\tCUDA Capability Major/Minor version number: %d.%d\n",
               deviceProp.major, deviceProp.minor);
        printf("\tTotal amount of global memory: %.2f MBytes (%llu bytes)\n",
               (float) deviceProp.totalGlobalMem / (pow(1024.0, 3)),
               (unsigned long long) deviceProp.totalGlobalMem);
        printf("\tGPU Clock rate: %.0f MHz (%0.2f GHz)\n", deviceProp.clockRate * 1e-3f, deviceProp.clockRate * 1e-6f);
        printf("\tMemory Clock rate: %.0f Mhz\n", deviceProp.memoryClockRate * 1e-3f);
        printf("\tMemory Bus Width: %d-bit\n", deviceProp.memoryBusWidth);

        if (deviceProp.l2CacheSize) {
            printf("\tL2 Cache Size: %d bytes\n", deviceProp.l2CacheSize);
        }

        printf("\tMax Texture Dimension Size (x,y,z) "
               " 1D=(%d), 2D=(%d,%d), 3D=(%d,%d,%d)\n",
               deviceProp.maxTexture1D
               , deviceProp.maxTexture2D[0],
               deviceProp.maxTexture2D[1],
               deviceProp.maxTexture3D[0], deviceProp.maxTexture3D[1],
               deviceProp.maxTexture3D[2]);
        printf("\tMax Layered Texture Size (dim) x layers 1D=(%d) x %d, 2D=(%d,%d) x %d\n",
               deviceProp.maxTexture1DLayered[0], deviceProp.maxTexture1DLayered[1],
               deviceProp.maxTexture2DLayered[0], deviceProp.maxTexture2DLayered[1],
               deviceProp.maxTexture2DLayered[2]);

        printf("\tTotal amount of constant memory: %lu bytes\n", deviceProp.totalConstMem);
        printf("\tTotal amount of shared memory per block: %lu bytes\n", deviceProp.sharedMemPerBlock);
        printf("\tTotal number of registers available per block: %d\n", deviceProp.regsPerBlock);
        printf("\tWarp size: %d\n", deviceProp.warpSize);
        printf("\tMaximum number of threads per multiprocessor: %d\n", deviceProp.maxThreadsPerMultiProcessor);
        printf("\tMaximum number of threads per block: %d\n", deviceProp.maxThreadsPerBlock);
        printf("\tMaximum sizes of each dimension of a block: %d x %d x %d\n",
               deviceProp.maxThreadsDim[0],
               deviceProp.maxThreadsDim[1],
               deviceProp.maxThreadsDim[2]);
        printf("\tMaximum sizes of each dimension of a grid: %d x %d x %d\n",
               deviceProp.maxGridSize[0],
               deviceProp.maxGridSize[1],
               deviceProp.maxGridSize[2]);
        printf("\tMaximum memory pitch: %lu bytes\n", deviceProp.memPitch);
    }

    exit(EXIT_SUCCESS);
}
