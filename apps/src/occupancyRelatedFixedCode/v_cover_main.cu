#include "v_cover.h"
#include "v_cover.cu"
#include <fstream>

v_coverMacro;
#include "graph.h"
#define NUM_THREADS_PER_BLOCK 5
int v_cover_CPU(int* G0, int* G1, bool * selectEdge) {

    {
        remain = (int)(NumEdges * 2);
        h___S4 = 0;
        err = cudaMemcpyToSymbol(__S4, &h___S4, sizeof(int), 0, cudaMemcpyHostToDevice);
        CUDA_ERR_CHECK;
        int numThreadsPerBlock = 400; 
        //NumNodes = 10000;
  int blockSize;   // The launch configurator returned block size
  int minGridSize; // The minimum grid size needed to achieve the
                   // maximum occupancy for a full device launch
  int gridSize;    // The actual grid size needed, based on input size

  cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize,
                                      forEachKernel0, 0, 0);
  // Round up according to array size
  gridSize = (NumNodes + blockSize - 1) / blockSize;

    int i = 0, offset = 0;
    while (i < gridSize) {
        printf("NumBlocks = %d, numThreadsPerBlock = %d, minGridSize = %d\n", gridSize, blockSize, minGridSize);
        forEachKernel0<<<minGridSize, blockSize>>>(G0, G1, NumNodes, NumEdges, selectEdge, Deg, Covered, offset);
        CUDA_ERR_CHECK;
        i += minGridSize;
        offset += minGridSize * blockSize;
    }
    // calculate theoretical occupancy
  int maxActiveBlocks;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor( &maxActiveBlocks,
                                                 forEachKernel0, blockSize,
                                                 0);

  int device;
  cudaDeviceProp props;
  cudaGetDevice(&device);
  cudaGetDeviceProperties(&props, device);

  float occupancy = (maxActiveBlocks * blockSize / props.warpSize) /
                    (float)(props.maxThreadsPerMultiProcessor /
                            props.warpSize);

  printf("Launched blocks of size %d. Theoretical occupancy: %f\n",
         blockSize, occupancy);
      
        //printf("NumNodes = %d, numBlocks = %d\n", NumNodes, NumNodes / NUM_THREADS_PER_BLOCK + 1);
        //forEachKernel0<<<NumNodes / NUM_THREADS_PER_BLOCK + 1, NUM_THREADS_PER_BLOCK>>>(G0, G1, NumNodes, NumEdges, selectEdge, Deg, Covered);
        return 1;
        printf("After kernel0\n");
        forEachKernel1<<<NumEdges / numThreadsPerBlock + 1, numThreadsPerBlock>>>(G0, G1, NumNodes, NumEdges, selectEdge);
        CUDA_ERR_CHECK;
        printf("After kernel1\n");
        while (remain > 0)
        {
            h_max_val = 0;
            err = cudaMemcpyToSymbol(max_val, &h_max_val, sizeof(int), 0, cudaMemcpyHostToDevice);
            CUDA_ERR_CHECK;
            forEachKernel2<<<NumNodes / 512 + 1, 512>>>(G0, G1, NumNodes, NumEdges, selectEdge, Covered, Deg);
            CUDA_ERR_CHECK;
            printf("After kernel2\n");
            bool tempVar0 = true;
            err = cudaMemcpyFromSymbol(&h_from, from, sizeof(int), 0, cudaMemcpyDeviceToHost);
            CUDA_ERR_CHECK;
            err = cudaMemcpy(Covered + h_from, &tempVar0, 1 * sizeof(bool), cudaMemcpyHostToDevice);
            CUDA_ERR_CHECK;
            tempVar0 = true;
            err = cudaMemcpyFromSymbol(&h_to, to, sizeof(int), 0, cudaMemcpyDeviceToHost);
            CUDA_ERR_CHECK;
            err = cudaMemcpy(Covered + h_to, &tempVar0, 1 * sizeof(bool), cudaMemcpyHostToDevice);
            CUDA_ERR_CHECK;
            tempVar0 = true;
            err = cudaMemcpyFromSymbol(&h_e, e, sizeof(int), 0, cudaMemcpyDeviceToHost);
            CUDA_ERR_CHECK;
            err = cudaMemcpy(selectEdge + h_e, &tempVar0, 1 * sizeof(bool), cudaMemcpyHostToDevice);
            CUDA_ERR_CHECK;
            int tempVar1 = 0;
            err = cudaMemcpyFromSymbol(&h_from, from, sizeof(int), 0, cudaMemcpyDeviceToHost);
            CUDA_ERR_CHECK;
            err = cudaMemcpy(Deg + h_from, &tempVar1, 1 * sizeof(int), cudaMemcpyHostToDevice);
            CUDA_ERR_CHECK;
            tempVar1 = 0;
            err = cudaMemcpyFromSymbol(&h_to, to, sizeof(int), 0, cudaMemcpyDeviceToHost);
            CUDA_ERR_CHECK;
            err = cudaMemcpy(Deg + h_to, &tempVar1, 1 * sizeof(int), cudaMemcpyHostToDevice);
            CUDA_ERR_CHECK;
            err = cudaMemcpyFromSymbol(&h_max_val, max_val, sizeof(int), 0, cudaMemcpyDeviceToHost);
            CUDA_ERR_CHECK;
            remain = remain - h_max_val;
        }
        forEachKernel3<<<NumNodes / 512 + 1, 512>>>(G0, G1, NumNodes, NumEdges, selectEdge, Covered);
        CUDA_ERR_CHECK;
        printf("After kernel3\n");
        err = cudaMemcpyFromSymbol(&h___S4, __S4, sizeof(int), 0, cudaMemcpyDeviceToHost);
        CUDA_ERR_CHECK;
        return h___S4;
    }
}


using namespace std;
// v_cover -? : for how to run generated main program
int main(int argc, char* argv[])
{

    if (argc != 2 || argv[1] == NULL) {
        printf("Wrong Number of Arguments");
        exit(1);
    }
    ifstream inputFile;
    inputFile.open(argv[1]);
    if (!inputFile.is_open()){
        printf("invalid file");
        exit(1);
    }
    inputFile >> NumNodes >> NumEdges;
    cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 1048576 * 100);
    cudaSetDevice(7);
    err = cudaMalloc((void **)&G0, (NumNodes + 2) * sizeof(int));
    CUDA_ERR_CHECK;
    err = cudaMalloc((void **)&G1, (NumEdges + 1) * sizeof(int));
    CUDA_ERR_CHECK;
    err = cudaMalloc((void **)&selectEdge, (NumEdges + 1) * sizeof(bool));
    CUDA_ERR_CHECK;
    err = cudaMalloc((void **)&Deg, (NumNodes + 1) * sizeof(int));
    CUDA_ERR_CHECK;
    err = cudaMalloc((void **)&Covered, (NumNodes + 1) * sizeof(bool));
    CUDA_ERR_CHECK;
    err = cudaMalloc((void **)&host_threadBlockBarrierReached, (NumNodes / NUM_THREADS_PER_BLOCK + 1) * sizeof(bool));
    CUDA_ERR_CHECK;
    err = cudaMemset(host_threadBlockBarrierReached, 0x0, (NumNodes / NUM_THREADS_PER_BLOCK + 1) * sizeof(bool));
    CUDA_ERR_CHECK;
    err = cudaMemcpyToSymbol(gm_threadBlockBarrierReached, &host_threadBlockBarrierReached, sizeof(bool *), 0, cudaMemcpyHostToDevice);
    CUDA_ERR_CHECK;
    err = cudaMalloc((void **)&host_threadBlockBarrierReached, (NumNodes / NUM_THREADS_PER_BLOCK + 1) * sizeof(bool));
    CUDA_ERR_CHECK;
    err = cudaMemset(host_threadBlockBarrierReached, 0x0, (NumNodes / NUM_THREADS_PER_BLOCK + 1) * sizeof(bool));
    CUDA_ERR_CHECK;
    err = cudaMemcpyToSymbol(gm_threadBlockBarrierSignal, &host_threadBlockBarrierReached, sizeof(bool *), 0, cudaMemcpyHostToDevice);
    CUDA_ERR_CHECK;
    int* h_G[2];
    printf("Graph Population began\n");
     populate(argv[1], h_G);
    printf("Graph Population end\n");
    int MainReturn;
    MainReturn = v_cover_CPU(G0, G1, selectEdge);

    return MainReturn;
}
