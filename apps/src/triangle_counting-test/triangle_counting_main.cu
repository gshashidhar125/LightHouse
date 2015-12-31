#include "triangle_counting.h"
#include "triangle_counting.cu"
#include <fstream>

triangle_countingMacro;
#include "graph.h"
#include "verify_triangle_counting.h"
unsigned long long int triangle_counting_CPU(int* G0, int* G1) {

    {
        h_T = 0;
        err = cudaMemcpyToSymbol(T, &h_T, sizeof(unsigned long long int), 0, cudaMemcpyHostToDevice);
        CUDA_ERR_CHECK;
        cudaOccupancyMaxPotentialBlockSize(&gm_minGridSize, &gm_blockSize,forEachKernel0, 0, 0);
        gm_gridSize = (NumNodes + 1 + gm_blockSize - 1) / gm_blockSize;
        gm_numBlocksStillToProcess = gm_gridSize, gm_offsetIntoBlocks = 0;
        while (gm_numBlocksStillToProcess > 0) {
            if (gm_numBlocksStillToProcess > gm_minGridSize)
                gm_numBlocksKernelParameter = gm_minGridSize;
            else
                gm_numBlocksKernelParameter = gm_numBlocksStillToProcess;
            forEachKernel0<<<gm_numBlocksKernelParameter, gm_blockSize>>>(G0, G1, NumNodes, NumEdges, gm_offsetIntoBlocks, h_inNode);
            CUDA_ERR_CHECK;
            gm_numBlocksStillToProcess -= gm_minGridSize;
            gm_offsetIntoBlocks += gm_minGridSize * gm_blockSize;
        }
        err = cudaMemcpyFromSymbol(&h_T, T, sizeof(unsigned long long int), 0, cudaMemcpyDeviceToHost);
        CUDA_ERR_CHECK;
        return h_T;
    }
}


using namespace std;
// triangle_counting -? : for how to run generated main program
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
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    err = cudaMalloc((void **)&G0, (NumNodes + 2) * sizeof(int));
    CUDA_ERR_CHECK;
    err = cudaMalloc((void **)&G1, (NumEdges + 1) * sizeof(int));
    CUDA_ERR_CHECK;
    err = cudaMalloc((void **)&h_inNode, (NumEdges + 1) * sizeof(int));
    CUDA_ERR_CHECK;
    err = cudaMalloc((void **)&host_threadBlockBarrierReached, 10000 * sizeof(bool));
    CUDA_ERR_CHECK;
    err = cudaMemset(host_threadBlockBarrierReached, 0x0, 10000 * sizeof(bool));
    CUDA_ERR_CHECK;
    err = cudaMemcpyToSymbol(gm_threadBlockBarrierReached, &host_threadBlockBarrierReached, sizeof(bool *), 0, cudaMemcpyHostToDevice);
    CUDA_ERR_CHECK;
    int* h_G[2];
    printf("Graph Population began\n");
    populate(argv[1], h_G);
    printf("Graph Population end\n");
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Loading time(milliseconds)  = %f\n", elapsedTime);
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    unsigned long long int MainReturn;
    MainReturn = triangle_counting_CPU(G0, G1);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Execution time(milliseconds)  = %f\n", elapsedTime);
    bool gm_verify = verifytriangle_counting(h_G);
    if (!gm_verify) {
        printf("Verification Failed\n");
        return -1;
    } else {
        printf("Verification Success\n");
    }
    err = cudaFree(G0);
    CUDA_ERR_CHECK;
    err = cudaFree(G1);
    CUDA_ERR_CHECK;
    err = cudaFree(h_inNode);
    CUDA_ERR_CHECK;
    err = cudaFree(host_threadBlockBarrierReached);
    CUDA_ERR_CHECK;
    free(h_G[0]);
    free(h_G[1]);
    printf("Return value = %d\n", MainReturn);return MainReturn;
}
