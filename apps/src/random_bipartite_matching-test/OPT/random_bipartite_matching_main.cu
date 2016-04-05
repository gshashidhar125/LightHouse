#include "random_bipartite_matching.h"
#include "random_bipartite_matching.cu"
#include <fstream>

random_bipartite_matchingMacro;
#include "graph.h"
#include "verify_random_bipartite_matching.h"
int random_bipartite_matching_CPU(int* G0, int* G1, bool * isLeft, int * Match) {

    {
        h_count = 0;
        err = cudaMemcpyToSymbol(count, &h_count, sizeof(int), 0, cudaMemcpyHostToDevice);
        CUDA_ERR_CHECK;
        h_finished = false;
        err = cudaMemcpyToSymbol(finished, &h_finished, sizeof(bool), 0, cudaMemcpyHostToDevice);
        CUDA_ERR_CHECK;
        cudaOccupancyMaxPotentialBlockSize(&gm_minGridSize, &gm_blockSize,forEachKernel0, 0, 0);
        gm_gridSize = (NumNodes + 1 + gm_blockSize - 1) / gm_blockSize;
        gm_numBlocksStillToProcess = gm_gridSize, gm_offsetIntoBlocks = 0;
        while (gm_numBlocksStillToProcess > 0) {
            if (gm_numBlocksStillToProcess > gm_minGridSize)
                gm_numBlocksKernelParameter = gm_minGridSize;
            else
                gm_numBlocksKernelParameter = gm_numBlocksStillToProcess;
            forEachKernel0<<<gm_numBlocksKernelParameter, gm_blockSize>>>(G0, G1, NumNodes, NumEdges, isLeft, Match, Suitor, gm_offsetIntoBlocks);
            CUDA_ERR_CHECK;
            gm_numBlocksStillToProcess -= gm_minGridSize;
            gm_offsetIntoBlocks += gm_minGridSize * gm_blockSize;
        }
        while ( !h_finished)
        {
            h_finished = false;
            err = cudaMemcpyToSymbol(finished, &h_finished, sizeof(bool), 0, cudaMemcpyHostToDevice);
            CUDA_ERR_CHECK;
            h_finished = true;
            err = cudaMemcpyToSymbol(finished, &h_finished, sizeof(bool), 0, cudaMemcpyHostToDevice);
            CUDA_ERR_CHECK;
            cudaOccupancyMaxPotentialBlockSize(&gm_minGridSize, &gm_blockSize,forEachKernel1, 0, 0);
            gm_gridSize = (NumNodes + 1 + gm_blockSize - 1) / gm_blockSize;
            gm_numBlocksStillToProcess = gm_gridSize, gm_offsetIntoBlocks = 0;
            while (gm_numBlocksStillToProcess > 0) {
                if (gm_numBlocksStillToProcess > gm_minGridSize)
                    gm_numBlocksKernelParameter = gm_minGridSize;
                else
                    gm_numBlocksKernelParameter = gm_numBlocksStillToProcess;
                forEachKernel1<<<gm_numBlocksKernelParameter, gm_blockSize>>>(G0, G1, NumNodes, NumEdges, isLeft, Match, Suitor, gm_offsetIntoBlocks);
                CUDA_ERR_CHECK;
                gm_numBlocksStillToProcess -= gm_minGridSize;
                gm_offsetIntoBlocks += gm_minGridSize * gm_blockSize;
            }
            cudaOccupancyMaxPotentialBlockSize(&gm_minGridSize, &gm_blockSize,forEachKernel2, 0, 0);
            gm_gridSize = (NumNodes + 1 + gm_blockSize - 1) / gm_blockSize;
            gm_numBlocksStillToProcess = gm_gridSize, gm_offsetIntoBlocks = 0;
            while (gm_numBlocksStillToProcess > 0) {
                if (gm_numBlocksStillToProcess > gm_minGridSize)
                    gm_numBlocksKernelParameter = gm_minGridSize;
                else
                    gm_numBlocksKernelParameter = gm_numBlocksStillToProcess;
                forEachKernel2<<<gm_numBlocksKernelParameter, gm_blockSize>>>(G0, G1, NumNodes, NumEdges, isLeft, Match, Suitor, gm_offsetIntoBlocks);
                CUDA_ERR_CHECK;
                gm_numBlocksStillToProcess -= gm_minGridSize;
                gm_offsetIntoBlocks += gm_minGridSize * gm_blockSize;
            }
            cudaOccupancyMaxPotentialBlockSize(&gm_minGridSize, &gm_blockSize,forEachKernel3, 0, 0);
            gm_gridSize = (NumNodes + 1 + gm_blockSize - 1) / gm_blockSize;
            gm_numBlocksStillToProcess = gm_gridSize, gm_offsetIntoBlocks = 0;
            while (gm_numBlocksStillToProcess > 0) {
                if (gm_numBlocksStillToProcess > gm_minGridSize)
                    gm_numBlocksKernelParameter = gm_minGridSize;
                else
                    gm_numBlocksKernelParameter = gm_numBlocksStillToProcess;
                forEachKernel3<<<gm_numBlocksKernelParameter, gm_blockSize>>>(G0, G1, NumNodes, NumEdges, isLeft, Match, Suitor, gm_offsetIntoBlocks);
                CUDA_ERR_CHECK;
                gm_numBlocksStillToProcess -= gm_minGridSize;
                gm_offsetIntoBlocks += gm_minGridSize * gm_blockSize;
            }
        }
        err = cudaMemcpyFromSymbol(&h_count, count, sizeof(int), 0, cudaMemcpyDeviceToHost);
        CUDA_ERR_CHECK;
        return h_count;
    }
}


using namespace std;
// random_bipartite_matching -? : for how to run generated main program
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
    err = cudaMalloc((void **)&isLeft, (NumNodes + 1) * sizeof(bool));
    CUDA_ERR_CHECK;
    err = cudaMalloc((void **)&Match, (NumNodes + 1) * sizeof(int));
    CUDA_ERR_CHECK;
    err = cudaMalloc((void **)&Suitor, (NumNodes + 1) * sizeof(int));
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
    int MainReturn;
    MainReturn = random_bipartite_matching_CPU(G0, G1, isLeft, Match);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Execution time(milliseconds)  = %f\n", elapsedTime);
    bool gm_verify = verifyrandom_bipartite_matching(h_G);
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
    err = cudaFree(isLeft);
    CUDA_ERR_CHECK;
    err = cudaFree(Match);
    CUDA_ERR_CHECK;
    err = cudaFree(Suitor);
    CUDA_ERR_CHECK;
    err = cudaFree(host_threadBlockBarrierReached);
    CUDA_ERR_CHECK;
    free(h_G[0]);
    free(h_G[1]);
    printf("Return value = %d\n", MainReturn);return MainReturn;
}
