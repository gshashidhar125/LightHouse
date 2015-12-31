#include "v_cover.h"
#include "v_cover.cu"
#include <fstream>

v_coverMacro;
#include "graph.h"
#include "verify_v_cover.h"
int v_cover_CPU(int* G0, int* G1, bool * selectEdge) {

    {
        remain = (int)(NumEdges * 2);
        h___S4 = 0;
        err = cudaMemcpyToSymbol(__S4, &h___S4, sizeof(int), 0, cudaMemcpyHostToDevice);
        CUDA_ERR_CHECK;
        cudaOccupancyMaxPotentialBlockSize(&gm_minGridSize, &gm_blockSize,forEachKernel0, 0, 0);
        gm_gridSize = (NumNodes + 1 + gm_blockSize - 1) / gm_blockSize;
        gm_numBlocksStillToProcess = gm_gridSize, gm_offsetIntoBlocks = 0;
        while (gm_numBlocksStillToProcess > 0) {
            if (gm_numBlocksStillToProcess > gm_minGridSize)
                gm_numBlocksKernelParameter = gm_minGridSize;
            else
                gm_numBlocksKernelParameter = gm_numBlocksStillToProcess;
            forEachKernel0<<<gm_numBlocksKernelParameter, gm_blockSize>>>(G0, G1, NumNodes, NumEdges, selectEdge, Deg, Covered, gm_offsetIntoBlocks);
            CUDA_ERR_CHECK;
            gm_numBlocksStillToProcess -= gm_minGridSize;
            gm_offsetIntoBlocks += gm_minGridSize * gm_blockSize;
        }
        cudaOccupancyMaxPotentialBlockSize(&gm_minGridSize, &gm_blockSize,forEachKernel1, 0, 0);
        gm_gridSize = (NumEdges + gm_blockSize - 1) / gm_blockSize;
        gm_numBlocksStillToProcess = gm_gridSize, gm_offsetIntoBlocks = 0;
        while (gm_numBlocksStillToProcess > 0) {
            if (gm_numBlocksStillToProcess > gm_minGridSize)
                gm_numBlocksKernelParameter = gm_minGridSize;
            else
                gm_numBlocksKernelParameter = gm_numBlocksStillToProcess;
            forEachKernel1<<<gm_numBlocksKernelParameter, gm_blockSize>>>(G0, G1, NumNodes, NumEdges, selectEdge, gm_offsetIntoBlocks);
            CUDA_ERR_CHECK;
            gm_numBlocksStillToProcess -= gm_minGridSize;
            gm_offsetIntoBlocks += gm_minGridSize * gm_blockSize;
        }
        while (remain > 0)
        {
            h_max_val = 0;
            err = cudaMemcpyToSymbol(max_val, &h_max_val, sizeof(int), 0, cudaMemcpyHostToDevice);
            CUDA_ERR_CHECK;
            cudaOccupancyMaxPotentialBlockSize(&gm_minGridSize, &gm_blockSize,forEachKernel2, 0, 0);
            gm_gridSize = (NumNodes + 1 + gm_blockSize - 1) / gm_blockSize;
            gm_numBlocksStillToProcess = gm_gridSize, gm_offsetIntoBlocks = 0;
            while (gm_numBlocksStillToProcess > 0) {
                if (gm_numBlocksStillToProcess > gm_minGridSize)
                    gm_numBlocksKernelParameter = gm_minGridSize;
                else
                    gm_numBlocksKernelParameter = gm_numBlocksStillToProcess;
                forEachKernel2<<<gm_numBlocksKernelParameter, gm_blockSize>>>(G0, G1, NumNodes, NumEdges, selectEdge, Covered, Deg, gm_offsetIntoBlocks);
                CUDA_ERR_CHECK;
                gm_numBlocksStillToProcess -= gm_minGridSize;
                gm_offsetIntoBlocks += gm_minGridSize * gm_blockSize;
            }
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
        cudaOccupancyMaxPotentialBlockSize(&gm_minGridSize, &gm_blockSize,forEachKernel3, 0, 0);
        gm_gridSize = (NumNodes + 1 + gm_blockSize - 1) / gm_blockSize;
        gm_numBlocksStillToProcess = gm_gridSize, gm_offsetIntoBlocks = 0;
        while (gm_numBlocksStillToProcess > 0) {
            if (gm_numBlocksStillToProcess > gm_minGridSize)
                gm_numBlocksKernelParameter = gm_minGridSize;
            else
                gm_numBlocksKernelParameter = gm_numBlocksStillToProcess;
            forEachKernel3<<<gm_numBlocksKernelParameter, gm_blockSize>>>(G0, G1, NumNodes, NumEdges, selectEdge, Covered, gm_offsetIntoBlocks);
            CUDA_ERR_CHECK;
            gm_numBlocksStillToProcess -= gm_minGridSize;
            gm_offsetIntoBlocks += gm_minGridSize * gm_blockSize;
        }
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
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
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
    MainReturn = v_cover_CPU(G0, G1, selectEdge);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Execution time(milliseconds)  = %f\n", elapsedTime);
    bool gm_verify = verifyv_cover(h_G);
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
    err = cudaFree(selectEdge);
    CUDA_ERR_CHECK;
    err = cudaFree(Deg);
    CUDA_ERR_CHECK;
    err = cudaFree(Covered);
    CUDA_ERR_CHECK;
    err = cudaFree(host_threadBlockBarrierReached);
    CUDA_ERR_CHECK;
    free(h_G[0]);
    free(h_G[1]);
    printf("Return value = %d\n", MainReturn);return MainReturn;
}
