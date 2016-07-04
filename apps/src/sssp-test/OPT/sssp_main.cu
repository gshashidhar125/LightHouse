#include "sssp.h"
#include "sssp.cu"
#include <fstream>
#include <stdio.h>
#include <time.h>

ssspMacro;
#include "graph.h"
#include "verify_sssp.h"

void allocateThreads(int* h_G[2]) {
    int maxThreadsPerBlock =  1024;//prop.maxThreadsPerBlock;
#ifdef ALLOC_K2EXACTEDGES
    numThreadsReq = (NumEdges - 1) / (K1 + K2) + 1;
#else
    numThreadsReq = (NumEdges - 1) / (K1) + 1;
#endif
    int numBlocks = (numThreadsReq - 1) / maxThreadsPerBlock + 1;
    
    int *h_allocEdgesToThreads0, *h_allocEdgesToThreads1;
    int* nodeNeighborOffset;
    bool* isEdgeAllocated;
    h_allocEdgesToThreads0 = new int [numThreadsReq + 1]();
    h_allocEdgesToThreads1 = new int [NumEdges + 1]();
    nodeNeighborOffset = new int [NumNodes]();
    isEdgeAllocated = new bool [NumEdges]();
    
    for (int i = 0; i < NumNodes; i++) {
        nodeNeighborOffset[i] = 0;
    }
    for (int i = 0; i < NumEdges; i++) {
        isEdgeAllocated[i] = false;
    }

    int currentEdge = 0, currentThread = 0, threadIndex = 0;
    //allocK1IncomingEdgesToThreads();
    /*for (int i = 0; i < NumEdges; i += K1) {
        for (int j = 0; j < K1; j++) {
            h_allocEdgesToThreads1[currentEdge] = currentThread;
            currentEdge++;
        }
        currentThread++;
    }*/

    currentEdge = 0;
    currentThread = 0;
    h_allocEdgesToThreads0[0] = 0;
    for (int  i = 0; i < NumEdges && threadIndex < NumEdges; i += K1) {
        
        h_allocEdgesToThreads0[currentThread] = threadIndex;
        /*
//        currentEdge = currentThread * (K1 + K2);
        for (int j = 0; j < K1; j++) {
            h_allocEdgesToThreads1[threadIndex] = threadIndex;
            //currentEdge++;
            threadIndex++;
        }*/
        while (isEdgeAllocated[currentEdge++] && currentEdge < NumEdges);

        currentEdge--;
        int lastIncomingEdge = currentEdge;
        for (int j = 0, k = 0; k < K1 && (currentEdge + j) < NumEdges; j++) {
            if (!isEdgeAllocated[currentEdge + j]) {
                h_allocEdgesToThreads1[threadIndex] = currentEdge + j;
                isEdgeAllocated[currentEdge + j] = 1;
                k++;
                threadIndex++;
                lastIncomingEdge = currentEdge + j;
            }
        }
        //printf("###Thread: %d\n", currentThread);
        int j;
        for (j = 0; j < K2 && currentEdge <= lastIncomingEdge && currentEdge < NumEdges && threadIndex < NumEdges;) {
            int currentNode = h_G[1][currentEdge];
            int numNeighbors = h_G[0][currentNode + 1] - h_G[0][currentNode];
            if (numNeighbors <= nodeNeighborOffset[currentNode]) {
                currentEdge++;
            } else {
                for (int k = nodeNeighborOffset[currentNode]; j < K2 && k < numNeighbors && threadIndex < NumEdges; k++) {
                    if (!isEdgeAllocated[h_G[0][currentNode] + k]) {
                        h_allocEdgesToThreads1[threadIndex++] = h_G[0][currentNode] + k;
                        isEdgeAllocated[h_G[0][currentNode] + k] = true;
                        j++;
                        nodeNeighborOffset[currentNode] ++;
                    }
                }
                currentEdge++;
            }
            /*else if ((numNeighbors - nodeNeighborOffset[currentNode]) >= (K2 - j)) {
                printf("    Allocate %d outgoing Edges from the node %d from index %d\n", K2 - j, currentNode, threadIndex);
                for (int k = 0; k < (K2 - j) && threadIndex < NumEdges; k++) {
                    h_allocEdgesToThreads1[threadIndex++] = h_G[0][currentNode] + k + nodeNeighborOffset[currentNode];
                    isEdgeAllocated[h_G[0][currentNode] + k + nodeNeighborOffset[currentNode]] = 1;
                }
                nodeNeighborOffset[currentNode] += K2 - j;
                currentEdge++;
                j = K2;
            } else {
                printf("    Allocate %d outgoing Edges from the node %d from index %d\n", (numNeighbors - nodeNeighborOffset[currentNode]), currentNode, threadIndex);
                for (int k = 0; k < (numNeighbors - nodeNeighborOffset[currentNode]) && threadIndex < NumEdges; k++) {
                    h_allocEdgesToThreads1[threadIndex++] = h_G[0][currentNode] + k + nodeNeighborOffset[currentNode];
                    isEdgeAllocated[h_G[0][currentNode] + k + nodeNeighborOffset[currentNode]] = 1;
                }
                j += numNeighbors - nodeNeighborOffset[currentNode];
                nodeNeighborOffset[currentNode] = numNeighbors;
                currentEdge++;
            }*/
        }
//        threadIndex++;
#ifdef ALLOC_K2EXACTEDGES
// If K2 outgoing edges are not available, then select any edges for K2.
        if (j < K2) {
            for (; j < K2 && currentEdge < NumEdges;) {
                if (!isEdgeAllocated[currentEdge]) {
                    h_allocEdgesToThreads1[threadIndex++] = currentEdge;
                    isEdgeAllocated[currentEdge] = 1;
                    j++;
                }
                currentEdge++;
            }
        }
#endif
        currentThread++;
    }
    h_allocEdgesToThreads0[currentThread] = threadIndex;
#ifdef ALLOC_K2EXACTEDGES
    ;
#else
    numThreadsReq = currentThread;
#endif

    /*printf("ALlocated Edges per thread");
    for (int i = 0; i < numThreadsReq; i++) {
        printf("Thread: %d ## Allocated Edges = ", i);
        for (int j = h_allocEdgesToThreads0[i]; j < h_allocEdgesToThreads0[i + 1]; j++) {
            printf("%d ", h_allocEdgesToThreads1[j]);
        }
        printf("\n");
    }*/

    err = cudaMalloc((void **)&allocEdgesToThreads0, (numThreadsReq + 1) * sizeof(int));
    CUDA_ERR_CHECK;
    err = cudaMalloc((void **)&allocEdgesToThreads1, (NumEdges + 1) * sizeof(int));
    CUDA_ERR_CHECK;
    err = cudaMemcpy(allocEdgesToThreads0, h_allocEdgesToThreads0, (numThreadsReq + 1) * sizeof(int), cudaMemcpyHostToDevice);
    CUDA_ERR_CHECK;
    err = cudaMemcpy(allocEdgesToThreads1, h_allocEdgesToThreads1, (NumEdges + 1) * sizeof(int), cudaMemcpyHostToDevice);
    CUDA_ERR_CHECK;

    //printf("Device print Allocated Edges per thread\n");
    //printAllocThreads<<<numBlocks, maxThreadsPerBlock>>>(numThreadsReq, allocEdgesToThreads0, allocEdgesToThreads1);
}

void sssp_CPU(int* G0, int* G1, int * dist, int * len, int root) {

    {
        fin = false;
        cudaOccupancyMaxPotentialBlockSize(&gm_minGridSize, &gm_blockSize,forEachKernel0, 0, 0);
            gm_minGridSize = 1024;
        gm_gridSize = (NumNodes + 1 + gm_blockSize - 1) / gm_blockSize;
        gm_numBlocksStillToProcess = gm_gridSize, gm_offsetIntoBlocks = 0;
        while (gm_numBlocksStillToProcess > 0) {
            if (gm_numBlocksStillToProcess > gm_minGridSize)
                gm_numBlocksKernelParameter = gm_minGridSize;
            else
                gm_numBlocksKernelParameter = gm_numBlocksStillToProcess;
            forEachKernel0<<<gm_numBlocksKernelParameter, gm_blockSize>>>(G0, G1, NumNodes, NumEdges, edgeFrom, dist, len, root, updated, dist_nxt, updated_nxt, gm_offsetIntoBlocks);
            CUDA_ERR_CHECK;
            gm_numBlocksStillToProcess -= gm_minGridSize;
            gm_offsetIntoBlocks += gm_minGridSize * gm_blockSize;
        }
        while ( !fin)
        {
            fin = true;
            h___E8 = false;
            err = cudaMemcpyToSymbol(__E8, &h___E8, sizeof(bool), 0, cudaMemcpyHostToDevice);
            CUDA_ERR_CHECK;
            cudaOccupancyMaxPotentialBlockSize(&gm_minGridSize, &gm_blockSize,forEachKernel1, 0, 0);
            gm_minGridSize = 1024;
            gm_gridSize = (NumEdges + gm_blockSize - 1) / gm_blockSize;
            gm_numBlocksStillToProcess = gm_gridSize, gm_offsetIntoBlocks = 0;
            int iteration = 0;
            while (gm_numBlocksStillToProcess > 0) {
                if (gm_numBlocksStillToProcess > gm_minGridSize)
                    gm_numBlocksKernelParameter = gm_minGridSize;
                else
                    gm_numBlocksKernelParameter = gm_numBlocksStillToProcess;
                //printf("Kernel para = (%d, %d)\n", gm_numBlocksKernelParameter, gm_blockSize);
                forEachKernel1<<<gm_numBlocksKernelParameter, gm_blockSize>>>(G0, G1, NumNodes, NumEdges, edgeFrom, dist, len, root, updated, dist_nxt, updated_nxt, gm_offsetIntoBlocks, allocEdgesToThreads0, allocEdgesToThreads1, numThreadsReq);
                CUDA_ERR_CHECK;
                gm_numBlocksStillToProcess -= gm_minGridSize;
                gm_offsetIntoBlocks += gm_minGridSize * gm_blockSize;
                iteration++;
            }
            //printf("# Iterations = %d\n", iteration);
            /*int blockSize = 1024;
            int numBlocks = (NumEdges - 1) / blockSize + 1;
            printf("Kernel para = (%d, %d)\n", numBlocks, blockSize);
            forEachKernel1<<<numBlocks, blockSize>>>(G0, G1, NumNodes, NumEdges, edgeFrom, dist, len, root, updated, dist_nxt, updated_nxt, gm_offsetIntoBlocks, allocEdgesToThreads0, allocEdgesToThreads1, numThreadsReq);
            CUDA_ERR_CHECK;*/

            cudaOccupancyMaxPotentialBlockSize(&gm_minGridSize, &gm_blockSize,forEachKernel2, 0, 0);
            gm_minGridSize = 1024;
            gm_gridSize = (NumNodes + 1 + gm_blockSize - 1) / gm_blockSize;
            gm_numBlocksStillToProcess = gm_gridSize, gm_offsetIntoBlocks = 0;
            while (gm_numBlocksStillToProcess > 0) {
                if (gm_numBlocksStillToProcess > gm_minGridSize)
                    gm_numBlocksKernelParameter = gm_minGridSize;
                else
                    gm_numBlocksKernelParameter = gm_numBlocksStillToProcess;
                forEachKernel2<<<gm_numBlocksKernelParameter, gm_blockSize>>>(G0, G1, NumNodes, NumEdges, edgeFrom, dist, len, root, dist_nxt, updated, updated_nxt, gm_offsetIntoBlocks);
                CUDA_ERR_CHECK;
                gm_numBlocksStillToProcess -= gm_minGridSize;
                gm_offsetIntoBlocks += gm_minGridSize * gm_blockSize;
            }
            err = cudaMemcpyFromSymbol(&h___E8, __E8, sizeof(bool), 0, cudaMemcpyDeviceToHost);
            CUDA_ERR_CHECK;
            fin =  !h___E8;
        }
    }
}


using namespace std;
// sssp -? : for how to run generated main program
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
    cudaSetDevice(7);
    inputFile >> NumNodes >> NumEdges;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    err = cudaMalloc((void **)&G0, (NumNodes + 2) * sizeof(int));
    CUDA_ERR_CHECK;
    err = cudaMalloc((void **)&G1, (NumEdges + 1) * sizeof(int));
    CUDA_ERR_CHECK;
    err = cudaMalloc((void **)&dist, (NumNodes + 1) * sizeof(int));
    CUDA_ERR_CHECK;
    err = cudaMalloc((void **)&dist_nxt, (NumNodes + 1) * sizeof(int));
    CUDA_ERR_CHECK;
    err = cudaMalloc((void **)&len, (NumEdges + 1) * sizeof(int));
    CUDA_ERR_CHECK;
    err = cudaMalloc((void **)&updated, (NumNodes + 1) * sizeof(bool));
    CUDA_ERR_CHECK;
    err = cudaMalloc((void **)&updated_nxt, (NumNodes + 1) * sizeof(bool));
    CUDA_ERR_CHECK;
    err = cudaMalloc((void **)&edgeFrom, (NumEdges + 1) * sizeof(int));
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

    clock_t cpuStart, cpuEnd;
    cpuStart = clock();
    /*cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);*/
    
    numThreadsReq = (NumEdges - 1) / K1 + 1;
    //allocateThreads(h_G);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    
    cpuEnd = clock();
    elapsedTime = ((double) (cpuEnd - cpuStart)) / CLOCKS_PER_SEC;
/*    printf("Wall Time = %d\n", wall1 - wall0);
    printf("CPU Time  = %d\n", cpu1  - cpu0);
*/
    printf("Allocating Threads time(milliseconds)  = %f\n", elapsedTime);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    sssp_CPU(G0, G1, dist, len, root);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Execution time(milliseconds)  = %f\n", elapsedTime);
    bool gm_verify = verifysssp(h_G);
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
    err = cudaFree(dist);
    CUDA_ERR_CHECK;
    err = cudaFree(dist_nxt);
    CUDA_ERR_CHECK;
    err = cudaFree(len);
    CUDA_ERR_CHECK;
    err = cudaFree(updated);
    CUDA_ERR_CHECK;
    err = cudaFree(updated_nxt);
    CUDA_ERR_CHECK;
    err = cudaFree(host_threadBlockBarrierReached);
    CUDA_ERR_CHECK;
    free(h_G[0]);
    free(h_G[1]);
    return 0;
}
