#include <time.h>
#include <stdlib.h>
#include <fstream>

using namespace std;
int*  G0, * G1 , NumNodes , NumEdges , *edgeFrom;
int* dist, *dist_nxt;
int* len;   
int root;   
bool* updated, *updated_nxt;   
bool fin, h___E8;   
bool* host_threadBlockBarrierReached;  
#define K1 10
#define K2 1
void allocateThreads(int* h_G[2]) {
    int maxThreadsPerBlock =  1024;//prop.maxThreadsPerBlock;
    int numThreadsReq = (NumEdges - 1) / (K1 + K2) + 1;
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
        currentThread++;
    }
    h_allocEdgesToThreads0[currentThread] = threadIndex;

    printf("ALlocated Edges per thread");
    for (int i = 0; i < numThreadsReq; i++) {
        printf("Thread: %d ## Allocated Edges = ", i);
        for (int j = h_allocEdgesToThreads0[i]; j < h_allocEdgesToThreads0[i + 1]; j++) {
            printf("%d ", h_allocEdgesToThreads1[j]);
        }
        printf("\n");
    }
}
/*
void sssp_CPU(int* G0, int* G1, int * dist, int * len, int root) {

    {
        fin = false;
        cudaOccupancyMaxPotentialBlockSize(&gm_minGridSize, &gm_blockSize,forEachKernel0, 0, 0);
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
            gm_gridSize = (NumEdges + gm_blockSize - 1) / gm_blockSize;
            gm_numBlocksStillToProcess = gm_gridSize, gm_offsetIntoBlocks = 0;
            while (gm_numBlocksStillToProcess > 0) {
                if (gm_numBlocksStillToProcess > gm_minGridSize)
                    gm_numBlocksKernelParameter = gm_minGridSize;
                else
                    gm_numBlocksKernelParameter = gm_numBlocksStillToProcess;
                forEachKernel1<<<gm_numBlocksKernelParameter, gm_blockSize>>>(G0, G1, NumNodes, NumEdges, edgeFrom, dist, len, root, updated, dist_nxt, updated_nxt, gm_offsetIntoBlocks);
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
}*/

int populate(char *fileName, int* row[2]) {

    ifstream inputFile;
    inputFile.open(fileName);
    if (!inputFile.is_open()){
        printf("invalid file");
        exit(1);
    }

    inputFile >> NumNodes >> NumEdges;
 
    row[0] = new int [NumNodes + 2]();
    row[1] = new int [NumEdges + 1]();
    
    // for v_cover
    //bool* edgeProp = new bool [NumEdges + 1]();
    // for SSSP
    int* edgeProp = new int [NumEdges + 1]();
    int* parent = new int[NumEdges + 1]();
    
    int i, j, k;
    // For v_cover
    //bool l;
    //For sssp
    int l;

    i = NumEdges;
    int lastj = 0, currentIndex = 0;
    while(i > 0) {

        // For v_cover, SSSP
        inputFile >> j >> k >> l;
        while (lastj <= j || lastj == 0) {
            if (lastj == 0) {
                row[0][0] = currentIndex;
                row[0][1] = currentIndex;
            }else {
                row[0][lastj] = currentIndex;
            }
            lastj++;
        }
        row[1][currentIndex] = k;
        parent[currentIndex] = j;
        // For v_cover
        // For SSSP
        edgeProp[currentIndex] = l;
        currentIndex ++;
        i--;
    }
    // Sentinel node just points to the end of the last node in the graph
    while (lastj <= NumNodes + 1) {
        row[0][lastj] = currentIndex;
        lastj++;
    }
    /*for (i = 0; i <= NumNodes + 1; i++)
        printf("Vertex: %d = %d\n", i, row[0][i]);
 
    printf("Second Array:\n");
    for (i = 0; i <= NumEdges; i++)
        printf("Edges: Index: %d, Value = %d\n", i, row[1][i]);*/

    // For v_cover
    //err = cudaMemcpy(selectEdge, edgeProp, (NumEdges + 1) * sizeof(bool), cudaMemcpyHostToDevice);
    //CUDA_ERR_CHECK;
    // For SSSP
    root = 1;
    
    /*printf("\n Parent Array:\n");
    for (int i = 0; i <= NumEdges; i++) {
        printf("%d ", parent[i]);
    }*/
    delete edgeProp;
    //printGraph(row);
    //printGraphOnDevice<<<1, 1>>>(G0, G1, NumNodes, NumEdges);
    //CUDA_ERR_CHECK;
    delete parent;
   
    return 0;
}

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
    inputFile >> NumNodes >> NumEdges;
    int* h_G[2];
    printf("Graph Population began\n");
    populate(argv[1], h_G);
    printf("Graph Population end\n");
    allocateThreads(h_G);
    //sssp_CPU(G0, G1, dist, len, root);

    bool gm_verify;// = verifysssp(h_G);
    if (!gm_verify) {
        printf("Verification Failed\n");
        return -1;
    } else {
        printf("Verification Success\n");
    }
    free(h_G[0]);
    free(h_G[1]);
}
