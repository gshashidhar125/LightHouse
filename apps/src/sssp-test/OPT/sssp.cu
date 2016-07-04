
ssspMacroGPU;
#include "GlobalBarrier.cuh"
__global__ void printAllocThreads(int numThreads, int* allocEdgesToThreads0, int* allocEdgesToThreads1) {

    int tId = blockIdx.x * blockDim.x + threadIdx.x;// + gm_offsetIntoBlocks;
    if (tId >= numThreads)
        return;
//    printf("Thread: %d ## Allocated Edges = ", tId);
    for (int j = allocEdgesToThreads0[tId]; j < allocEdgesToThreads0[tId + 1]; j++) {
        printf("Thread: %d ## Allocated Edges = %d\n", tId, allocEdgesToThreads1[j]);
    }
    printf("\n");
}

__global__ void forEachKernel0 (int *G0, int *G1, int NumNodes, int NumEdges, int *edgeFrom, int * dist, int * len, int root, bool * updated, int * dist_nxt, bool * updated_nxt, int gm_offsetIntoBlocks) {
    kernelMacro0;
    int tId = blockIdx.x * blockDim.x + threadIdx.x + gm_offsetIntoBlocks;
    if (tId >= NumNodes + 1) {
        return;
    }
    t0 = tId;
    {
        dist[t0] = (t0 == root)?0:2147483647;//99999;
        updated[t0] = (t0 == root)?true:false;
        dist_nxt[t0] = dist[t0];
        updated_nxt[t0] = updated[t0];
    }
}
__global__ void forEachKernel1 (int *G0, int *G1, int NumNodes, int NumEdges, int *edgeFrom, int * dist, int * len, int root, bool * updated, int * dist_nxt, bool * updated_nxt, int gm_offsetIntoBlocks, int* allocEdgesToThreads0, int* allocEdgesToThreads1, int numThreadsReq) {
    kernelMacro1;
    int tId = blockIdx.x * blockDim.x + threadIdx.x + gm_offsetIntoBlocks;
    /*if (tId >= NumEdges) {
        return;
    }*/
    if (updated[edgeFrom[tId]] && tId < NumEdges/*numThreadsReq*/) {
//        for (EdgeIter = allocEdgesToThreads0[tId]; EdgeIter < allocEdgesToThreads0[tId + 1]; EdgeIter++) {
//        for (EdgeIter = tId * K1; EdgeIter < ((tId + 1) * K1) && EdgeIter < NumEdges; EdgeIter++) {
            EdgeIter = tId;
            n = edgeFrom[EdgeIter];;
            s = G1[EdgeIter];
//            if (updated[n])
            {
                {
                    e = EdgeIter;
//                    localExpr = dist_nxt[s];
                    expr = dist[n] + len[e];
                    atomicMin(&dist_nxt[s], expr);
/*                    if (localExpr > expr) {
                        updated_nxt[s] = true;
                    }*/
                }
//                updated[n] = false;
//            }
        }
    }
}
__global__ void forEachKernel2 (int *G0, int *G1, int NumNodes, int NumEdges, int *edgeFrom, int * dist, int * len, int root, int * dist_nxt, bool * updated, bool * updated_nxt, int gm_offsetIntoBlocks) {
    kernelMacro2;
    int tId = blockIdx.x * blockDim.x + threadIdx.x + gm_offsetIntoBlocks;
    /*if (tId >= NumNodes + 1) {
        return;
    }*/
    if (dist_nxt[tId] < dist[tId] && tId <= NumNodes) {
    t4 = tId;
    {
        dist[t4] = dist_nxt[t4];
//        updated[t4] = updated_nxt[t4];
//        updated_nxt[t4] = false;
        updated[t4] = true;
//        if (updated[t4])
            __E8 = true;
    }
    } else if (tId <= NumNodes) {
        t4 = tId;
        updated[t4] = false;
    }
}
