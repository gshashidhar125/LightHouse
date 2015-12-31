
ssspMacroGPU;
#include "GlobalBarrier.cuh"
__global__ void forEachKernel0 (int *G0, int *G1, int NumNodes, int NumEdges, int * dist, int * len, int root, bool * updated, int * dist_nxt, bool * updated_nxt, int gm_offsetIntoBlocks) {
    kernelMacro0;
    int tId = blockIdx.x * blockDim.x + threadIdx.x + gm_offsetIntoBlocks;
    if (tId >= NumNodes + 1) {
        return;
    }
    t0 = tId;
    {
        dist[t0] = (t0 == root)?0:2147483647;
        updated[t0] = (t0 == root)?true:false;
        dist_nxt[t0] = dist[t0];
        updated_nxt[t0] = updated[t0];
    }
}
__global__ void forEachKernel1 (int *G0, int *G1, int NumNodes, int NumEdges, int * dist, int * len, int root, bool * updated, int * dist_nxt, bool * updated_nxt, int gm_offsetIntoBlocks) {
    kernelMacro1;
    int tId = blockIdx.x * blockDim.x + threadIdx.x + gm_offsetIntoBlocks;
    if (tId >= NumNodes + 1) {
        return;
    }
    n = tId;
    {
        if (updated[n])
        {
            for (int iter = G0[n], s = G1[iter]; iter < G0[n + 1]; iter++, s = G1[iter]) {
                e = iter;
                localExpr = dist_nxt[s];
                expr = dist[n] + len[e];
                if (localExpr > expr) {
                    atomicMin(&dist_nxt[s], expr);
                    updated_nxt[s] = true;
                }
                /*if (localExpr > expr) {
                    dist_nxt[s] = expr;
                    updated_nxt[s] = true;
                }*/
            }
        }
    }
}
__global__ void forEachKernel2 (int *G0, int *G1, int NumNodes, int NumEdges, int * dist, int * len, int root, int * dist_nxt, bool * updated, bool * updated_nxt, int gm_offsetIntoBlocks) {
    kernelMacro2;
    int tId = blockIdx.x * blockDim.x + threadIdx.x + gm_offsetIntoBlocks;
    if (tId >= NumNodes + 1) {
        return;
    }
    t4 = tId;
    {
        dist[t4] = dist_nxt[t4];
        updated[t4] = updated_nxt[t4];
        updated_nxt[t4] = false;
        atomicOr((int *)&__E8, (int)updated[t4]);
    }
}
