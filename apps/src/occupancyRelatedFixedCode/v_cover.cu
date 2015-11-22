
v_coverMacroGPU;
#include "GlobalBarrier.cuh"
__global__ void forEachKernel0 (int *G0, int *G1, int NumNodes, int NumEdges, bool * selectEdge, int * Deg, bool * Covered, int offset) {
    kernelMacro0;
    int tId = blockIdx.x * blockDim.x + threadIdx.x + offset;
    if (tId >= NumNodes) {
        return;
    }
    t0 = tId;
    //printf("TId = %d. Reached Barrier One\n", tId);
    if (threadIdx.x == 0) {
        printf("BlockId: %d. Reached Value = %d\n", blockIdx.x, gm_threadBlockBarrierReached[blockIdx.x]);
    }
    softwareBarrier();

    if (threadIdx.x == 0) {
        printf("Blockid = %d. After Barrier One\n", blockIdx.x);
        printf("BlockId: %d. Reached Value = %d\n", blockIdx.x, gm_threadBlockBarrierReached[blockIdx.x]);
    }
    {
        Deg[t0] = G0[t0 + 1] - G0[t0] + G0[t0 + 1] - G0[t0];
        Covered[t0] = false;
    }
}
__global__ void forEachKernel1 (int *G0, int *G1, int NumNodes, int NumEdges, bool * selectEdge) {
    kernelMacro1;
    int tId = blockIdx.x * blockDim.x + threadIdx.x;
    if (tId >= NumEdges) {
        return;
    }
    t2 = tId;
    selectEdge[t2] = false;
}
__global__ void forEachKernel2 (int *G0, int *G1, int NumNodes, int NumEdges, bool * selectEdge, bool * Covered, int * Deg) {
    kernelMacro2;
    int tId = blockIdx.x * blockDim.x + threadIdx.x;
    if (tId >= NumNodes) {
        return;
    }
    s = tId;
    {
        for (int iter = G0[s], t = G1[iter]; iter != G0[s + 1]; iter++, t = G1[iter]) {
            if ( !(Covered[s] && Covered[t]))
            {
                expr = Deg[s] + Deg[t];
                atomicMax(&max_val, expr);
                if (localExpr <= expr) {
                    localExpr = expr;
                    localfrom = s;
                    localto = t;
                    locale = iter;
                }
            }
        }
    }
    //printf("TId = %d. Reached Barrier One\n", tId);
    if (threadIdx.x == 0) {
        printf("Reached Value = %d\n", gm_threadBlockBarrierReached[blockIdx.x]);
    }
    softwareBarrier();
    if (threadIdx.x == 0) {
        printf("Reached Value = %d\n", gm_threadBlockBarrierReached[blockIdx.x]);
    }
    printf("Tid = %d. After Barrier One\n", tId);
    return;
    if (localExpr == max_val)
        chooseThread = tId;
    printf("TId = %d. Reached Barrier Second\n", tId);
    softwareBarrier();
    if (chooseThread == tId)
    {
        from = localfrom;
        to = localto;
        e = locale;
    }
}
__global__ void forEachKernel3 (int *G0, int *G1, int NumNodes, int NumEdges, bool * selectEdge, bool * Covered) {
    kernelMacro3;
    int tId = blockIdx.x * blockDim.x + threadIdx.x;
    if (tId >= NumNodes) {
        return;
    }
    t3 = tId;
    if (Covered[t3])
        atomicAdd(&__S4, 1);
}
