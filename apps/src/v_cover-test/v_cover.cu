
v_coverMacroGPU;
#include "GlobalBarrier.cuh"
__global__ void forEachKernel0 (int *G0, int *G1, int NumNodes, int NumEdges, bool * selectEdge, int * Deg, bool * Covered, int gm_offsetIntoBlocks) {
    kernelMacro0;
    int tId = blockIdx.x * blockDim.x + threadIdx.x + gm_offsetIntoBlocks;
    if (tId >= NumNodes + 1) {
        return;
    }
    t0 = tId;
    {
        Deg[t0] = G0[t0 + 1] - G0[t0] + G0[t0 + 1] - G0[t0];
        Covered[t0] = false;
    }
}
__global__ void forEachKernel1 (int *G0, int *G1, int NumNodes, int NumEdges, bool * selectEdge, int gm_offsetIntoBlocks) {
    kernelMacro1;
    int tId = blockIdx.x * blockDim.x + threadIdx.x + gm_offsetIntoBlocks;
    if (tId >= NumEdges) {
        return;
    }
    t2 = tId;
    selectEdge[t2] = false;
}
__global__ void forEachKernel2 (int *G0, int *G1, int NumNodes, int NumEdges, bool * selectEdge, bool * Covered, int * Deg, int gm_offsetIntoBlocks) {
    kernelMacro2;
    int tId = blockIdx.x * blockDim.x + threadIdx.x + gm_offsetIntoBlocks;
    if (tId >= NumNodes + 1) {
        return;
    }
    s = tId;
    {
        for (int iter = G0[s], t = G1[iter]; iter < G0[s + 1]; iter++, t = G1[iter]) {
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
    softwareBarrier();
    if (localExpr == max_val)
        chooseThread = tId;
    softwareBarrier();
    if (chooseThread == tId)
    {
        from = localfrom;
        to = localto;
        e = locale;
    }
}
__global__ void forEachKernel3 (int *G0, int *G1, int NumNodes, int NumEdges, bool * selectEdge, bool * Covered, int gm_offsetIntoBlocks) {
    kernelMacro3;
    int tId = blockIdx.x * blockDim.x + threadIdx.x + gm_offsetIntoBlocks;
    if (tId >= NumNodes + 1) {
        return;
    }
    t3 = tId;
    if (Covered[t3])
        atomicAdd(&__S4, 1);
}
