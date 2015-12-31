
random_bipartite_matchingMacroGPU;
#include "GlobalBarrier.cuh"
__global__ void forEachKernel0 (int *G0, int *G1, int NumNodes, int NumEdges, bool * isLeft, int * Match, int * Suitor, int gm_offsetIntoBlocks) {
    kernelMacro0;
    int tId = blockIdx.x * blockDim.x + threadIdx.x + gm_offsetIntoBlocks;
    if (tId >= NumNodes + 1) {
        return;
    }
    t0 = tId;
    {
        Match[t0] = -1;
        Suitor[t0] = -1;
    }
}
__global__ void forEachKernel1 (int *G0, int *G1, int NumNodes, int NumEdges, bool * isLeft, int * Match, int * Suitor, int gm_offsetIntoBlocks) {
    kernelMacro1;
    int tId = blockIdx.x * blockDim.x + threadIdx.x + gm_offsetIntoBlocks;
    if (tId >= NumNodes + 1) {
        return;
    }
    n = tId;
    {
        if (isLeft[n] && (Match[n] == -1))
        {
            for (int iter = G0[n], t = G1[iter]; iter < G0[n + 1]; iter++, t = G1[iter]) {
                if (Match[t] == -1)
                {
                    Suitor[t] = n;
                    atomicAnd((int *)&finished, (int)false);
                    //if (!false)
                    //    finished = false;
                }
            }
        }
    }
}
__global__ void forEachKernel2 (int *G0, int *G1, int NumNodes, int NumEdges, bool * isLeft, int * Match, int * Suitor, int gm_offsetIntoBlocks) {
    kernelMacro2;
    int tId = blockIdx.x * blockDim.x + threadIdx.x + gm_offsetIntoBlocks;
    if (tId >= NumNodes + 1) {
        return;
    }
    t2 = tId;
    {
        if ( !isLeft[t2] && (Match[t2] == -1))
        {
            if (Suitor[t2] != -1)
            {
                n3 = Suitor[t2];
                Suitor[n3] = t2;
                Suitor[t2] = -1;
            }
        }
    }
}
__global__ void forEachKernel3 (int *G0, int *G1, int NumNodes, int NumEdges, bool * isLeft, int * Match, int * Suitor, int gm_offsetIntoBlocks) {
    kernelMacro3;
    int tId = blockIdx.x * blockDim.x + threadIdx.x + gm_offsetIntoBlocks;
    if (tId >= NumNodes + 1) {
        return;
    }
    n4 = tId;
    {
        if (isLeft[n4] && (Match[n4] == -1))
        {
            if (Suitor[n4] != -1)
            {
                t5 = Suitor[n4];
                Match[n4] = t5;
                Match[t5] = n4;
                atomicAdd(&count, 1);
            }
        }
    }
}
