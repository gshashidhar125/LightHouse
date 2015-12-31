
conductMacroGPU;
#include "GlobalBarrier.cuh"
__global__ void forEachKernel0 (int *G0, int *G1, int NumNodes, int NumEdges, int * member, int num, int gm_offsetIntoBlocks) {
    kernelMacro0;
    int tId = blockIdx.x * blockDim.x + threadIdx.x + gm_offsetIntoBlocks;
    if (tId >= NumNodes + 1) {
        return;
    }
    u = tId;
    if ((member[u] == num))
        atomicAdd(&__S2, G0[u + 1] - G0[u]);
}
__global__ void forEachKernel1 (int *G0, int *G1, int NumNodes, int NumEdges, int * member, int num, int gm_offsetIntoBlocks) {
    kernelMacro1;
    int tId = blockIdx.x * blockDim.x + threadIdx.x + gm_offsetIntoBlocks;
    if (tId >= NumNodes + 1) {
        return;
    }
    u0 = tId;
    if ((member[u0] != num))
        atomicAdd(&__S3, G0[u0 + 1] - G0[u0]);
}
__global__ void forEachKernel2 (int *G0, int *G1, int NumNodes, int NumEdges, int * member, int num, int gm_offsetIntoBlocks) {
    kernelMacro2;
    int tId = blockIdx.x * blockDim.x + threadIdx.x + gm_offsetIntoBlocks;
    if (tId >= NumNodes + 1) {
        return;
    }
    u1 = tId;
    if ((member[u1] == num))
    {
        for (int iter = G0[u1], j = G1[iter]; iter < G0[u1 + 1]; iter++, j = G1[iter]) if ((member[j] != num))
            atomicAdd(&__S4, 1);
    }
}
