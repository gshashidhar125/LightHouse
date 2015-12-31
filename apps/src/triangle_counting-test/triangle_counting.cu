
triangle_countingMacroGPU;
#include "GlobalBarrier.cuh"
__global__ void forEachKernel0 (int *G0, int *G1, int NumNodes, int NumEdges, int gm_offsetIntoBlocks, int* inNode) {
    kernelMacro0;
    int tId = blockIdx.x * blockDim.x + threadIdx.x + gm_offsetIntoBlocks;
    if (tId >= NumNodes + 1) {
        return;
    }
    v = tId;
    for (int iter = G0[v], u = G1[iter]; iter < G0[v + 1]; iter++, u = G1[iter]) {
        if (u > v)
        {
            for (int iter = G0[v], w = G1[iter]; iter < G0[v + 1]; iter++, w = G1[iter]) {
                if (w > u)
                {
                    if (hasEdgeTo(G0, G1, w, u))
                        atomicAdd(&T, 1);
                }
            }
        }
    }
}
