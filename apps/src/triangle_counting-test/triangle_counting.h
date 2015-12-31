#ifndef GM_GENERATED_CUDA_TRIANGLE_COUNTING_H
#define GM_GENERATED_CUDA_TRIANGLE_COUNTING_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <float.h>

#define kernelMacro0 \
int u, v, w;   \

#define triangle_countingMacro \
int*  G0, * G1 , NumNodes , NumEdges, *h_inNode ;   \
unsigned long long int h_T;   \
bool* host_threadBlockBarrierReached;  \
cudaError_t err;  \
cudaEvent_t start, stop;
float elapsedTime;
int gm_blockSize;   /* The launch configurator returned block size*/ \
int gm_minGridSize; /* The minimum grid size needed to achieve the*/ \
           /* maximum occupancy for a full device launch*/ \
int gm_gridSize;    /* The actual grid size needed, based on input size*/ \
int gm_numBlocksStillToProcess, gm_offsetIntoBlocks; \
int gm_numBlocksKernelParameter; \

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

#define CUDA_ERR_CHECK  \
gpuErrchk(cudaPeekAtLastError());\
gpuErrchk(cudaDeviceSynchronize());\

__device__ bool hasEdgeTo(int* gm_G0, int* gm_G1, int gm_from, int gm_to) {
    for (int i = gm_G0[gm_from]; i < gm_G0[gm_from + 1]; i++) {
        if (gm_G1[i] == gm_to) {
            return true;
        }
    }
    return false;
}
#define triangle_countingMacroGPU \
__device__ unsigned long long int T;   \
__device__ bool* gm_threadBlockBarrierReached;  \


#endif
