#ifndef GM_GENERATED_CUDA_HOP_DIST_H
#define GM_GENERATED_CUDA_HOP_DIST_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <float.h>

#define kernelMacro0 \
int t0;   \

#define kernelMacro1 \
int n, s, localExpr = 999999 , expr;   \

#define kernelMacro2 \
int t4;   \

#define hop_distMacro \
int*  G0, * G1 , NumNodes , NumEdges ;   \
int32_t* dist, *dist_nxt;   \
int root;   \
bool* updated, *updated_nxt;   \
bool fin, h___E8;   \
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

#define hop_distMacroGPU \
__device__ bool __E8;   \
__device__ bool* gm_threadBlockBarrierReached;  \


#endif
