#ifndef GM_GENERATED_CUDA_SSSP_H
#define GM_GENERATED_CUDA_SSSP_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <float.h>

#define kernelMacro0 \
int t0;   \

#define kernelMacro1 \
int n, EdgeIter, s, e, localExpr = 2147483647, expr;   \

#define kernelMacro2 \
int t4;   \

#define ssspMacro \
int*  G0, * G1 , NumNodes , NumEdges , *edgeFrom;   \
int32_t* dist, *dist_nxt;   \
int32_t* len;   \
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
int *allocEdgesToThreads0, *allocEdgesToThreads1, numThreadsReq; 

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

#define ssspMacroGPU \
__device__ bool __E8;   \
__device__ bool* gm_threadBlockBarrierReached;  \


#define K1 2
#define K2 4
#endif
