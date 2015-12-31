#ifndef GM_GENERATED_CUDA_CONDUCT_H
#define GM_GENERATED_CUDA_CONDUCT_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <float.h>

#define kernelMacro0 \
int u;   \

#define kernelMacro1 \
int u0;   \

#define kernelMacro2 \
int u1, j;   \

#define conductMacro \
int*  G0, * G1 , NumNodes , NumEdges ;   \
int32_t* member;   \
int num, h___S2, h___S3, h___S4;   \
float m;   \
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

#define conductMacroGPU \
__device__ int __S2, __S3, __S4;   \
__device__ bool* gm_threadBlockBarrierReached;  \


#endif
