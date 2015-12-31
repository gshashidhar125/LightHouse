#ifndef GM_GENERATED_CUDA_V_COVER_H
#define GM_GENERATED_CUDA_V_COVER_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <float.h>
#include <limits.h>
#include <cmath>
#include <algorithm>

#define kernelMacro0 \
int t0;   \

#define kernelMacro1 \
int t2;   \

#define kernelMacro2 \
int s, t, expr, localExpr = 0 , localfrom, localto, locale;   \

#define kernelMacro3 \
int t3;   \

#define v_coverMacro \
int*  G0, * G1 , NumNodes , NumEdges ;   \
bool* selectEdge;   \
int32_t* Deg;   \
bool* Covered;   \
int remain, h___S4, h_max_val, h_from, h_to, h_e;   \
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

#define v_coverMacroGPU \
__device__ int chooseThread, __S4, max_val, from, to, e;   \
__device__ bool* gm_threadBlockBarrierReached;  \


#endif
