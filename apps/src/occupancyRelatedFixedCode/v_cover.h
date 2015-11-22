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
__device__ bool* gm_threadBlockBarrierSignal;  \


#endif
