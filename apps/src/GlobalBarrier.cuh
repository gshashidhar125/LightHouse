__device__ __forceinline__ void softwareBarrier() {

    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
        gm_threadBlockBarrierReached[blockIdx.x] = 1;
        __threadfence();
    }

    if (blockIdx.x == 0) {

        __syncthreads();

        for (int i = threadIdx.x; i < gridDim.x; i += blockDim.x) {
        
            while (gm_threadBlockBarrierReached[i] == 0)
                __threadfence();
            gm_threadBlockBarrierReached[i] = 0;
            __threadfence();
        }
        __syncthreads();
    } else {

        if (threadIdx.x == 0) {
            while (gm_threadBlockBarrierReached[blockIdx.x] == 1)
                __threadfence();
        }
        __syncthreads();
    }
}
